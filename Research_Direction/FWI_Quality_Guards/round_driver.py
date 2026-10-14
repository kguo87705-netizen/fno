"""Authorized first round only, with an absolute eight-hour deadline."""
import subprocess,shutil
from common import *

config=json.loads((BUILD/'config.json').read_text())
deadline=config['deadline_unix']
events=[]

def job(script,args,name,limit=3600):
    remaining=deadline-time.time()
    if remaining<120:raise RuntimeError('STOP eight-hour round budget')
    t=time.time();log=BUILD/(name+'.log')
    with log.open('w',encoding='utf-8') as h:
        p=subprocess.Popen([sys.executable,str(HERE/script),*args],stdout=h,stderr=subprocess.STDOUT)
        try:rc=p.wait(timeout=min(limit,remaining-30))
        except subprocess.TimeoutExpired:
            p.kill();p.wait();raise RuntimeError('STOP time budget: '+name)
    event=dict(name=name,seconds=time.time()-t,returncode=rc);events.append(event)
    dump(BUILD/'round_events.json',events)
    print('JOB_FINISHED',json.dumps(event),flush=True)
    if rc:raise RuntimeError('STOP job failure: '+name)

def reuse_probe():
    src=BUILD/'serial_probe/seed30_epoch0/validation_000'
    for dest in ['feasibility','calibration']:
        path=BUILD/dest/'seed30_epoch0/validation_000';path.mkdir(parents=True,exist_ok=True)
        for p in src.iterdir():shutil.copy2(p,path/p.name)

try:
    job('correct_geometry_margins.py',[],'continuous_geometry_correction',600)
    job('runtime_probe.py',[],'runtime_probe',1200)
    # Physical and FNO configurations are independent; retain individual OOMs.
    # Two actual OOMs stop this task, and later tasks remain independent of scale timing.
    oom=0;scale_stopped=False
    for scale in range(3):
        if scale_stopped:break
        for kind in ['physical_forward','physical_forward_adjoint','fno_forward','fno_forward_backward']:
            job('benchmark_scale.py',['--scale',str(scale),'--kind',kind],f'benchmark_{scale}_{kind}',1800)
            result=json.loads((BUILD/'scale'/f'scale{scale}_{kind}.json').read_text())
            if result['status']=='OOM':
                oom+=1
                # FNO full-output OOM: record a same-structure receiver-output attempt.
                if kind.startswith('fno') and oom<2:
                    receiver=kind.replace('fno_','fno_receiver_')
                    job('benchmark_scale.py',['--scale',str(scale),'--kind',receiver],f'benchmark_{scale}_{receiver}',1800)
                    r=json.loads((BUILD/'scale'/f'scale{scale}_{receiver}.json').read_text())
                    oom+=int(r['status']=='OOM')
                if oom>=2:
                    scale_stopped=True;dump(BUILD/'scale_stop.json',dict(status='stopped after two OOMs',scale=scale,kind=kind,oom_count=oom));break
    for seed in range(30,35):
        t=time.time();job('retrain_checkpoints.py',['--seed',str(seed)],f'retrain_{seed}',1900)
        if time.time()-t>1800:
            dump(BUILD/'training_stop.json',dict(seed=seed,reason='single seed >30 minutes; no remaining seeds launched'))
            raise RuntimeError('STOP single training seed exceeded 30 minutes')
    reuse_probe()
    rows=[]
    for seed in range(30,35):
        for case in config['feasibility_ids']:
            out=BUILD/'feasibility'/f'seed{seed}_epoch0'/case
            if not (out/'metrics.json').exists():job('invert.py',['--case',case,'--seed',str(seed),'--epoch','0','--destination','feasibility'],f'feasibility_{seed}_{case}',900)
            rows.append(json.loads((out/'metrics.json').read_text()))
    per_seed=[dict(seed=seed,mean_reduction=float(np.mean([r['relative_cost_reduction'] for r in rows if r['seed']==seed]))) for seed in range(30,35)]
    passes=sum(r['mean_reduction']>=.5 for r in per_seed)
    dump(BUILD/'feasibility.json',dict(passed=passes>=4,seed_passes=passes,total_seeds=5,per_seed=per_seed,cases=rows,H2_status='not yet evaluated'))
    if passes<4:
        raise RuntimeError('STOP: native epoch0 feasibility failed; near-background fallback not authorized')
    # Calibration geometry only, predeclared seed30. Reuse completed epoch0 cases.
    for epoch in [0,100]:
        for case_idx in range(25):
            case=f'validation_{case_idx:03d}';dest=BUILD/'calibration'/f'seed30_epoch{epoch}'/case
            if epoch==0 and case_idx<4:
                src=BUILD/'feasibility'/f'seed30_epoch0'/case
                dest.mkdir(parents=True,exist_ok=True)
                for p in src.iterdir():shutil.copy2(p,dest/p.name)
            if not (dest/'metrics.json').exists():job('invert.py',['--case',case,'--seed','30','--epoch',str(epoch),'--destination','calibration'],f'calibration_{epoch}_{case}',900)
    job('evaluate.py',[],'freeze_thresholds',600)
    dump(BUILD/'round_status.json',dict(status='completed first-round scope',seconds=time.time()-config['start_unix'],scale_stopped=scale_stopped,test_inversions=0,T3_started=False))
except Exception as e:
    dump(BUILD/'round_status.json',dict(status='stopped',reason=str(e),seconds=time.time()-config['start_unix'],test_inversions=0,T3_started=False))
    print('ROUND_STOPPED',str(e),flush=True)
    raise
