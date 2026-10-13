import sys,time,json,subprocess,threading,os,gc
from pathlib import Path
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
BUILD=ROOT/'.build/singh-training-seeds-20261013'
original_start=(ROOT/'.build/training_seeds_20261013.log').stat().st_ctime
if len(sys.argv)>1:
    original=(BASE/'run_training_seeds_20261013.py').read_text(encoding='utf-8')
    head=original[:original.index("record=ROOT/")]
    defs=original[original.index("old=(BASE/"):original.index('all_times={}')]
    exec(compile(head+defs,'unchanged_training_setup','exec'),globals())
    seed=int(sys.argv[1]);INITSEED=30+seed;OUT=BUILD/f'seed{seed}';OUT.mkdir(exist_ok=True)
    START=time.time();DEADLINE=original_start+3550
    timings=json.loads((OUT/'timings.json').read_text()) if (OUT/'timings.json').exists() else {}
    for group in ['control','experiment']:
        if group not in timings or not (OUT/group/'model_Unet_200').exists():
            timings[group]={'training':train(group)}
            (OUT/'timings.json').write_text(json.dumps(timings,indent=2))
    transfer=transfer.replace('del ns;torch.cuda.empty_cache()','del ns;gc.collect();torch.cuda.empty_cache()')
    exec(compile(transfer,'unchanged_transfer_with_resource_cleanup','exec'),globals())
    timer.cancel()
else:
    started=time.time();failure_count=1
    for seed in range(2,5):
        if time.time()>original_start+3400:raise SystemExit('STOP budget')
        print('RESUME_SEED',seed,flush=True)
        run=subprocess.run([sys.executable,__file__,str(seed)])
        if run.returncode:
            failure_count+=1
            (BUILD/'stop_status.json').write_text(json.dumps(dict(failures=failure_count,seed=seed,wall_seconds=time.time()-original_start)))
            raise SystemExit('STOP second failure')
        path=BUILD/'run_times.json';r=json.loads(path.read_text())
        r['seeds'][str(seed)]=json.loads((BUILD/f'seed{seed}'/'timings.json').read_text())
        r['wall_seconds']=time.time()-original_start;r['failures']=failure_count
        path.write_text(json.dumps(r,indent=2))
    print('ALL_SEEDS_FINISHED',time.time()-original_start,flush=True)
