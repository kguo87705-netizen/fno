"""Retain verified nominal300; complete coverage on spacious D drive."""
import argparse,json,time,shutil
from pathlib import Path
import torch
from run_paired_depth_recovery import OUT as PREVIOUS,PLAN,safe_save,verify_eval,policy,data_setup,evaluation,report,ROOT,WIDE,BasicFNO,state_hash,dump,sha,encode_time_features,supervised_losses
from run_paired_depth_coverage import OUT as FAILED
OUT=Path('.build/fno-paired-depth-coverage-recovery-r2-20261009')

def summary(raw):
    r=report(raw['arms'],raw['elapsed_seconds']);r.update(retry_new_updates=300,recovery_new_updates=450,across_attempts_executed_updates=750,
        failed_first_run_seconds_lower_bound=302.1,prior_nominal_recovery_seconds=raw['arms']['nominal_repeat']['elapsed_seconds'],
        cost_warning='Selected trajectories600 updates;actual750 across attempts. Elapsed here is storage retry only. Nominal elapsed is prior resumed150,coverage elapsed is new300;not equal-budget timing comparison. Failed storage and verification costs not fully timed.')
    return r

def run():
    policy();plan=json.loads(PLAN.read_text(encoding='utf-8'));previous=json.loads((PREVIOUS/'protocol.json').read_text())
    frozen={**plan['frozen_sha256'],**previous['dependencies'],str(PREVIOUS/'nominal_repeat_update300.pt'):sha(PREVIOUS/'nominal_repeat_update300.pt'),str(PREVIOUS/'nominal_repeat_results.pt'):sha(PREVIOUS/'nominal_repeat_results.pt'),str(ROOT/'run_paired_depth_recovery.py'):sha(ROOT/'run_paired_depth_recovery.py')}
    for f,h in frozen.items():assert sha(Path(f))==h
    assert not OUT.exists();OUT.mkdir(parents=True)
    start=time.perf_counter();free=shutil.disk_usage(OUT).free;assert free>5*1024**3
    dump(OUT/'protocol.json',{'status':'fixed_before_coverage_retry','plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),'frozen_sha256':frozen,
        'storage_amendment':'Only output drive changed to D after transient ENOSPC during coverage0 write on C. Prior nominal300 safely saved; no nominal retraining. All old files preserved.',
        'budget':'Only300 outstanding coverage updates;cap360sec,within remaining recovery training budget;no new labels/downloads. Separate checkpoint verification allowed.','disk_free_bytes':free,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__})
    nominal=torch.load(PREVIOUS/'nominal_repeat_results.pt',weights_only=True,map_location='cpu')
    nck=torch.load(PREVIOUS/'nominal_repeat_update300.pt',weights_only=True,map_location='cpu');safe_save(nck,OUT/'nominal_repeat_update300.pt')
    device,nom,bg,scale,sr,samples=data_setup();model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(nck['state_dict']);model.eval()
    assert state_hash(model)==nominal['final_state_hash']==nck['state_hash'];verify_eval(evaluation(model,samples,nom,bg,scale,device,True),nominal['history'][-1]['evaluation']);del model,nck
    print('nominal final300 reload verified',flush=True)
    inputs={};targets={};ids=nom['receiver_physical_indices'].to(device)
    for k in [-2,0,2]:
        key=f'depth{k}';s={**nom,'gamma':samples[key]['gamma']}
        encoded={'gamma':s['gamma'],'source_xy_m':s['source_xy_m'][0],'x_m':s['x_m'],'y_m':s['y_m'],'time_seconds':s['time_seconds_snapshots']}
        inputs[key]=encode_time_features(encoded,mode='zero')[None].to(device)
        targets[key]=(samples[key]['physical']['field'].to(device)/scale-bg['wavefield_snapshots'][0].to(device)/scale)[None,None]
    armstart=time.perf_counter();torch.cuda.reset_peak_memory_stats();ck=torch.load(WIDE/'final.pt',weights_only=True,map_location='cpu')
    model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);del ck
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);assert len(optimizer.state)==0 and state_hash(model)==nominal['initial_state_hash']
    arm={'initial_state_hash':state_hash(model),'exposures':[0,0,0],'losses':[],'history':[]};order=[-2,0,2]
    try:
        for update in range(301):
            if time.perf_counter()-start>360:raise TimeoutError('6-minute outstanding coverage cap')
            if update in [0,150,300]:
                model.eval();ev=evaluation(model,samples,nom,bg,scale,device,update in [0,300]);arm['history'].append({'update':update,'evaluation':ev})
                if update==0:verify_eval(ev,nominal['history'][0]['evaluation'])
                safe_save({'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'update':update,'exposures':arm['exposures'].copy(),'state_hash':state_hash(model)},OUT/f'depth_coverage_update{update}.pt')
                print('coverage',update,'dev',[(k,ev['records'][k]['scatter_trace_relative_l2']) for k in ['depth-1','depth1']],'response',ev['response']['relative_l2'],flush=True)
            if update==300:break
            slot=update%3;key=f'depth{order[slot]}';model.train();optimizer.zero_grad(set_to_none=True)
            pred=model(inputs[key]);lf,lr=supervised_losses(pred,targets[key],ids,scale,sr);loss=lf+lr;assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            arm['losses'].append({'update':update+1,'slot':slot,'depth':order[slot],'field_loss':float(lf.detach()),'receiver_loss':float(lr.detach())});optimizer.step();arm['exposures'][slot]+=1;del pred,lf,lr,loss
            if (update+1)%25==0:safe_save(arm,OUT/f'coverage_progress{update+1}.pt');print('coverage updates',update+1,'elapsed',round(time.perf_counter()-start,1),flush=True)
    except Exception:
        safe_save({'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'update':len(arm['losses']),'exposures':arm['exposures'],'state_hash':state_hash(model)},OUT/f'coverage_interrupted{len(arm["losses"])}.pt')
        safe_save(arm,OUT/'coverage_interrupted_progress.pt');raise
    assert arm['exposures']==[100]*3
    torch.cuda.synchronize();arm.update(updates=300,elapsed_seconds=time.perf_counter()-armstart,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),final_state_hash=state_hash(model))
    raw={'arms':{'nominal_repeat':nominal,'depth_coverage':arm},'elapsed_seconds':time.perf_counter()-start};safe_save(raw,OUT/'raw.pt');dump(OUT/'summary.json',summary(raw))

def verify():
    policy();p=json.loads((OUT/'protocol.json').read_text());assert sha(PLAN)==p['plan_sha256'] and sha(Path(__file__))==p['script_sha256']
    for f,h in p['frozen_sha256'].items():assert sha(Path(f))==h
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');assert summary(raw)==json.loads((OUT/'summary.json').read_text())
    device,nom,bg,scale,sr,samples=data_setup()
    for name,arm in raw['arms'].items():
        assert arm['updates']==300 and arm['exposures']==[100]*3 and len(arm['losses'])==300
        order=[0,0,0] if name=='nominal_repeat' else [-2,0,2]
        for i,l in enumerate(arm['losses']):assert l['update']==i+1 and l['slot']==i%3 and l['depth']==order[i%3]
        ck=torch.load(OUT/f'{name}_update300.pt',weights_only=True,map_location='cpu');model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);model.eval()
        assert state_hash(model)==ck['state_hash']==arm['final_state_hash'];assert all(float(s['step'])==300 for s in ck['optimizer_state_dict']['state'].values())
        verify_eval(evaluation(model,samples,nom,bg,scale,device,True),arm['history'][-1]['evaluation']);del model,ck;torch.cuda.empty_cache()
    verify_eval(raw['arms']['nominal_repeat']['history'][0]['evaluation'],raw['arms']['depth_coverage']['history'][0]['evaluation'])
    dump(OUT/'verification.json',{'status':'passed','primary':'Both fixed300;Adam300,slots100,600 trajectory losses','initial':'Original parameters and all initial prediction fields/traces equal','final':'All14 fields SHA and traces bit-identical reload;all metrics exact','storage':'D drive CPU temporary writes checked before commit;prior artifacts retained','actual_updates_across_attempts':750,'inputs':'SHA unchanged','raw_sha256':sha(OUT/'raw.pt'),'scope':'Used development states;no material derivative/recovery/final test'})
    print('Verification passed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
    try:verify() if args.verify else run()
    except Exception as e:
        if OUT.exists():dump(OUT/'failure.json',{'class':type(e).__name__,'error':str(e)})
        raise
