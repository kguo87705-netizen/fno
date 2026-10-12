"""Complete frozen paired300 experiment with verified CPU checkpoint writes."""
import argparse,json,time,os,shutil
from pathlib import Path
import torch
from run_paired_depth_coverage import OUT as FAILED,policy,data_setup,evaluation,report,ROOT,BASE,WIDE,BasicFNO,state_hash,dump,sha,encode_time_features,supervised_losses
OUT=Path('.build/fno-paired-depth-coverage-recovery-20261009')
PLAN=FAILED/'next_recovery_protocol.json'
RESTART=FAILED/'save_diagnostic/checkpoint150.pt'

def cpu_tree(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu_tree(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu_tree(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu_tree(x) for x in v)
    return v

def same(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def safe_save(obj,path):
    assert not path.exists() and not path.with_suffix(path.suffix+'.tmp').exists()
    tmp=path.with_suffix(path.suffix+'.tmp');data=cpu_tree(obj)
    with tmp.open('wb') as f:torch.save(data,f);f.flush();os.fsync(f.fileno())
    again=torch.load(tmp,weights_only=True,map_location='cpu');assert same(data,again)
    os.replace(tmp,path)
    return sha(path)

def verify_eval(a,b):
    assert a['field_sha256']==b['field_sha256'] and a['records']==b['records'] and a['response']==b['response']
    assert all(torch.equal(a['traces'][k],b['traces'][k]) for k in a['traces'])

def run():
    policy();plan=json.loads(PLAN.read_text(encoding='utf-8'))
    for f,h in plan['frozen_sha256'].items():assert sha(Path(f))==h
    old=json.loads((FAILED/'protocol.json').read_text());deps={**old['dependency_sha256'],str(ROOT/'run_paired_depth_coverage.py'):sha(ROOT/'run_paired_depth_coverage.py')}
    for f,h in deps.items():assert sha(Path(f))==h
    assert not OUT.exists();OUT.mkdir()
    start=time.perf_counter();free=shutil.disk_usage(OUT).free;assert free>3*1024**3,'Insufficient preflight free space'
    dump(OUT/'protocol.json',{'status':'frozen_before_recovery','plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),'dependencies':deps,'disk_free_bytes':free,'storage':'CPU copies; write flush fsync, weights_only reload/all tensor equality before unique-path replace','execution':'Same IEEE policy and loss/input arithmetic as frozen paired experiment; fresh Adam only coverage arm','gpu':torch.cuda.get_device_name(),'torch':torch.__version__})
    restart=torch.load(RESTART,weights_only=True,map_location='cpu');safe_save(restart,OUT/'preflight_checkpoint150.pt')
    device,nom,bg,scale,sr,samples=data_setup();inputs={};targets={};ids=nom['receiver_physical_indices'].to(device)
    for k in [-2,0,2]:
        key=f'depth{k}';s={**nom,'gamma':samples[key]['gamma']}
        encoded={'gamma':s['gamma'],'source_xy_m':s['source_xy_m'][0],'x_m':s['x_m'],'y_m':s['y_m'],'time_seconds':s['time_seconds_snapshots']}
        inputs[key]=encode_time_features(encoded,mode='zero')[None].to(device)
        targets[key]=(samples[key]['physical']['field'].to(device)/scale-bg['wavefield_snapshots'][0].to(device)/scale)[None,None]
    original_arm=torch.load(FAILED/'nominal_repeat_progress.pt',weights_only=True,map_location='cpu');arms={};new_updates=0
    for name,order in plan['scientific_plan']['arms'].items():
        armstart=time.perf_counter();torch.cuda.reset_peak_memory_stats();model=BasicFNO(width=8,input_channels=8).to(device)
        if name=='nominal_repeat':
            model.load_state_dict(restart['state_dict']);optimizer=torch.optim.Adam(model.parameters(),lr=.001);optimizer.load_state_dict(restart['optimizer_state_dict'])
            assert same(cpu_tree(optimizer.state_dict()),restart['optimizer_state_dict'])
            assert state_hash(model)==restart['state_hash'];arm=cpu_tree(original_arm);first=150
            model.eval();verify_eval(evaluation(model,samples,nom,bg,scale,device,False),arm['history'][-1]['evaluation'])
            print('nominal checkpoint150 and optimizer parity passed',flush=True)
        else:
            ck=torch.load(WIDE/'final.pt',weights_only=True,map_location='cpu');model.load_state_dict(ck['state_dict']);del ck
            optimizer=torch.optim.Adam(model.parameters(),lr=.001);assert len(optimizer.state)==0
            assert state_hash(model)==original_arm['initial_state_hash'];first=0
            arm={'initial_state_hash':state_hash(model),'exposures':[0,0,0],'losses':[],'history':[]}
        update=first
        try:
            for update in range(first,301):
                if time.perf_counter()-start>600:raise TimeoutError('10-minute recovery cap')
                if update in [0,150,300] and not (name=='nominal_repeat' and update==150):
                    model.eval();ev=evaluation(model,samples,nom,bg,scale,device,update in [0,300]);arm['history'].append({'update':update,'evaluation':ev})
                    if update==0:verify_eval(ev,original_arm['history'][0]['evaluation'])
                    safe_save({'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'update':update,'exposures':arm['exposures'].copy(),'state_hash':state_hash(model)},OUT/f'{name}_update{update}.pt')
                    print(name,update,'dev',[(k,ev['records'][k]['scatter_trace_relative_l2']) for k in ['depth-1','depth1']],'response',ev['response']['relative_l2'],flush=True)
                if update==300:break
                slot=update%3;key=f'depth{order[slot]}';model.train();optimizer.zero_grad(set_to_none=True)
                pred=model(inputs[key]);lf,lr=supervised_losses(pred,targets[key],ids,scale,sr);loss=lf+lr
                assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                arm['losses'].append({'update':update+1,'slot':slot,'depth':order[slot],'field_loss':float(lf.detach()),'receiver_loss':float(lr.detach())});optimizer.step();arm['exposures'][slot]+=1;new_updates+=1;del pred,lf,lr,loss
                if (update+1)%25==0:
                    safe_save(arm,OUT/f'{name}_progress{update+1}.pt')
                    print(name,'updates',update+1,'recovery elapsed',round(time.perf_counter()-start,1),flush=True)
        except Exception:
            safe_save({'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'update':len(arm['losses']),'exposures':arm['exposures'],'state_hash':state_hash(model)},OUT/f'{name}_interrupted{len(arm["losses"])}.pt')
            safe_save(arm,OUT/f'{name}_interrupted_progress.pt')
            raise
        assert arm['exposures']==[100]*3 and len(arm['losses'])==300
        torch.cuda.synchronize();arm.update(elapsed_seconds=time.perf_counter()-armstart,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),final_state_hash=state_hash(model),updates=300,recovery_start_update=first)
        arms[name]=arm;safe_save(arm,OUT/f'{name}_results.pt');del model,optimizer;torch.cuda.empty_cache()
    assert new_updates==450
    raw={'arms':arms,'elapsed_seconds':time.perf_counter()-start,'new_updates':new_updates};safe_save(raw,OUT/'raw.pt')
    summary=report(arms,raw['elapsed_seconds']);summary.update(recovery_new_updates=450,across_attempts_executed_updates=750,failed_run_seconds_lower_bound=302.1,cost_warning='600 updates in the selected trajectories;750 actual across attempts. Per-arm elapsed here is recovery only; original costs/verification/historical training separately, no total inversion benchmark.')
    dump(OUT/'summary.json',summary)

def verify():
    policy();p=json.loads((OUT/'protocol.json').read_text());assert sha(PLAN)==p['plan_sha256'] and sha(Path(__file__))==p['script_sha256']
    for f,h in {**p['plan']['frozen_sha256'],**p['dependencies']}.items():assert sha(Path(f))==h
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');summary=json.loads((OUT/'summary.json').read_text())
    expected=report(raw['arms'],raw['elapsed_seconds']);assert all(summary[k]==v for k,v in expected.items()) and raw['new_updates']==450
    device,nom,bg,scale,sr,samples=data_setup()
    for name,arm in raw['arms'].items():
        assert arm['updates']==300 and arm['exposures']==[100]*3 and len(arm['losses'])==300
        for i,l in enumerate(arm['losses']):assert l['update']==i+1 and l['slot']==i%3 and l['depth']==p['plan']['scientific_plan']['arms'][name][i%3]
        ck=torch.load(OUT/f'{name}_update300.pt',weights_only=True,map_location='cpu');model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);model.eval()
        assert state_hash(model)==ck['state_hash']==arm['final_state_hash']
        assert all(float(s['step'])==300 for s in ck['optimizer_state_dict']['state'].values())
        verify_eval(evaluation(model,samples,nom,bg,scale,device,True),arm['history'][-1]['evaluation']);del model,ck;torch.cuda.empty_cache()
    dump(OUT/'verification.json',{'status':'passed','primary':'Both final300 checkpoints;Adam300;100 exposures per slot;all600 trajectory losses','recovery':'150 state and optimizer parity;original initial equality for coverage','final_predictions':'All14 field hashes and traces bit-identical checkpoint reload;metrics exact','storage':'All writes CPU-converted and reloaded before commit','new_updates':450,'across_attempts_executed_updates':750,'inputs':'SHA unchanged','raw_sha256':sha(OUT/'raw.pt'),'limits':'One initialization and used development states;no material Jacobian/recovery/final test'})
    print('Verification passed')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    try:verify() if args.verify else run()
    except Exception as e:
        if OUT.exists():dump(OUT/'failure.json',{'class':type(e).__name__,'error':str(e)})
        raise
