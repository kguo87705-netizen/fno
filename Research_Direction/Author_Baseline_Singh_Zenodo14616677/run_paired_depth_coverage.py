"""Paired identical-checkpoint, fixed300 geometry-distribution intervention."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from run_frozen_depth_shift import policy,setup,predict,BASE,WIDE,ROOT,OUT as DEPTH
from run_depth_interpolation import OUT as INTERP
from run_contrast_derivative_audit import dump,metrics
from run_reduced_material_scan import sha
from time_feature_input import encode_time_features
from basic_fno import BasicFNO
from train_multiworkpiece_control import state_hash
from run_receiver_loss_control import supervised_losses
OUT=Path('.build/fno-paired-depth-coverage-control-20261009')
PLAN=INTERP/'next_paired_depth_coverage_protocol.json'

def tensor_hash(x):return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()

def data_setup():
    _,_,_,_,_,_,device,nom,bg,scale,model=setup();del model
    depth=torch.load(DEPTH/'raw.pt',weights_only=True,map_location='cpu')
    val=torch.load(INTERP/'raw.pt',weights_only=True,map_location='cpu')
    samples={}
    for k in [-2,0,2]:samples[f'depth{k}']={'gamma':depth['states'][str(k)]['gamma'],'physical':depth['states'][str(k)]['physical'],'role':'training_geometry'}
    for k in [-1,1]:samples[f'depth{k}']={'gamma':val['states'][str(k)]['gamma'],'physical':val['states'][str(k)]['physical'],'role':'used_development'}
    for a in [3,7]:
        d=torch.load(BASE/f'contrast_{a:02d}.pt',weights_only=True,map_location='cpu')
        samples[f'contrast0{a}']={'gamma':d['gamma'],'physical':{'field':d['wavefield_snapshots'][0],'trace':d['traces_full']},'role':'forgetting_diagnostic'}
    sr=json.loads((BASE/'data_manifest.json').read_text())['normalization']['trace_rms']
    return device,nom,bg,scale,sr,samples

def evaluation(model,samples,nom,bg,scale,device,forgetting):
    records={};traces={};hashes={}
    for key,s in samples.items():
        if s['role']=='forgetting_diagnostic' and not forgetting:continue
        p=predict(model,{**nom,'gamma':s['gamma']},device,scale,bg);ref=s['physical'];rec={'role':s['role']}
        for space in ['field','trace']:
            b=bg['wavefield_snapshots'][0] if space=='field' else bg['traces_full'][0:1]
            e=p[space].double()-ref[space].double()
            rec['full_'+space+'_relative_l2']=float(e.norm()/ref[space].double().norm())
            rec['scatter_'+space+'_relative_l2']=float(e.norm()/(ref[space].double()-b.double()).norm())
        records[key]=rec;traces[key]=p['trace'];hashes[key]=tensor_hash(p['field'])
        del p
    f=samples['depth1']['physical']['trace'].double()-samples['depth-1']['physical']['trace'].double()
    response=metrics(traces['depth1']-traces['depth-1'],f)
    passed=response['relative_l2']<=.2 and response['cosine']>=.9 and all(records[k]['scatter_'+n+'_relative_l2']<=.2 for k in ['depth-1','depth1'] for n in ['field','trace'])
    return {'records':records,'response':response,'screen_passed':passed,'traces':traces,'field_sha256':hashes}

def report(arms,elapsed):
    final={a:v['history'][-1]['evaluation'] for a,v in arms.items()};c=final['depth_coverage'];n=final['nominal_repeat']
    dominates=all(c['records'][k]['scatter_'+s+'_relative_l2']<n['records'][k]['scatter_'+s+'_relative_l2'] for k in ['depth-1','depth1'] for s in ['field','trace']) and c['response']['relative_l2']<n['response']['relative_l2']
    decision=('coverage_repairs_and_repeat_does_not' if c['screen_passed'] and not n['screen_passed'] and dominates else ('both_repair_similarly' if c['screen_passed'] and n['screen_passed'] else ('coverage_improves_but_fails_screens' if dominates and not c['screen_passed'] else 'neither_repairs')))
    def strip(e):return {k:v for k,v in e.items() if k not in ['traces','field_sha256']}
    return {'status':'completed','decision':decision,'coverage_dominates_repeat_all_metrics':dominates,'causal_screen_passed':dominates and c['screen_passed'],
      'arms':{a:{**{k:v for k,v in arm.items() if k!='history'},'history':[{'update':h['update'],'evaluation':strip(h['evaluation'])} for h in arm['history']]} for a,arm in arms.items()},
      'elapsed_seconds':elapsed,'primary_update':300,'total_updates':600,'physical_calls':0,
      'linear_baselines':json.loads((INTERP/'summary.json').read_text()),
      'scope':'Paired one-initialization local-depth intervention on used development states; not final testing, material derivative/recovery or full-cost benefit'}

def run():
    policy();plan=json.loads(PLAN.read_text(encoding='utf-8'));frozen=plan['frozen_sha256']
    for p,h in frozen.items():assert sha(Path(p))==h
    assert not OUT.exists();OUT.mkdir()
    deps=['run_frozen_depth_shift.py','run_depth_interpolation.py','run_fno_local_contrast.py','time_reconstruction.py','train_multiworkpiece_control.py','run_reduced_material_scan.py','run_contrast_derivative_audit.py','operator_data.py']
    dump(OUT/'protocol.json',{'status':'frozen_before_training','plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),'frozen_sha256':frozen,'dependency_sha256':{str(ROOT/n):sha(ROOT/n) for n in deps},'training_target':'physical full snapshot / original field scale - homogeneous snapshot / same scale, exactly prior normalization arithmetic','initialization':'Original state_dict and fresh Adam both arms; no old moments','evaluation':'CPU float64 accumulation from frozen predictor representation; final300 fixed','gpu':torch.cuda.get_device_name(),'torch':torch.__version__})
    start=time.perf_counter();device,nom,bg,scale,sr,samples=data_setup()
    inputs={};targets={};ids=nom['receiver_physical_indices'].to(device)
    for k in [-2,0,2]:
        key=f'depth{k}';s={**nom,'gamma':samples[key]['gamma']}
        encoded={'gamma':s['gamma'],'source_xy_m':s['source_xy_m'][0],'x_m':s['x_m'],'y_m':s['y_m'],'time_seconds':s['time_seconds_snapshots']}
        inputs[key]=encode_time_features(encoded,mode='zero')[None].to(device)
        targets[key]=(samples[key]['physical']['field'].to(device)/scale-bg['wavefield_snapshots'][0].to(device)/scale)[None,None]
    arms={};initial=None
    for name,order in plan['arms'].items():
        armstart=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        ck=torch.load(WIDE/'final.pt',weights_only=True,map_location='cpu');model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);del ck
        optimizer=torch.optim.Adam(model.parameters(),lr=.001);assert len(optimizer.state)==0
        h0=state_hash(model);arm={'initial_state_hash':h0,'exposures':[0,0,0],'losses':[],'history':[]}
        for update in range(301):
            if time.perf_counter()-start>480:raise TimeoutError('8-minute paired cap')
            if update in [0,150,300]:
                model.eval();ev=evaluation(model,samples,nom,bg,scale,device,update in [0,300]);arm['history'].append({'update':update,'evaluation':ev})
                if update==0:
                    if initial is None:initial=ev
                    else:
                        assert h0==arms['nominal_repeat']['initial_state_hash']
                        assert ev['field_sha256']==initial['field_sha256'] and all(torch.equal(ev['traces'][k],initial['traces'][k]) for k in ev['traces'])
                if update in [150,300]:torch.save({'state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'update':update,'exposures':arm['exposures'].copy(),'state_hash':state_hash(model)},OUT/f'{name}_update{update}.pt')
                torch.save(arm,OUT/f'{name}_progress.pt')
                print(name,update,'dev',[(k,ev['records'][k]['scatter_trace_relative_l2']) for k in ['depth-1','depth1']],'response',ev['response']['relative_l2'],flush=True)
            if update==300:break
            slot=update%3;key=f'depth{order[slot]}';model.train();optimizer.zero_grad(set_to_none=True)
            pred=model(inputs[key]);lf,lr=supervised_losses(pred,targets[key],ids,scale,sr);loss=lf+lr
            assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            arm['losses'].append({'update':update+1,'slot':slot,'depth':order[slot],'field_loss':float(lf.detach()),'receiver_loss':float(lr.detach())});optimizer.step();arm['exposures'][slot]+=1;del pred,lf,lr,loss
            if (update+1)%75==0:print(name,'updates',update+1,'elapsed',round(time.perf_counter()-start,1),flush=True)
        assert arm['exposures']==[100]*3
        torch.cuda.synchronize();arm.update(elapsed_seconds=time.perf_counter()-armstart,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),final_state_hash=state_hash(model),updates=300)
        arms[name]=arm;torch.save(arm,OUT/f'{name}_results.pt');del model,optimizer;torch.cuda.empty_cache()
    raw={'arms':arms,'elapsed_seconds':time.perf_counter()-start};torch.save(raw,OUT/'raw.pt');dump(OUT/'summary.json',report(arms,raw['elapsed_seconds']))

def verify():
    policy();p=json.loads((OUT/'protocol.json').read_text());assert sha(PLAN)==p['plan_sha256'] and sha(Path(__file__))==p['script_sha256']
    for f,h in {**p['frozen_sha256'],**p['dependency_sha256']}.items():assert sha(Path(f))==h
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');assert report(raw['arms'],raw['elapsed_seconds'])==json.loads((OUT/'summary.json').read_text())
    device,nom,bg,scale,sr,samples=data_setup()
    for name,arm in raw['arms'].items():
        assert arm['updates']==300 and arm['exposures']==[100]*3 and len(arm['losses'])==300
        for i,l in enumerate(arm['losses']):assert l['update']==i+1 and l['slot']==i%3 and l['depth']==p['plan']['arms'][name][i%3]
        ck=torch.load(OUT/f'{name}_update300.pt',weights_only=True,map_location='cpu');model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);model.eval()
        assert state_hash(model)==arm['final_state_hash']==ck['state_hash']
        assert all(float(s['step'])==300 for s in ck['optimizer_state_dict']['state'].values())
        ev=evaluation(model,samples,nom,bg,scale,device,True);saved=arm['history'][-1]['evaluation']
        assert ev['field_sha256']==saved['field_sha256'] and ev['records']==saved['records'] and ev['response']==saved['response']
        assert all(torch.equal(ev['traces'][k],saved['traces'][k]) for k in ev['traces']);del model,ck;torch.cuda.empty_cache()
    assert raw['arms']['nominal_repeat']['initial_state_hash']==raw['arms']['depth_coverage']['initial_state_hash']
    dump(OUT/'verification.json',{'status':'passed','updates':'600 recorded; each arm Adam states300,slots100 each','initial':'same parameter hashes and all initial field hashes/trace bits','final':'All14 final fields SHA and traces bit-identical on checkpoint reload; all metrics exact','inputs':'SHA unchanged','raw_sha256':sha(OUT/'raw.pt'),'limits':'One initialization, used development states, same solver data; no physical material derivative test'})
    print('Verification passed')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--verify',action='store_true');args=a.parse_args()
    try:verify() if args.verify else run()
    except (TimeoutError,torch.cuda.OutOfMemoryError,AssertionError) as e:
        if OUT.exists():dump(OUT/'failure.json',{'error':str(e),'class':type(e).__name__})
        raise
