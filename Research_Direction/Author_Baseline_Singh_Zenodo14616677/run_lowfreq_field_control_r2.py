"""Fixed100 paired replay composition, restoring identical coverage Adam300."""
import argparse,json,time,shutil
from pathlib import Path
import torch
from run_paired_depth_coverage import policy,data_setup,evaluation,ROOT,BASE,BasicFNO,state_hash,dump,sha,encode_time_features,supervised_losses,metrics
from run_depth_coverage_storage_retry import OUT as PREV
from run_paired_depth_recovery import safe_save,verify_eval,same,cpu_tree
PLAN=Path('.build/fno-lowfreq-field-control-r2-20261011/preregistered_protocol.json')
OUT=Path('.build/fno-lowfreq-field-control-r2-20261011')
INTERVALS=[('contrast03','depth0'),('depth0','contrast07'),('contrast03','contrast07')]

def response(ev,samples):
    return [{'interval':[lo,hi],**metrics(ev['traces'][hi]-ev['traces'][lo],samples[hi]['physical']['trace'].double()-samples[lo]['physical']['trace'].double())} for lo,hi in INTERVALS]

def report(raw):
    strip=lambda e:{k:v for k,v in e.items() if k not in ['traces','field_sha256']}
    finals={k:a['history'][-1]['evaluation'] for k,a in raw['arms'].items()}
    r,c=finals['loss_lowfreq'],finals['loss_original'];base=raw['arms']['loss_lowfreq']['history'][0]['evaluation']
    rm=raw['arms']['loss_lowfreq']['material_response'][-1]['rows'];cm=raw['arms']['loss_original']['material_response'][-1]['rows'];bm=raw['arms']['loss_lowfreq']['material_response'][0]['rows']
    primary=rm[-1]['relative_l2'];gain=primary<=.8*cm[-1]['relative_l2'] and primary<=.8*bm[-1]['relative_l2']
    anchor=all(r['records'][k]['scatter_'+sp+'_relative_l2']<=1.1*base['records'][k]['scatter_'+sp+'_relative_l2'] for k in ['contrast03','contrast07'] for sp in ['field','trace'])
    geometry=all(r['records'][k]['scatter_'+sp+'_relative_l2']<=1.1*base['records'][k]['scatter_'+sp+'_relative_l2'] for k in ['depth-1','depth1'] for sp in ['field','trace']) and all(r['records'][k]['scatter_trace_relative_l2']<=.2 for k in ['depth-1','depth1']) and r['response']['relative_l2']<=.2 and r['response']['cosine']>=.9
    contradiction=any(rm[i]['relative_l2']>=cm[i]['relative_l2'] or rm[i]['relative_l2']>=bm[i]['relative_l2'] for i in [0,1])
    decision='success' if gain and anchor and geometry and not contradiction else ('material_gain_geometry_harm' if gain and not geometry else ('failure_no_primary_improvement' if primary>=cm[-1]['relative_l2'] else 'inconclusive'))
    if raw['elapsed_seconds_including_verification']>240:decision='incomplete_budget_exceeded'
    return {'status':'completed' if raw['elapsed_seconds_including_verification']<=240 else 'incomplete','decision':decision,'material_effect_passed':gain,'anchor_guardrail_passed':anchor,'geometry_guardrail_passed':geometry,'subinterval_contradiction':contradiction,'proxy_qualified':r['screen_passed'],
      'primary_response_error':{'replay':primary,'control':cm[-1]['relative_l2'],'frozen_coverage':bm[-1]['relative_l2'],'replay_over_control':primary/cm[-1]['relative_l2'],'replay_over_frozen':primary/bm[-1]['relative_l2']},
      'arms':{name:{**{k:v for k,v in arm.items() if k!='history'},'history':[{'update':h['update'],'evaluation':strip(h['evaluation'])} for h in arm['history']]} for name,arm in raw['arms'].items()},
      'cost':{k:v for k,v in raw.items() if k!='arms'},'scope':'One initialization;all7 states used for development;nominal material finite responses,not local Jacobian,unseen joint generalization,material recovery,blind detection or total inversion cost.'}

def run():
    policy();plan=json.loads(PLAN.read_text(encoding='utf-8'))
    assert Path(plan['output_directory']).resolve()==OUT.resolve() and OUT.exists() and (OUT/'preregistered_protocol.json').exists()
    for p,h in plan['frozen_sha256'].items():assert sha(Path(p))==h
    # Include the complete frozen-data lineage and exact local functions before training.
    prior=json.loads((PREV/'protocol.json').read_text(encoding='utf-8'))
    frozen={**plan['frozen_sha256'],**prior['frozen_sha256']}
    for p,h in frozen.items():assert sha(Path(p))==h
    deps=['run_paired_depth_coverage.py','run_paired_depth_recovery.py','run_frozen_depth_shift.py','run_fno_local_contrast.py','discrete_reference.py','time_reconstruction.py','time_feature_input.py','basic_fno.py','run_receiver_loss_control.py','operator_data.py','train_multiworkpiece_control.py']
    assert shutil.disk_usage(OUT).free>3*1024**3
    dump(OUT/'protocol.json',{'status':'frozen_before_training','plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),'frozen_sha256':frozen,'dependencies':{str(ROOT/n):sha(ROOT/n) for n in deps},'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'storage':'D drive;CPU fsync temporary writes,reload tensor equality before commit','timing':'Clock starts before setup;includes both training,evaluations,safe writes and final model reload/replay;protocol SHA preflight separately.'})
    start=time.perf_counter();arm=None;model=None;opt=None;name=None
    def cap():
        if time.perf_counter()-start>240:raise TimeoutError('Frozen240sec inclusive cap')
    try:
        device,nom,bg,scale,sr,samples=data_setup();ids=nom['receiver_physical_indices'].to(device);inputs={};targets={}
        for key in ['depth-2','depth0','depth2','contrast03','contrast07']:
            s={'gamma':samples[key]['gamma'],'source_xy_m':nom['source_xy_m'][0],'x_m':nom['x_m'],'y_m':nom['y_m'],'time_seconds':nom['time_seconds_snapshots']}
            inputs[key]=encode_time_features(s,mode='zero')[None]
            # Preserve prior float32 division/subtraction order exactly,compute targets on device.
            targets[key]=(samples[key]['physical']['field'].to(device)/scale-bg['wavefield_snapshots'][0].to(device)/scale)[None,None].cpu()
        prior_raw=torch.load(Path(plan['initial_checkpoint']).parent/'material_replay_results.pt',weights_only=True,map_location='cpu');old=prior_raw['history'][-1]['evaluation'];del prior_raw
        initialck=torch.load(Path(plan['initial_checkpoint']),weights_only=True,map_location='cpu')
        assert initialck['update']==400 and all(float(s['step'])==400 for s in initialck['optimizer_state_dict']['state'].values())
        arms={}
        for name,order in plan['arms'].items():
            receiver_weight=1.0
            cap();astart=time.perf_counter();torch.cuda.reset_peak_memory_stats()
            model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(initialck['state_dict'])
            opt=torch.optim.Adam(model.parameters(),lr=.001)
            opt.load_state_dict(cpu_tree(initialck['optimizer_state_dict']))
            assert same(cpu_tree(opt.state_dict()),initialck['optimizer_state_dict'])
            assert all(g['lr']==.001 for g in opt.param_groups)
            arm={'initial_state_hash':state_hash(model),'initial_adam_equal':True,'slot_exposures':[0]*5,'losses':[],'history':[],'material_response':[]}
            assert arm['initial_state_hash']==initialck['state_hash']
            for update in range(101):
                cap()
                if update in [0,100]:
                    model.eval();ev=evaluation(model,samples,nom,bg,scale,device,True)
                    if update==0:verify_eval(ev,old)
                    arm['history'].append({'update':update,'evaluation':ev});arm['material_response'].append({'update':update,'rows':response(ev,samples)})
                    print(name,update,'material',arm['material_response'][-1]['rows'][-1]['relative_l2'],'geometry',ev['response']['relative_l2'],flush=True)
                if update==100:break
                slot=update%5;key=order[slot];model.train();opt.zero_grad(set_to_none=True)
                x=inputs[key].to(device);target=targets[key].to(device);pred=model(x);lf,lr=supervised_losses(pred,target,ids,scale,sr);low=((torch.fft.irfft(torch.fft.rfft(pred-target,dim=-1)*((torch.arange(torch.fft.rfft(pred-target,dim=-1).shape[-1],device=pred.device)<max(1,int(.1*torch.fft.rfft(pred-target,dim=-1).shape[-1]))).view(1,1,1,-1)),n=pred.shape[-1],dim=-1)).square().mean()) if name=='loss_lowfreq' else torch.zeros((),device=pred.device)
                loss=lf+receiver_weight*lr+0.5*low
                assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                opt.step();arm['slot_exposures'][slot]+=1;arm['losses'].append({'update':update+1,'slot':slot,'key':key,'field_loss':float(lf.detach()),'receiver_loss':float(lr.detach())})
                del x,target,pred,lf,lr,loss
                if (update+1)%25==0:print(name,'updates',update+1,'elapsed',round(time.perf_counter()-start,2),flush=True)
            assert arm['slot_exposures']==[20]*5
            safe_save({'state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'update':400,'new_updates':100,'slot_exposures':arm['slot_exposures'],'state_hash':state_hash(model)},OUT/f'{name}_update400.pt')
            torch.cuda.synchronize();arm.update(new_updates=100,final_state_hash=state_hash(model),elapsed_seconds=time.perf_counter()-astart,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            safe_save(arm,OUT/f'{name}_results.pt');arms[name]=arm;del model,opt;model=opt=None;torch.cuda.empty_cache()
        training_eval_seconds=time.perf_counter()-start;vstart=time.perf_counter()
        # The preregistered independent model reload/replay is included in the wall cap.
        for name,arm in arms.items():
            cap();ck=torch.load(OUT/f'{name}_update400.pt',weights_only=True,map_location='cpu');model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);model.eval()
            assert state_hash(model)==ck['state_hash']==arm['final_state_hash'];expected_step=500;assert all(float(s['step'])==expected_step for s in ck['optimizer_state_dict']['state'].values())
            ev=evaluation(model,samples,nom,bg,scale,device,True);verify_eval(ev,arm['history'][-1]['evaluation']);assert response(ev,samples)==arm['material_response'][-1]['rows']
            del model,ck;model=None;torch.cuda.empty_cache()
        cap()
        raw={'arms':arms,'training_evaluation_storage_seconds':training_eval_seconds,'reload_verification_seconds':time.perf_counter()-vstart,'elapsed_seconds_including_verification':time.perf_counter()-start,'new_updates':200,'new_physical_calls':0,'receiver_weights':{'loss_original':1.0,'loss_lowfreq':0.0},'model_forward_calls':242,'model_backward_calls':200,'calls_detail':'200training+14initial+14final+14reload;data_setup constructs model but no forward'}
        safe_save(raw,OUT/'raw.pt');dump(OUT/'summary.json',report(raw))
        verify_artifacts()
        cap()
        dump(OUT/'verification.json',{'status':'passed','final_reload':'All14 field hashes/trace bits and metrics match','initial':'Both weights/Adam300 exact;all14 initial field/trace predictions match previous coverage300','schedule':'Each arm100 updates;each slot20;Adamfinal400','saved_summaries':'Exact recomputation','physical':'0 new calls','raw_sha256':sha(OUT/'raw.pt'),'limits':'One init,used development data,no physical Jacobian/recovery test','execution_seconds_through_final_sha_audit':time.perf_counter()-start,'inclusive_clock_note':'Summary clock includes replay;verification clock additionally includes raw/report serialization and final SHA audit.'})
        print('Completed and verified',report(raw)['decision'],flush=True)
    except Exception as e:
        if model is not None and opt is not None and arm is not None:
            safe_save({'state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'completed_new_updates':len(arm['losses']),'arm':arm},OUT/f'{name}_interrupted.pt')
        dump(OUT/'failure.json',{'class':type(e).__name__,'error':str(e),'elapsed_seconds':time.perf_counter()-start,'arm':name,'completed_updates_current_arm':len(arm['losses']) if arm else 0,'cost_scope':'Keep all previous arm records too;no automatic rerun'})
        raise

def verify_artifacts():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert sha(PLAN)==p['plan_sha256'] and sha(Path(__file__))==p['script_sha256']
    for f,h in {**p['frozen_sha256'],**p['dependencies']}.items():assert sha(Path(f))==h
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');assert report(raw)==json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    for name,a in raw['arms'].items():
        assert a['new_updates']==100 and a['slot_exposures']==[20]*5 and len(a['losses'])==100
        for i,l in enumerate(a['losses']):assert l['update']==i+1 and l['slot']==i%5 and l['key']==p['plan']['arms'][name][i%5]
    assert raw['arms']['loss_original']['initial_state_hash']==raw['arms']['loss_lowfreq']['initial_state_hash']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:verify_artifacts();print('Saved artifacts verified;no extra model calls')
    else:run()
