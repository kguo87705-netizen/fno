"""Width8-only diagnostic against frozen width4; fixed1200, no data generation."""
from pathlib import Path
import argparse
import json
import time
import torch
from basic_fno import BasicFNO
from fixed_source_inputs import FixedSourceInputs
from time_feature_input import encode_time_features
from train_multiworkpiece_control import state_hash
from run_receiver_loss_control import supervised_losses
from run_four_materials import evaluate
from run_fno_local_contrast import samples, refs
from run_reduced_material_scan import sha, check_sources

ROOT=Path(__file__).resolve().parent
BASE=Path('.build/fno-local-contrast-20261009')
OUT=Path('.build/fno-width8-control-20261009')
PLAN=BASE/'next_width_control_protocol.json'
HOMOGENEOUS=Path('.build/operator-pilot-v3-20261008/homogeneous.pt')


def counts(model):
    return {'stored_elements':sum(p.numel() for p in model.parameters()),
            'real_parameters':sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())}


def check_plan(plan):
    check_sources()
    for name,digest in plan['frozen_files_sha256'].items():assert sha(Path(name))==digest,name


def run():
    torch.set_num_threads(4)
    plan=json.loads(PLAN.read_text());assert plan['status']=='planned_not_run';check_plan(plan)
    assert not OUT.exists(),'Preserve prior results; select a new directory for a new protocol.'
    OUT.mkdir()
    dependency_names=['run_fno_local_contrast.py','basic_fno.py','fixed_source_inputs.py','time_feature_input.py',
                      'operator_data.py','run_four_materials.py','run_receiver_loss_control.py','time_reconstruction.py',
                      'train_multiworkpiece_control.py','run_reduced_material_scan.py']
    protocol={'status':'fixed_before_training','plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),
              'dependencies':{name:sha(ROOT/name) for name in dependency_names},'updates':1200,
              'evaluation_updates':[0,400,800,1200],'sample_order':[i%3 for i in range(1200)],
              'scope':plan['scope']}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    manifest,groups=samples();train,held=groups['train'],groups['development_interpolation']
    assert [s['contrast'] for s in train]==[.3,.5,.7] and [s['contrast'] for s in held]==[.4,.6]
    scale,sr=manifest['normalization']['wavefield_rms'],manifest['normalization']['trace_rms']
    bg=torch.load(HOMOGENEOUS,weights_only=True,map_location='cpu')
    references,coefficients,targets=refs(train,held,bg,scale,sr,device)
    frozen_refs=json.loads((BASE/'references.json').read_text())
    assert references==frozen_refs['metrics'] and coefficients==frozen_refs['linear_coefficients']
    inputs=FixedSourceInputs(train,device)
    for i in range(3):assert torch.equal(inputs[i],encode_time_features(train[i],mode='zero')[None].to(device))
    torch.manual_seed(2);model=BasicFNO(width=8,input_channels=8).to(device)
    old_initial=json.loads((BASE/'initial_model.json').read_text())
    assert [k for k in model.config if model.config[k]!=old_initial['config'][k]]==['width']
    old_model=BasicFNO(width=4,input_channels=8)
    initial={'seed':2,'state_hash':state_hash(model),'config':model.config,'counts':counts(model),
             'width4_counts':counts(old_model),'device':device,'cuda_name':torch.cuda.get_device_name() if device=='cuda' else None,
             'input_parity':'three training inputs bit identical to canonical encoding',
             'initialization_limit':'Different dimensions; same seed does not match initial predictions or isolate initialization.'}
    del old_model
    (OUT/'initial_model.json').write_text(json.dumps(initial,indent=2),encoding='utf-8')
    opt=torch.optim.Adam(model.parameters(),lr=.001);history=[];exposures=[0,0,0]
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    for update in range(1201):
        if update in protocol['evaluation_updates']:
            model.eval();training,_=evaluate(model,train,bg,scale,sr,device,inputs=inputs)
            interpolation,example=evaluate(model,held,bg,scale,sr,device,save_example=update==1200)
            history.append({'update':update,'training':training,'development_interpolation':interpolation})
            (OUT/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            print(json.dumps({'update':update,'train_scatter_field':training['scatter_field_relative_l2'],
                             'train_scatter_trace':training['scatter_trace_relative_l2'],
                             'interpolation_scatter_field':interpolation['scatter_field_relative_l2'],
                             'interpolation_scatter_trace':interpolation['scatter_trace_relative_l2']}),flush=True)
        if update==1200:break
        i=update%3;model.train();opt.zero_grad(set_to_none=True)
        pred=model(inputs[i]);lf,lr=supervised_losses(pred,targets[i][None,None],train[i]['receiver_indices'].to(device),scale,sr)
        loss=lf+lr;assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step();exposures[i]+=1;del pred,lf,lr,loss
        if (update+1)%100==0:print(f'Update {update+1}/1200, elapsed {time.perf_counter()-start:.1f}s',flush=True)
    assert exposures==[400]*3
    if device=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    memory={'peak_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None,
            'scope':'From reference/input/model preparation through fixed training and four evaluations'}
    torch.save({'state_dict':model.state_dict(),'optimizer_state_dict':opt.state_dict(),'config':model.config,
                'updates':1200,'exposures':exposures,'scale':scale},OUT/'final.pt')
    torch.save(example,OUT/'interpolation_example.pt')
    previous=json.loads((BASE/'comparison.json').read_text())
    report={'status':'completed','history':history,'final_training':training,'final_interpolation':interpolation,
            'references':references,'exposures':exposures,'elapsed_seconds':elapsed,'cuda_memory':memory,
            'width4_results':{k:previous[k] for k in ['final_training','final_interpolation','elapsed_seconds']},
            'scope':plan['scope']}
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


def verify():
    torch.set_num_threads(4)
    p=json.loads((OUT/'protocol.json').read_text());check_plan(p['plan'])
    assert sha(PLAN)==p['plan_sha256'] and sha(Path(__file__))==p['script_sha256']
    for name,digest in p['dependencies'].items():assert sha(ROOT/name)==digest
    manifest,groups=samples();train,held=groups['train'],groups['development_interpolation']
    scale,sr=manifest['normalization']['wavefield_rms'],manifest['normalization']['trace_rms']
    device='cuda' if torch.cuda.is_available() else 'cpu';bg=torch.load(HOMOGENEOUS,weights_only=True,map_location='cpu')
    references,coefficients,_=refs(train,held,bg,scale,sr,device)
    r=json.loads((OUT/'comparison.json').read_text());assert r['status']=='completed' and references==r['references']
    assert coefficients==json.loads((BASE/'references.json').read_text())['linear_coefficients']
    torch.manual_seed(2);model=BasicFNO(width=8,input_channels=8).to(device)
    initial=json.loads((OUT/'initial_model.json').read_text());assert state_hash(model)==initial['state_hash'] and counts(model)==initial['counts']
    saved=torch.load(OUT/'final.pt',weights_only=True,map_location='cpu')
    assert saved['updates']==1200 and saved['exposures']==r['exposures']==[400]*3
    assert all(torch.isfinite(v).all() for v in saved['state_dict'].values())
    for state in saved['optimizer_state_dict']['state'].values():
        assert int(state['step'])==1200 and all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor))
    model.load_state_dict(saved['state_dict']);model.eval();inputs=FixedSourceInputs(train,device)
    for i in range(3):assert torch.equal(inputs[i],encode_time_features(train[i],mode='zero')[None].to(device))
    training,_=evaluate(model,train,bg,scale,sr,device,inputs=inputs)
    interpolation,example=evaluate(model,held,bg,scale,sr,device,save_example=True)
    assert training==r['final_training'] and interpolation==r['final_interpolation']
    example_saved=torch.load(OUT/'interpolation_example.pt',weights_only=True,map_location='cpu')
    for k in example:
        if isinstance(example[k],torch.Tensor):assert torch.equal(example[k],example_saved[k])
        else:assert example[k]==example_saved[k]
    old=json.loads((BASE/'comparison.json').read_text())
    for k,v in r['width4_results'].items():assert v==old[k]
    checks={'status':'passed','nine_frozen_files':'SHA unchanged','code_and_initialization':'SHA and seed reconstruction passed',
            'input_encoding':'bit identical','references':'exact recomputed','final_metrics':'exact reload',
            'example_prediction':'bit identical','optimizer_steps':1200,'exposures':[400]*3,'scope':p['scope']}
    (OUT/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    try:
        verify() if args.verify else run()
    except torch.cuda.OutOfMemoryError as exc:
        if OUT.exists():
            (OUT/'failure.json').write_text(json.dumps({'status':'failed','reason':'CUDA out of memory','detail':str(exc),
                'policy':'No grid, modes, precision, losses or data changed; stop this preregistered arm.'},indent=2),encoding='utf-8')
        raise
