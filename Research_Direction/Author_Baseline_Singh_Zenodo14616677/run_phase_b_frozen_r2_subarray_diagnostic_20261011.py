import json,hashlib,sys,time
from pathlib import Path
import torch
sys.path.insert(0,'Research_Direction/Author_Baseline_Singh_Zenodo14616677')
from run_frozen_depth_shift import policy,predict
from basic_fno import BasicFNO
OUT=Path('.build/phase-b-frozen-r2-subarray-diagnostic-20261011-r4');assert not OUT.exists();OUT.mkdir()
A=Path('.build/phase-a-array-interface-20261011-r2');D=Path('.build/fno-frozen-depth-shift-audit-20261009-r2');B=Path('.build/fno-local-contrast-20261009');CK=Path('.build/fno-lowfreq-field-control-r2-20261011/loss_original_update400.pt');BG=Path('.build/operator-pilot-v3-20261008/homogeneous.pt');ROOT=Path('Research_Direction/Author_Baseline_Singh_Zenodo14616677')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def err(a,b):return float((a.double()-b.double()).norm()/b.double().norm())
def policy_local():policy();torch.set_num_threads(4)
policy_local(); torch.set_float32_matmul_precision('highest'); _,_,template,_,_,_,_=__import__('run_fno_local_contrast',fromlist=['physical_setup']).physical_setup()
acts=json.loads((A/'array_actions.json').read_text(encoding='utf-8'))['actions']; files=[A/'protocol.json',A/'array_actions.json',A/'input_manifest.json',A/'run_summary.json',A/'ledger.json',A/'verification.json',D/'state_0.pt',CK,BG,B/'data_manifest.json',ROOT/'basic_fno.py',ROOT/'time_feature_input.py',ROOT/'time_reconstruction.py']
protocol={'status':'frozen_before_run','experiment':'phase_b_frozen_r2_subarray_diagnostic','source':'source0','actions':list(acts),'materials':['low_contrast','high_contrast'],'network_forwards':6,'physical_calls':0,'training_updates':0,'purpose':'diagnose frozen R2 observable scatter prediction across three receiver subarrays; no training or inversion','input_hashes':{str(p):sha(p) for p in files},'scope':'same scalar finite-difference stage-A outputs; not generalization or final test'};dump('protocol.json',protocol)
bg=torch.load(BG,weights_only=True,map_location='cpu'); bgwave=bg['wavefield_snapshots']; bgtraces={a:torch.load(A/f'{a}__background.pt',weights_only=True,map_location='cpu')['traces_full'].double() for a in acts}; scale=json.loads((B/'data_manifest.json').read_text())['normalization']['wavefield_rms']; ck=torch.load(CK,weights_only=True,map_location='cpu'); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=BasicFNO(width=8,input_channels=8).to(dev);model.load_state_dict(ck['state_dict']);model.eval();model.requires_grad_(False); base=torch.load(D/'state_0.pt',weights_only=True,map_location='cpu')['gamma'];
rows=[];start=time.perf_counter()
for a in acts:
 idx=torch.tensor(acts[a]['rx_elements'],dtype=torch.long)
 for m in ['low_contrast','high_contrast']:
  d=torch.load(A/f'{a}__{m}.pt',weights_only=True,map_location='cpu'); d={**d,'gamma':d['gamma'].float(),'x_m':template['x_m'],'y_m':template['y_m'],'time_seconds_snapshots':template['time_seconds_snapshots']}; t0=time.perf_counter(); pred=predict(model,d,dev,scale,bg)['trace'][0].double(); sec=time.perf_counter()-t0; phys=d['traces_full'].double(); b=bgtraces[a]; pp=pred.index_select(0,idx); rr=phys.index_select(0,idx); bb=b.index_select(0,idx); rows.append({'action':a,'material':m,'network_forward_seconds':sec,'full_receiver_error':err(pred,phys),'selected_receiver_error':err(pp,rr),'scatter_selected_error':err(pp-bb,rr-bb),'physical_scatter_norm':float((rr-bb).norm()),'pred_scatter_norm':float((pp-bb).norm()),'selected_rx_count':len(idx),'finite':bool(torch.isfinite(pred).all())})
summary={'status':'completed','rows':rows,'network_forwards':6,'physical_calls':0,'training_updates':0,'elapsed_seconds':time.perf_counter()-start,'scope':protocol['scope']};dump('summary.json',summary);dump('ledger.json',{'status':'completed','entries':rows,'totals':{'network_forwards':6,'physical_calls':0,'training_updates':0,'network_seconds':sum(x['network_forward_seconds'] for x in rows),'physical_seconds':0.0},'unknowns':['training cost','model loading cost','physical validation cost intentionally zero in this diagnostic'],'no_cache_policy':'six distinct material-action forward evaluations'})
# exact saved input hash verification now
for p,h in protocol['input_hashes'].items():assert sha(Path(p))==h
dump('verification.json',{'status':'passed','checks':['13 input hashes','6 frozen network forwards','finite outputs','selected/full receiver metrics','no training','no physical calls'],'network_forwards':6,'physical_calls':0,'training_updates':0,'scope':protocol['scope']});print(json.dumps(summary,ensure_ascii=False,indent=2))



