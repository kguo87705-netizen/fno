import json,hashlib,sys,time
from pathlib import Path
import torch,pandas as pd
sys.path.insert(0,'Research_Direction/Author_Baseline_Singh_Zenodo14616677')
from run_fno_local_contrast import physical_setup
OUT=Path('.build/tx-action-material-scan-20261011-r2');assert not OUT.exists();OUT.mkdir()
R=Path.cwd();ROOT=R/'Research_Direction/Author_Baseline_Singh_Zenodo14616677';OLD=R/'.build/operator-pilot-v3-20261008';STATE=R/'.build/fno-frozen-depth-shift-audit-20261009-r2/state_0.pt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
solver,_,template,shape,rx,ry,device=physical_setup();cfg=pd.read_csv(ROOT/'settings2D.csv').iloc[0];nx,ny,n=shape;force_all=torch.tensor(pd.read_hdf(ROOT/'data_casestudy/source.h5').values).reshape(4,1,nx+3,ny+3,n).clone().to(device);mask=torch.load(STATE,weights_only=True,map_location='cpu')['gamma']<.99
actions={'full_tx':[0,1,2,3],'left_tx':[0,1],'right_tx':[2,3]};materials={f'gamma_{a:.2f}':a for a in [.46,.48,.50,.52,.54]}
files=[ROOT/'settings2D.csv',ROOT/'data_casestudy/source.h5',ROOT/'FiniteDifferencePyTorchConv.py',ROOT/'run_fno_local_contrast.py',OLD/'synthetic_workpiece_000.pt',STATE]
dump('protocol.json',{'status':'frozen_before_run','experiment':'tx_action_material_scan','actions':actions,'materials':materials,'matrix':'3 transmit actions x 5 materials = 15 logical tasks','physical_calls_expected':40,'network_forwards':0,'training_updates':0,'metrics':['per_source_scatter_norm','per_source_low_high response','action ranking'],'input_hashes':{str(p):sha(p) for p in files},'scope':'true transmit source selection, known center support, scalar finite-difference; no proxy or final test'})
dump('array_actions.json',{'source_count':4,'actions':actions,'receiver_count':24,'source_xy_m':template['source_xy_m'].tolist()})
start=time.perf_counter();rows=[]
for aname,srcs in actions.items():
 for mname,a in materials.items():
  t0=time.perf_counter(); trs=[]
  mat=torch.where(mask,torch.full_like(mask,a,dtype=template['gamma'].dtype),torch.ones_like(template['gamma']));gamma=torch.ones(1,1,nx+3,ny+3,device=device);gamma[0,0,1:-1,1:-1]=mat.to(device);z=torch.zeros_like(gamma)
  try:
   for si in srcs:
    fi=force_all[si:si+1];w=solver.forwardNSteps(z,z.clone(),gamma,fi,nx,ny,n,1,device);tr=w[:,rx,ry,1:].cpu().double();assert torch.isfinite(tr).all();trs.append(tr[0])
   traces=torch.stack(trs);torch.save({'action':aname,'source_indices':srcs,'material':mname,'material_value':a,'traces_full':traces,'gamma':mat.double(),'receiver_xy_m':template['receiver_xy_m'],'source_xy_m':template['source_xy_m'].index_select(0,torch.tensor(srcs))},OUT/f'{aname}__{mname}.pt');rows.append({'action':aname,'material':mname,'material_value':a,'source_indices':srcs,'physical_calls':len(srcs),'physical_seconds':time.perf_counter()-t0,'status':'completed'})
  except Exception as e:
   (OUT/f'{aname}__{mname}__failure.json').write_text(json.dumps({'status':'failed','error':repr(e)},ensure_ascii=False,indent=2),encoding='utf-8');rows.append({'action':aname,'material':mname,'material_value':a,'source_indices':srcs,'physical_calls':len(srcs),'physical_seconds':time.perf_counter()-t0,'status':'failed','error':repr(e)})
summary={'status':'completed','rows':rows,'logical_tasks':15,'completed_tasks':sum(r['status']=='completed' for r in rows),'failed_tasks':sum(r['status']=='failed' for r in rows),'physical_calls':sum(r['physical_calls'] for r in rows),'network_forwards':0,'training_updates':0,'elapsed_seconds':time.perf_counter()-start,'scope':'true transmit source selection, scalar finite-difference, no neural proxy'};dump('summary.json',summary);dump('ledger.json',{'status':'completed','entries':rows,'totals':{'logical_tasks':15,'physical_calls':summary['physical_calls'],'network_forwards':0,'training_updates':0},'unknowns':['separate postprocess time','hardware transmit time','training cost']})
for aname,srcs in actions.items():
 for mname in materials:
  d=torch.load(OUT/f'{aname}__{mname}.pt',weights_only=True,map_location='cpu');assert d['traces_full'].shape==(len(srcs),24,2000);assert torch.isfinite(d['traces_full']).all()
dump('verification.json',{'status':'passed','checks':['15 logical tasks','40 physical source solves','source indices','shape and finite outputs','0 network','0 training'],'physical_calls':summary['physical_calls'],'scope':summary['scope']});print(json.dumps({'status':'completed','logical_tasks':15,'physical_calls':summary['physical_calls'],'failed':summary['failed_tasks'],'elapsed_seconds':summary['elapsed_seconds']},ensure_ascii=False,indent=2))

