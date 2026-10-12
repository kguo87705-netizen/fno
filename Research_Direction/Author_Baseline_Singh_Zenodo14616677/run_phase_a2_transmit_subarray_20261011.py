import json,hashlib,sys,time
from pathlib import Path
import torch,pandas as pd
sys.path.insert(0,'Research_Direction/Author_Baseline_Singh_Zenodo14616677')
from run_fno_local_contrast import physical_setup
import FiniteDifferencePyTorchConv as FD
OUT=Path('.build/phase-a2-transmit-subarray-20261011-r2');assert not OUT.exists();OUT.mkdir()
R=Path.cwd();ROOT=R/'Research_Direction/Author_Baseline_Singh_Zenodo14616677';OLD=R/'.build/operator-pilot-v3-20261008';STATE=R/'.build/fno-frozen-depth-shift-audit-20261009-r2/state_0.pt'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def err(a,b):return float((a.double()-b.double()).norm()/b.double().norm())
# setup and full source bank
solver,_,template,shape,rx,ry,device=physical_setup(); cfg=pd.read_csv(ROOT/'settings2D.csv').iloc[0]; nx,ny,n=shape
force_all=torch.tensor(pd.read_hdf(ROOT/'data_casestudy/source.h5').values).reshape(4,1,nx+3,ny+3,n).clone().to(device)
mask=torch.load(STATE,weights_only=True,map_location='cpu')['gamma']<.99
actions={'full_tx':list(range(4)),'left_tx':[0,1],'right_tx':[2,3]}; materials={'background':1.0,'gamma_048':.48,'gamma_052':.52}
files=[ROOT/'settings2D.csv',ROOT/'data_casestudy/source.h5',ROOT/'FiniteDifferencePyTorchConv.py',ROOT/'run_fno_local_contrast.py',OLD/'synthetic_workpiece_000.pt',STATE]
dump('array_actions.json',{'source_count':4,'receiver_count':len(rx),'actions':actions,'receiver_mode':'all24','source_mode':'selected_source_rows','source_xy_m':template['source_xy_m'].tolist()})
dump('protocol.json',{'status':'frozen_before_run','experiment':'phase_a2_true_transmit_subarray','actions':actions,'materials':materials,'matrix':'3 transmit actions x 3 materials = 9 physical tasks','source0_legacy':'previous phase used force[0:1]; this phase uses selected source rows explicitly','network_forwards':0,'training_updates':0,'max_seconds':300,'input_hashes':{str(p):sha(p) for p in files},'metrics':['per_source_receiver_shape','background_scatter_difference','finite_outputs','source_index_consistency'],'failure_rule':'preserve failure.json and stop task; do not delete failures','scope':'scalar finite-difference, known support, true transmit source selection; not final test'})
start=time.perf_counter();rows=[]
for aname,srcs in actions.items():
 for mname,a in materials.items():
  t0=time.perf_counter(); status='completed';errtxt=None
  try:
   mat=torch.where(mask,torch.full_like(mask,dtype=template['gamma'].dtype,fill_value=a),torch.ones_like(template['gamma']))
   gamma=torch.ones(1,1,nx+3,ny+3,device=device);gamma[0,0,1:-1,1:-1]=mat.to(device);z=torch.zeros_like(gamma)
   f=force_all.index_select(0,torch.tensor(srcs,device=device));z_batch=z.expand(len(srcs),-1,-1,-1).clone(); gamma_batch=gamma.expand(len(srcs),-1,-1,-1).clone(); w=solver.forwardNSteps(z_batch,z_batch.clone(),gamma_batch,f,nx,ny,n,1,device);tr=w[:,rx,ry,1:].cpu().double()
   assert torch.isfinite(tr).all();out={'action':aname,'source_indices':srcs,'material':mname,'material_value':a,'traces_full':tr,'receiver_xy_m':template['receiver_xy_m'],'source_xy_m':template['source_xy_m'].index_select(0,torch.tensor(srcs)),'gamma':mat.double()};torch.save(out,OUT/f'{aname}__{mname}.pt')
   rows.append({'run_id':f'{aname}__{mname}','action':aname,'material':mname,'source_indices':srcs,'physical_calls':1,'network_forwards':0,'training_updates':0,'physical_seconds':time.perf_counter()-t0,'status':status})
  except Exception as e:
   status='failed';errtxt=repr(e);(OUT/f'{aname}__{mname}__failure.json').write_text(json.dumps({'status':status,'error':errtxt,'elapsed_seconds':time.perf_counter()-t0},ensure_ascii=False,indent=2),encoding='utf-8');rows.append({'run_id':f'{aname}__{mname}','action':aname,'material':mname,'source_indices':srcs,'physical_calls':1,'network_forwards':0,'training_updates':0,'physical_seconds':time.perf_counter()-t0,'status':status,'error':errtxt})
summary={'status':'completed','rows':rows,'physical_tasks':9,'completed_tasks':sum(r['status']=='completed' for r in rows),'failed_tasks':sum(r['status']=='failed' for r in rows),'new_physical_calls':9,'network_forwards':0,'training_updates':0,'elapsed_seconds':time.perf_counter()-start,'scope':'true transmit source selection, scalar finite-difference, not final test'};dump('run_summary.json',summary);dump('ledger.json',{'status':'completed','entries':rows,'totals':{'physical_calls':9,'network_forwards':0,'training_updates':0},'unknowns':['separate solver vs postprocess time','hardware transmit time','training cost']})
for p,h in json.loads((OUT/'protocol.json').read_text())['input_hashes'].items():assert sha(Path(p))==h
# structural checks
for aname,srcs in actions.items():
 for m in materials:
  d=torch.load(OUT/f'{aname}__{m}.pt',weights_only=True,map_location='cpu');assert d['traces_full'].shape==(len(srcs),24,2000);assert torch.isfinite(d['traces_full']).all();assert d['source_indices']==srcs
dump('verification.json',{'status':'passed','checks':['9 tasks','3 true transmit source selections','source indices and shapes','finite outputs','input hashes','0 network forwards','0 training updates'],'new_physical_calls':9,'scope':summary['scope']});print(json.dumps({'status':'completed','completed':summary['completed_tasks'],'failed':summary['failed_tasks'],'elapsed_seconds':summary['elapsed_seconds']},ensure_ascii=False,indent=2))

