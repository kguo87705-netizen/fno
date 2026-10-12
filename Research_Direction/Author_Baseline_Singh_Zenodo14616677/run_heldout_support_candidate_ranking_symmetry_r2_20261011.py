import json,time,math,hashlib
from pathlib import Path
import torch
from run_fno_local_contrast import physical_setup,generate
from run_frozen_depth_shift import policy,predict,shift_mask
from basic_fno import BasicFNO
O=Path('.build/heldout-support-candidate-ranking-symmetry-r2-20261011');assert not O.exists();O.mkdir()
P=Path('.build/same-quality-cost-audit-20261011/heldout_support_candidate_ranking_symmetry_protocol.json')
ROOT=Path('Research_Direction/Author_Baseline_Singh_Zenodo14616677'); CK=Path('.build/fno-lowfreq-field-control-r2-20261011/loss_original_update400.pt'); BG=Path('.build/operator-pilot-v3-20261008/homogeneous.pt'); D=Path('.build/fno-frozen-depth-shift-audit-20261009-r2')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def metric(p,r):return float((p.double()-r.double()).norm()/r.double().norm())
def rank(vals):return sorted(range(len(vals)),key=lambda i:(vals[i],i))
def rho(a,b):
 n=len(a);pa=rank(a);pb=rank(b);ra=[0]*n;rb=[0]*n
 for i,j in enumerate(pa):ra[j]=i
 for i,j in enumerate(pb):rb[j]=i
 d=sum((ra[i]-rb[i])**2 for i in range(n));return 1-6*d/(n*(n*n-1))
def kendall(a,b):
 n=len(a);x=y=0
 for i in range(n):
  for j in range(i+1,n):
   x+=((a[i]-a[j])*(b[i]-b[j])>0);y+=((a[i]-a[j])*(b[i]-b[j])<0)
 return (x-y)/(n*(n-1)/2)
plan=json.loads(P.read_text());policy();torch.set_num_threads(4)
files=[P,CK,BG,D/'state_0.pt',ROOT/'run_fno_local_contrast.py',ROOT/'run_frozen_depth_shift.py',ROOT/'basic_fno.py',ROOT/'FiniteDifferencePyTorchConv.py',ROOT/'settings2D.csv',ROOT/'data_casestudy/source.h5',Path(__file__)]
protocol={'plan':plan,'status':'frozen_before_generation','hashes':{str(p):sha(p) for p in files},'physical_calls_expected':16,'network_forwards_expected':10,'truth_policy':'target shift/material used only for observation generation and posthoc metrics; no threshold tuning','scope':plan['scope']}
dump('protocol.json',protocol)
solver,force,template,shape,rx,ry,dev=physical_setup(); bg=torch.load(BG,weights_only=True,map_location='cpu'); scale=json.loads(Path('.build/fno-local-contrast-20261009/data_manifest.json').read_text())['normalization']['wavefield_rms']
ck=torch.load(CK,weights_only=True,map_location='cpu');model=BasicFNO(width=8,input_channels=8).to(dev);model.load_state_dict(ck['state_dict']);model.eval();model.requires_grad_(False)
base=torch.load(D/'state_0.pt',weights_only=True,map_location='cpu')['gamma'];mask0=base<.99;calls=[];net=0
def make(k,a):
 mask=shift_mask(mask0,k).to(dtype=template['gamma'].dtype);g=torch.where(mask>.5,torch.full_like(mask,a),torch.ones_like(mask))
 t=time.perf_counter();d=generate(solver,force,{**template,'gamma':g},shape,rx,ry,dev,a);calls.append({'shift':k,'a':a,'seconds':time.perf_counter()-t});return d
target=make(3,.47);ttrace=target['traces_full'].double(); candidates=[]; fno=[]; physical=[]; zero=bg['traces_full'][0:1].double()
for k in plan['candidate_depth_rows']:
 for a in plan['candidate_materials']:
  d=make(k,a);ref=d['traces_full'].double();physical.append(metric(ref,ttrace))
  pred=predict(model,d,dev,scale,bg)['trace'];net+=1;fno.append(metric(pred,ttrace))
  candidates.append({'shift':k,'a':a,'physical_error':physical[-1],'fno_error':fno[-1],'zero_error':metric(zero,ttrace)})
pr=rank(physical);fr=rank(fno);zr=rank([c['zero_error'] for c in candidates])
result={'status':'completed','target':{'shift':3,'a':plan['target']['material']},'candidates':candidates,'rank_metrics':{'spearman_fno_vs_physical':rho(fno,physical),'kendall_fno_vs_physical':kendall(fno,physical),'physical_top1':candidates[pr[0]],'fno_top1':candidates[fr[0]],'physical_top3':[candidates[i] for i in pr[:3]],'fno_top3':[candidates[i] for i in fr[:3]],'fno_top3_contains_physical_top1':pr[0] in fr[:3],'zero_top1':candidates[zr[0]]},'cost':{'physical_calls':len(calls),'network_forwards':net,'target_calls':1,'candidate_calls':len(candidates),'elapsed_seconds':sum(x['seconds'] for x in calls)},'scope':plan['scope']}
dump('summary.json',result);torch.save({'target':target,'candidates':candidates,'physical_errors':physical,'fno_errors':fno},O/'raw.pt')
for p,h in protocol['hashes'].items():assert sha(Path(p))==h
dump('verification.json',{'status':'passed','checks':'candidate count, finite traces, rank metrics, and input hashes','physical_calls':len(calls),'network_forwards':net,'no_independent_final_test':True})
print(json.dumps(result,indent=2,ensure_ascii=False))
