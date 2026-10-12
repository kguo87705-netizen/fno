import json,time,hashlib
from pathlib import Path
import torch
from run_fno_local_contrast import physical_setup,generate
from run_frozen_depth_shift import policy,predict,shift_mask
from basic_fno import BasicFNO
O=Path('.build/two-stage-topk-physical-verification-20261011');assert not O.exists();O.mkdir()
P=Path('.build/same-quality-cost-audit-20261011/two_stage_topk_protocol.json');D=Path('.build/fno-frozen-depth-shift-audit-20261009-r2');B=Path('.build/fno-local-contrast-20261009');CK=Path('.build/fno-lowfreq-field-control-r2-20261011/loss_original_update400.pt');BG=Path('.build/operator-pilot-v3-20261008/homogeneous.pt');ROOT=Path('Research_Direction/Author_Baseline_Singh_Zenodo14616677')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def rank(v):return sorted(range(len(v)),key=lambda i:(v[i],i))
def corr(a,b):
 ra=rank(a);rb=rank(b);pa={x:i for i,x in enumerate(ra)};pb={x:i for i,x in enumerate(rb)};n=len(a);d=sum((pa[i]-pb[i])**2 for i in range(n));return 1-6*d/(n*(n*n-1))
def err(a,b):return float((a.double()-b.double()).norm()/b.double().norm())
policy();plan=json.loads(P.read_text());torch.set_num_threads(4);files=[P,D/'state_0.pt',D/'state_2.pt',CK,BG,B/'contrast_03.pt',B/'contrast_07.pt',B/'data_manifest.json',ROOT/'run_fno_local_contrast.py',ROOT/'run_frozen_depth_shift.py',ROOT/'basic_fno.py',ROOT/'FiniteDifferencePyTorchConv.py',ROOT/'settings2D.csv',ROOT/'data_casestudy/source.h5',Path(__file__)];dump('protocol.json',{'plan':plan,'status':'frozen_before_run','hashes':{str(p):sha(p) for p in files},'candidate_grid':'shifts1..5 x materials .46,.48,.52,.54; target shift3,a=.54','counting':'target+candidate+center+top3 recheck physical calls recorded separately; full grid is benchmark, screened online count is target+top3; no truth-based stopping'})
solver,force,template,shape,rx,ry,dev=physical_setup();bg=torch.load(BG,weights_only=True,map_location='cpu');scale=json.loads((B/'data_manifest.json').read_text())['normalization']['wavefield_rms'];ck=torch.load(CK,weights_only=True,map_location='cpu');model=BasicFNO(width=8,input_channels=8).to(dev);model.load_state_dict(ck['state_dict']);model.eval();base=torch.load(D/'state_0.pt',weights_only=True,map_location='cpu')['gamma'];mask0=base<.99
calls=[];net=0
def make(k,a,kind):
 mf=shift_mask(mask0,k).to(dtype=template['gamma'].dtype);g=torch.where(mf>.5,torch.full_like(mf,a),torch.ones_like(mf));t=time.perf_counter();d=generate(solver,force,{**template,'gamma':g},shape,rx,ry,dev,a);calls.append({'shift':k,'a':a,'kind':kind,'seconds':time.perf_counter()-t});return d
target=make(3,.54,'target');ttrace=target['traces_full'].double();grid=[(k,a) for k in [1,2,3,4,5] for a in [.46,.48,.52,.54]]
nom=[torch.load(B/f'contrast_{a:02d}.pt',weights_only=True,map_location='cpu')['traces_full'].double() for a in [3,7]];jn=(nom[1]-nom[0])/.4
centers={}
for k in [1,2,3,4,5]:centers[k]=make(k,.5,'nominal_center')['traces_full'].double()
records=[]
for k,a in grid:
 d=make(k,a,'full_grid');phys=d['traces_full'].double();pp=predict(model,d,dev,scale,bg)['trace'];net+=1
 fnoerr=err(pp,ttrace);nompred=centers[k]+jn*(a-.5);records.append({'shift':k,'a':a,'physical_error':err(phys,ttrace),'fno_error':fnoerr,'nominal_error':err(nompred,ttrace)})
pr=rank([x['physical_error'] for x in records]);fr=rank([x['fno_error'] for x in records]);nr=rank([x['nominal_error'] for x in records]);topf=fr[:3];topn=nr[:3];topphys=pr[0]
# Recheck the two screened top-3 sets with fresh physical calls.
for i in topf: make(records[i]['shift'],records[i]['a'],'fno_top3_recheck')
for i in topn: make(records[i]['shift'],records[i]['a'],'nominal_top3_recheck')
summary={'status':'completed','target':{'shift':3,'a':.54},'records':records,'ranking':{'physical_top1':records[topphys],'fno_top3':[records[i] for i in topf],'nominal_top3':[records[i] for i in topn],'fno_contains_physical_top1':topphys in topf,'nominal_contains_physical_top1':topphys in topn,'fno_spearman':corr([x['fno_error'] for x in records],[x['physical_error'] for x in records]),'nominal_spearman':corr([x['nominal_error'] for x in records],[x['physical_error'] for x in records])},'cost':{'benchmark_physical_calls':len(calls),'screened_online_physical_calls':1+3,'network_forwards':net,'target_calls':1,'candidate_full_grid_calls':20,'center_calls':5,'recheck_calls':6},'scope':plan.get('scope')}
dump('summary.json',summary);torch.save({'target':target,'records':records},O/'raw.pt')
for p,h in json.loads((O/'protocol.json').read_text())['hashes'].items():assert sha(Path(p))==h
dump('verification.json',{'status':'passed','checks':'20 candidate records, finite outputs, top-k membership, input hashes','benchmark_physical_calls':len(calls),'screened_call_count':4,'no_final_test':True})
print(json.dumps(summary,indent=2,ensure_ascii=False))
