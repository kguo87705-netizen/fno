"""Prospective observable-stopping scalar comparison, with saved numerical replay."""
import json,time,hashlib
from pathlib import Path
import torch
from run_fno_local_contrast import physical_setup,generate
from run_frozen_depth_shift import policy,predict
from basic_fno import BasicFNO
O=Path('.build/observable-stop-comparison-20261011'); P=Path('.build/same-quality-cost-audit-20261011/next_protocol.json'); B=Path('.build/fno-local-contrast-20261009'); D=Path('.build/fno-depth-interpolation-control-20261009'); ROOT=Path(__file__).parent
CK=Path('.build/fno-lowfreq-field-control-r2-20261011/loss_original_update400.pt'); BG=Path('.build/operator-pilot-v3-20261008/homogeneous.pt')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):(O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sync():
 if torch.cuda.is_available():torch.cuda.synchronize()
def clock():sync();return time.perf_counter()
def err(x,y):return float((x.double()-y.double()).norm()/y.double().norm())
def step(j,r):
 j=j.double().flatten();r=r.double().flatten();return max(-.02,min(.02,float(j.dot(r)/j.dot(j))))
assert not O.exists();policy();plan=json.loads(P.read_text());O.mkdir()
files=[P,CK,BG,B/'data_manifest.json',B/'contrast_03.pt',B/'contrast_07.pt',D/'state_-1.pt',D/'state_1.pt',Path(__file__),ROOT/'run_fno_local_contrast.py',ROOT/'run_frozen_depth_shift.py',ROOT/'FiniteDifferencePyTorchConv.py',ROOT/'basic_fno.py',ROOT/'settings2D.csv',ROOT/'data_casestudy/source.h5']
protocol={'plan':plan,'frozen_before_run':True,'hashes':{str(p):sha(p) for p in files},'execution':'one timed route per method/case, alternate method order; no extra warmup; timings descriptive; global physical cap 56 includes observations, so cap may censor last routes; 240 second compute cap','truth_policy':'target a only observation generation and final reporting; solve() receives obs and mask, no target a','direction_policy':'each FNO case two new network calls on gamma only; no physical labels to generate direction; nominal slope from historical two anchors; shared direction reuse estimated separately'}
dump('protocol.json',protocol);start=clock();solver,force,template,shape,rx,ry,dev=physical_setup();setup_seconds=clock()-start
bg=torch.load(BG,weights_only=True,map_location='cpu');scale=json.loads((B/'data_manifest.json').read_text())['normalization']['wavefield_rms']
t=clock();model=BasicFNO(width=8,input_channels=8).to(dev);model.load_state_dict(torch.load(CK,weights_only=True,map_location='cpu')['state_dict']);model.eval();model.requires_grad_(False);load_seconds=clock()-t
t=clock();anchors=[torch.load(B/f'contrast_{a:02}.pt',weights_only=True,map_location='cpu')['traces_full'].double() for a in [3,7]];jn=(anchors[1]-anchors[0])/.4;anchor_load_seconds=clock()-t
calls=[];netcalls=0;raw={'observations':{},'routes':[]}
def phys(mask,a,kind):
 if len(calls)>=56:raise RuntimeError('global physical cap')
 assert clock()-start<240
 gamma=torch.where(mask,torch.full_like(template['gamma'],a),torch.ones_like(template['gamma']));assert gamma.is_floating_point() and int((gamma<.99).sum())==449
 t=clock();d=generate(solver,force,{**template,'gamma':gamma},shape,rx,ry,dev,a);x=d['traces_full'].double();calls.append({'kind':kind,'a':a,'seconds':clock()-t});return x
nom=torch.load(B/'contrast_05.pt',weights_only=True,map_location='cpu')
def solve(mask,obs,method):
 global netcalls
 t=clock();n=len(calls);a=.5;cur=phys(mask,a,'initial');initial=cur.clone();e0=err(cur,obs);limit=e0*.001;hist=[];td=clock()
 if method=='frozen_r2_direction':
  preds=[]
  for a0 in [.501,.499]:
   gamma=torch.where(mask,torch.full_like(template['gamma'],a0),torch.ones_like(template['gamma']))
   preds.append(predict(model,{**nom,'gamma':gamma},dev,scale,bg)['trace']);netcalls+=1
  j=(preds[0]-preds[1])/.002
 else:j=jn
 direction_seconds=clock()-td
 for i in range(6):
  if err(cur,obs)<=limit or len(calls)>=56:break
  da=step(j,obs-cur);candidate=a+da;trial=phys(mask,candidate,'candidate');before=err(cur,obs);after=err(trial,obs);accept=after<before
  hist.append({'a_before':a,'candidate_a':candidate,'step':da,'before':before,'after':after,'accepted':accept,'trial':trial})
  if accept:a=candidate;cur=trial
 elapsed=clock()-t
 return {'method':method,'initial':initial,'direction':j,'history':hist,'final_a':a,'full_error':err(cur,obs),'scatter_error':float((cur-obs).norm()/(obs-bg['traces_full'][0:1].double()).norm()),'initial_error':e0,'threshold':limit,'reached':err(cur,obs)<=limit,'online_seconds':elapsed,'direction_seconds':direction_seconds,'physical_calls':len(calls)-n,'iterations':len(hist),'rejections':sum(not h['accepted'] for h in hist)}
for i,(depth,target) in enumerate(plan['cases']):
 state=torch.load(D/f'state_{depth}.pt',weights_only=True,map_location='cpu');mask=state['gamma']<.99;key=f'{depth}:{target}'
 obs=phys(mask,target,'observation');raw['observations'][key]=obs
 order=plan['methods'] if i%2==0 else list(reversed(plan['methods']))
 for method in order:
  route=solve(mask,obs,method);route.update(case=key,parameter_error=abs(route['final_a']-target));raw['routes'].append(route)
  print(key,method,route['iterations'],route['physical_calls'],round(route['online_seconds'],3),route['reached'],route['parameter_error'],flush=True)
  torch.save(raw,O/'progress.pt')
raw.update(calls=calls,network_calls=netcalls);torch.save(raw,O/'raw.pt')
# Replay all stored traces and stopping decisions; no physical rerun.
for r in raw['routes']:
 obs=raw['observations'][r['case']];cur=r['initial'];a=.5
 for h in r['history']:
  assert err(cur,obs)>r['threshold']
  assert abs(step(r['direction'],obs-cur)-h['step'])<1e-12
  assert h['candidate_a']==a+h['step'] and h['before']==err(cur,obs) and h['after']==err(h['trial'],obs)
  assert h['accepted']==(h['after']<h['before'])
  if h['accepted']:cur=h['trial'];a=h['candidate_a']
 assert a==r['final_a'] and err(cur,obs)==r['full_error'] and r['reached']==(err(cur,obs)<=r['threshold'])
assert len(calls)<=56 and netcalls==8
for p,h in protocol['hashes'].items():assert sha(Path(p))==h
rows=[{k:v for k,v in r.items() if k not in ['initial','direction','history']} for r in raw['routes']]
dump('summary.json',{'rows':rows,'actual_physical_calls':len(calls),'network_calls':netcalls,'setup_seconds':setup_seconds,'model_load_seconds':load_seconds,'anchor_load_seconds':anchor_load_seconds,'wall_seconds':clock()-start,'scope':plan['scope'],'note':'same observable stopping tolerance, material truth only posthoc; single timing per route, no speed significance or total-cost claim'})
dump('verification.json',{'status':'passed','checks':'saved trace residuals, LS proposal algebra, acceptance, first threshold crossing, final a, and input hashes; no independent physical replay','raw_sha256':sha(O/'raw.pt')})
