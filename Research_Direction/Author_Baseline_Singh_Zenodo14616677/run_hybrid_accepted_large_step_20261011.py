"""Frozen protocol, accepted physical steps, audited calls and saved trace replay."""
import json,time
from pathlib import Path
import torch
from run_fno_local_contrast import physical_setup,generate
from run_frozen_depth_shift import policy
from run_reduced_material_scan import sha
R=Path('.build/fno-known-support-cross-depth-material-response-20261011')
D=Path('.build/fno-depth-interpolation-control-20261009')
B=Path('.build/fno-local-contrast-20261009')
O=Path('.build/hybrid-accepted-large-step-20261011')
def dump(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def normcost(x,y): return float((x.double()-y.double()).norm()/y.double().norm())
def delta(j,r):
 j=j.double().flatten();r=r.double().flatten()
 return max(-.02,min(.02,float(j.dot(r)/j.dot(j))))
policy();assert not O.exists();O.mkdir()
files=[R/'raw.pt',*[D/f'state_{k}.pt' for k in [-1,1]],*[B/f'contrast_{a:02}.pt' for a in [3,7]],Path(__file__)]
protocol={'cases':[[k,a] for k in [-1,1] for a in [.44,.56]],'initial_a':.5,'iterations':3,'methods':['fno_frozen','nominal_secant','physical_frozen','physical_recomputed'],'eps':.001,'clip':.02,'accept':'strict decrease of physical full receiver relative L2; otherwise retain current state; no retries','stopping':'exactly three proposals; truth used only for posthoc parameter error','max_physical_calls':100,'wall_cap_seconds':240,'scope':'known support, larger scalar perturbations, source0, reused development depths; no independent final test','cost_policy':'each route computes own initial physical trace; observation generation separate; frozen physical direction costs 2 solves per route; FNO cached direction would need 2 neural forwards per depth; nominal secant costs 2 historical nominal physical solves shared, no training. Historical training excluded and unknown total prevents total-cost claim','hashes':{str(p):sha(p) for p in files}}
dump(O/'protocol.json',protocol)
start=time.perf_counter();solver,force,template,shape,rx,ry,dev=physical_setup()
raw=torch.load(R/'raw.pt',weights_only=True,map_location='cpu')
anchors=[torch.load(B/f'contrast_{a:02}.pt',weights_only=True,map_location='cpu')['traces_full'].double() for a in [3,7]]
jnom=(anchors[1]-anchors[0])/.4
calls=[];routes=[];observations={}
def physical(mask,a,kind):
 assert time.perf_counter()-start<240 and len(calls)<100
 gamma=torch.where(mask,torch.full_like(template['gamma'],a),torch.ones_like(template['gamma']))
 assert gamma.is_floating_point() and int((gamma<.99).sum())==449
 t=time.perf_counter();d=generate(solver,force,{**template,'gamma':gamma},shape,rx,ry,dev,a)
 trace=d['traces_full'].double();calls.append({'kind':kind,'a':a,'seconds':time.perf_counter()-t});return trace
for depth,target_a in protocol['cases']:
 state=torch.load(D/f'state_{depth}.pt',weights_only=True,map_location='cpu');mask=state['gamma']<.99
 obs=physical(mask,target_a,'observation');key=f'{depth}:{target_a}';observations[key]=obs
 for method in protocol['methods']:
  t=time.perf_counter();n=len(calls);a=.5;cur=physical(mask,a,'initial');initial=cur.clone();hist=[]
  if method=='physical_frozen': j=(physical(mask,.501,'frozen_direction')-physical(mask,.499,'frozen_direction'))/.002
  elif method=='fno_frozen': j=raw['depths'][str(depth)]['epsilons']['0.001']['predictions']['joint_loss_r2']['trace'].double()
  elif method=='nominal_secant': j=jnom
  for it in range(3):
   if method=='physical_recomputed': j=(physical(mask,a+.001,'direction')-physical(mask,a-.001,'direction'))/.002
   step=delta(j,obs-cur);proposal=a+step;trial=physical(mask,proposal,'proposal');before=normcost(cur,obs);after=normcost(trial,obs);accept=after<before
   hist.append({'a_before':a,'proposal':proposal,'step':step,'before':before,'after':after,'accepted':accept,'j':j.clone(),'trial':trial,'calls_so_far':len(calls)-n})
   if accept:a=proposal;cur=trial
  route={'case':key,'method':method,'initial':initial,'history':hist,'final_a':a,'final_full_error':normcost(cur,obs),'final_scatter_error':None,'parameter_error':abs(a-target_a),'online_physical_calls':len(calls)-n,'seconds':time.perf_counter()-t}
  routes.append(route);print(key,method,route['final_full_error'],route['parameter_error'],route['online_physical_calls'],flush=True)
rawout={'routes':routes,'observations':observations,'calls':calls};torch.save(rawout,O/'raw.pt')
# Independent replay of saved proposals, decisions, and numerical updates (no new solves).
saved=torch.load(O/'raw.pt',weights_only=True,map_location='cpu')
for r in saved['routes']:
 obs=saved['observations'][r['case']];cur=r['initial'];a=.5
 for h in r['history']:
  assert abs(delta(h['j'],obs-cur)-h['step'])<1e-12
  assert normcost(cur,obs)==h['before'] and normcost(h['trial'],obs)==h['after']
  assert h['accepted']==(h['after']<h['before']) and h['proposal']==a+h['step']
  if h['accepted']:a=h['proposal'];cur=h['trial']
 assert a==r['final_a'] and normcost(cur,obs)==r['final_full_error']
 assert torch.isfinite(cur).all()
for p,h in protocol['hashes'].items():assert sha(Path(p))==h
summary={'routes':[{k:v for k,v in r.items() if k not in ['initial','history']}|{'trajectory':[ {k:v for k,v in h.items() if k not in ['j','trial']} for h in r['history']]} for r in routes],'actual_physical_calls':len(calls),'observation_calls':4,'seconds':time.perf_counter()-start,'scope':protocol['scope']}
dump(O/'summary.json',summary)
dump(O/'verification.json',{'status':'passed','checked':'all saved trace residuals, LS steps, accepts and final states recomputed; input hashes unchanged; not independent physical replay','routes':len(routes),'physical_calls':len(calls),'raw_sha256':sha(O/'raw.pt')})

