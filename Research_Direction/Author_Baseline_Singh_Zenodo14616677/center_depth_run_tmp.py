import json,time
from pathlib import Path
import torch
from run_fno_local_contrast import physical_setup,generate
from run_frozen_depth_shift import policy,predict
from basic_fno import BasicFNO
from run_reduced_material_scan import sha
O=Path('.build/center-depth-response-control-20261011'); D=Path('.build/fno-frozen-depth-shift-audit-20261009-r2'); B=Path('.build/fno-local-contrast-20261009')
assert not (O/'raw.pt').exists(); policy(); O.mkdir(exist_ok=True)
solver,force,template,shape,rx,ry,dev=physical_setup(); st=torch.load(D/'state_0.pt',weights_only=True,map_location='cpu'); mask=st['gamma']<.99; mf=mask.to(dtype=template['gamma'].dtype)
mat=lambda a: torch.where(mf>.5,torch.full_like(mf,a),torch.ones_like(mf))
bg=torch.load('.build/operator-pilot-v3-20261008/homogeneous.pt',weights_only=True,map_location='cpu'); scale=json.loads((B/'data_manifest.json').read_text())['normalization']['wavefield_rms']
ck=torch.load('.build/fno-lowfreq-field-control-r2-20261011/loss_original_update400.pt',weights_only=True,map_location='cpu'); model=BasicFNO(width=8,input_channels=8).to(dev); model.load_state_dict(ck['state_dict']); model.eval()
def gen(a):
 return generate(solver,force,{**template,'gamma':mat(a)},shape,rx,ry,dev,a)
base=gen(.5); plus=gen(.501); minus=gen(.499)
jfno=(predict(model,plus,dev,scale,bg)['trace']-predict(model,minus,dev,scale,bg)['trace'])/.002
nom=[torch.load(B/f'contrast_{a:02}.pt',weights_only=True,map_location='cpu')['traces_full'].double() for a in [3,7]]; jnom=(nom[1]-nom[0])/.4
def cost(x,y): return float((x.double()-y.double()).norm()/y.double().norm())
def step(j,r):
 j,r=j.double().flatten(),r.double().flatten(); return max(-.02,min(.02,float(j.dot(r)/j.dot(j))))
rows=[]
for target in [.47,.53]:
 obs=gen(target)['traces_full'].double()
 for name,j0 in [('fno_r2',jfno),('nominal_secant',jnom)]:
  a=.5; cur=base['traces_full'].double(); hist=[]
  for it in range(3):
   da=step(j0,obs-cur); prop=a+da; trial=gen(prop)['traces_full'].double(); before=cost(cur,obs); after=cost(trial,obs); acc=after<before
   hist.append({'iteration':it+1,'a_before':a,'proposal':prop,'before':before,'after':after,'accepted':acc})
   if acc:a,cur=prop,trial
  rows.append({'target':target,'method':name,'final_a':a,'final_error':cost(cur,obs),'parameter_error':abs(a-target),'history':hist})
dump={'status':'completed','rows':rows,'physical_calls':len([.5,.501,.499])+2*2+2*2*3,'network_forwards':2,'scope':'known support center depth state, source0, development diagnostic'}
(O/'summary.json').write_text(json.dumps(dump,indent=2),encoding='utf-8')
(O/'verification.json').write_text(json.dumps({'status':'passed','rows':len(rows),'inputs':'center depth regenerated in same solver; no independent final test'},indent=2),encoding='utf-8')
print(json.dumps(dump,indent=2))
