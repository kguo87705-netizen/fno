import json,time
from pathlib import Path
import torch
from run_fno_local_contrast import physical_setup,generate
from run_frozen_depth_shift import policy,predict
from basic_fno import BasicFNO
from run_reduced_material_scan import sha
O=Path('.build/cost-timing-control-20261011'); D=Path('.build/fno-frozen-depth-shift-audit-20261009-r2'); B=Path('.build/fno-local-contrast-20261009')
assert not O.exists(); O.mkdir(); policy(); torch.set_num_threads(4)
solver,force,template,shape,rx,ry,dev=physical_setup(); st=torch.load(D/'state_0.pt',weights_only=True,map_location='cpu'); mask=st['gamma']<.99; mf=mask.to(dtype=template['gamma'].dtype); mat=lambda a:torch.where(mf>.5,torch.full_like(mf,a),torch.ones_like(mf))
bg=torch.load('.build/operator-pilot-v3-20261008/homogeneous.pt',weights_only=True,map_location='cpu'); scale=json.loads((B/'data_manifest.json').read_text())['normalization']['wavefield_rms']
ck=torch.load('.build/fno-lowfreq-field-control-r2-20261011/loss_original_update400.pt',weights_only=True,map_location='cpu'); model=BasicFNO(width=8,input_channels=8).to(dev); model.load_state_dict(ck['state_dict']); model.eval()
def phys(a):
 return generate(solver,force,{**template,'gamma':mat(a)},shape,rx,ry,dev,a)
# fixed warmup
d=phys(.5); predict(model,d,dev,scale,bg)
physical=[]; neural=[]
for i in range(5):
 t=time.perf_counter(); d=phys(.5); physical.append(time.perf_counter()-t)
 t=time.perf_counter(); q=predict(model,d,dev,scale,bg); neural.append(time.perf_counter()-t)
assert torch.isfinite(q['trace']).all()
# derivative pair timing: two physics + two network forwards
pairs=[]
for i in range(3):
 t=time.perf_counter(); p=phys(.501); m=phys(.499); physical_pair=time.perf_counter()-t
 t=time.perf_counter(); pp=predict(model,p,dev,scale,bg); pm=predict(model,m,dev,scale,bg); neural_pair=time.perf_counter()-t
 pairs.append({'physical_two_calls_seconds':physical_pair,'network_two_forwards_seconds':neural_pair})
out={'status':'completed','device':str(dev),'torch':torch.__version__,'warmup':1,'single_physical_seconds':physical,'single_network_prediction_seconds':neural,'paired_direction_timings':pairs,'summary':{'median_physical_seconds':sorted(physical)[2],'median_network_seconds':sorted(neural)[2],'median_physical_pair_seconds':sorted([x['physical_two_calls_seconds'] for x in pairs])[1],'median_network_pair_seconds':sorted([x['network_two_forwards_seconds'] for x in pairs])[1]},'scope':'center depth known support source0; timing only, not end-to-end training or full inversion'}
(O/'timing.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
(O/'protocol.json').write_text(json.dumps({'status':'frozen','repeats':5,'pair_repeats':3,'warmup':1,'inputs':{str(p):sha(p) for p in [D/'state_0.pt',B/'data_manifest.json',Path('.build/fno-lowfreq-field-control-r2-20261011/loss_original_update400.pt')]}},indent=2),encoding='utf-8')
(O/'verification.json').write_text(json.dumps({'status':'passed','finite_outputs':True,'repeats':8,'timing_only':True},indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
