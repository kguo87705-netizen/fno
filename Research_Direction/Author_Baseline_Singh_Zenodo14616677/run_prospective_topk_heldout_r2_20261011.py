import json,time,sys
from pathlib import Path
import torch
sys.path.insert(0,'Research_Direction/Author_Baseline_Singh_Zenodo14616677')
from run_fno_local_contrast import physical_setup,generate
from run_frozen_depth_shift import policy,shift_mask
from basic_fno import BasicFNO
from time_feature_input import encode_time_features
from time_reconstruction import cubic_time_reconstruction
O=Path('.build/prospective-topk-heldout-r2-20261011');assert not O.exists();O.mkdir()
D=Path('.build/fno-frozen-depth-shift-audit-20261009-r2');B=Path('.build/fno-local-contrast-20261009');CK=Path('.build/fno-lowfreq-field-control-r2-20261011/loss_original_update400.pt');BG=Path('.build/operator-pilot-v3-20261008/homogeneous.pt')
def dump(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def err(a,b):return float((a.double()-b.double()).norm()/b.double().norm())
def rank(v):return sorted(range(len(v)),key=lambda i:(v[i],i))
policy();torch.set_num_threads(4);solver,force,t,shape,rx,ry,dev=physical_setup();bg=torch.load(BG,weights_only=True,map_location='cpu');scale=json.loads((B/'data_manifest.json').read_text())['normalization']['wavefield_rms'];ck=torch.load(CK,weights_only=True,map_location='cpu');m=BasicFNO(width=8,input_channels=8).to(dev);m.load_state_dict(ck['state_dict']);m.eval();mask=torch.load(D/'state_0.pt',weights_only=True,map_location='cpu')['gamma']<.99
def mk(k,a,kind):
 mf=shift_mask(mask,k).to(dtype=t['gamma'].dtype);g=torch.where(mf>.5,torch.full_like(mf,a),torch.ones_like(mf));return generate(solver,force,{**t,'gamma':g},shape,rx,ry,dev,a)
def px(k,a):
 mf=shift_mask(mask,k).to(dtype=t['gamma'].dtype);g=torch.where(mf>.5,torch.full_like(mf,a),torch.ones_like(mf));s={'gamma':g,'source_xy_m':t['source_xy_m'][0],'x_m':t['x_m'],'y_m':t['y_m'],'time_seconds':t['time_seconds_snapshots']};ids=t['receiver_physical_indices'].to(dev)
 with torch.no_grad():z=m(encode_time_features(s,gamma=g.to(dev),mode='zero')[None])[0,0]*scale;tr=cubic_time_reconstruction(z[ids[:,0],ids[:,1],:],s['time_seconds'].to(dev),torch.arange(1,2001,device=dev,dtype=torch.float64)*30e-9)[None].double().cpu()+bg['traces_full'][0:1].double();return tr
protocol={'status':'frozen_before_run','target':{'shift':2,'a':.44},'candidate_grid':'shifts1..5 x materials .44,.48,.52,.56','proxy_keep':3,'physical_verify':3,'budget':{'network_forwards':20,'physical_calls':12},'scope':'prospective heldout support/material batch; no full-grid benchmark; not final test'};dump('protocol.json',protocol)
obs=mk(2,.44,'target')['traces_full'].double();cent={k:mk(k,.5,'center')['traces_full'].double() for k in [1,2,3,4,5]};grid=[(k,a) for k in [1,2,3,4,5] for a in [.44,.48,.52,.56]];d03=torch.load(B/'contrast_03.pt',weights_only=True,map_location='cpu')['traces_full'].double();d07=torch.load(B/'contrast_07.pt',weights_only=True,map_location='cpu')['traces_full'].double();delta=(d07-d03)/.4;recs=[]
for k,a in grid:
 p=px(k,a);recs.append({'shift':k,'a':a,'proxy_error':err(p,obs),'nominal_error':err(cent[k]+delta*(a-.5),obs)})
tf=rank([r['proxy_error'] for r in recs])[:3];tn=rank([r['nominal_error'] for r in recs])[:3];ver=[]
for lab,inds in [('r2_top3',tf),('nominal_top3',tn)]:
 for i in inds:
  r=recs[i];e=err(mk(r['shift'],r['a'],lab)['traces_full'].double(),obs);ver.append({'method':lab,'shift':r['shift'],'a':r['a'],'physical_error':e,**r})
summary={'status':'completed','target':protocol['target'],'proxy_top3':[recs[i] for i in tf],'nominal_top3':[recs[i] for i in tn],'physical_verification':ver,'best':{'r2':min((x for x in ver if x['method']=='r2_top3'),key=lambda x:x['physical_error']),'nominal':min((x for x in ver if x['method']=='nominal_top3'),key=lambda x:x['physical_error'])},'cost':{'network_forwards':20,'physical_calls':12,'target':1,'centers':5,'rechecks':6},'scope':protocol['scope']};dump('summary.json',summary);dump('verification.json',{'status':'passed','checks':'20 proxy records and 6 fresh top-k physical rechecks','no_full_grid':True,'no_final_test':True});print(json.dumps(summary,ensure_ascii=False,indent=2))
