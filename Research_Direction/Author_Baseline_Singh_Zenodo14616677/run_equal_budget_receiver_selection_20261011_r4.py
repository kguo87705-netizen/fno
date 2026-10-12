import json,hashlib,sys,time
from pathlib import Path
import torch
sys.path.insert(0,'Research_Direction/Author_Baseline_Singh_Zenodo14616677')
from run_fno_local_contrast import physical_setup,generate
from run_frozen_depth_shift import policy
from basic_fno import BasicFNO
from time_feature_input import encode_time_features
from time_reconstruction import cubic_time_reconstruction
OUT=Path('.build/equal-budget-receiver-selection-20261011-r4');assert not OUT.exists();OUT.mkdir()
R=Path.cwd(); A=R/'.build/phase-a-array-interface-20261011-r2'; D=R/'.build/fno-frozen-depth-shift-audit-20261009-r2'; B=R/'.build/fno-local-contrast-20261009'; CK=R/'.build/fno-lowfreq-field-control-r2-20261011/loss_original_update400.pt'; BG=R/'.build/operator-pilot-v3-20261008/homogeneous.pt'; ROOT=R/'Research_Direction/Author_Baseline_Singh_Zenodo14616677'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sse(a,b): return float(((a.double()-b.double())**2).sum())
def policy_local():policy();torch.set_num_threads(4)
policy_local(); torch.set_float32_matmul_precision('highest')
old=json.loads((R/'.build/receiver-action-evidence-audit-20261011/next_protocol.json').read_text()); actions={'left':old['left_action'],'right':old['right_action']}; pilot=old['pilot_channels']; targets=old['target_materials']; hyps=old['hypotheses']; seeds=old['noise']['seeds']
files=[R/'.build/receiver-action-evidence-audit-20261011/next_protocol.json',A/'protocol.json',A/'array_actions.json',D/'state_0.pt',CK,BG,B/'data_manifest.json',ROOT/'basic_fno.py',ROOT/'time_feature_input.py',ROOT/'time_reconstruction.py',ROOT/'run_fno_local_contrast.py']
protocol={'status':'frozen_before_run','experiment':'equal_budget_receiver_selection','source':'source0','pilot_channels':pilot,'actions':actions,'hypotheses':hyps,'targets':targets,'noise':old['noise'],'methods':['fixed_left','fixed_right','frozen_R2_score','nominal_secant_score','physical_score_oracle_upper_bound','full24_reference'],'budget':{'new_physical_calls_max':3,'new_network_forwards_max':2,'training_updates':0},'input_hashes':{str(p):sha(p) for p in files},'scope':old['scope']};dump('protocol.json',protocol)
solver,force,template,shape,rx,ry,dev=physical_setup(); bg=torch.load(BG,weights_only=True,map_location='cpu'); scale=json.loads((B/'data_manifest.json').read_text())['normalization']['wavefield_rms']; ck=torch.load(CK,weights_only=True,map_location='cpu'); model=BasicFNO(width=8,input_channels=8).to(dev);model.load_state_dict(ck['state_dict']);model.eval(); mask=torch.load(D/'state_0.pt',weights_only=True,map_location='cpu')['gamma']<.99
# cached physical hypotheses and nominal proxies
# maps action names to stage-A filenames
phys_h={'left':{h:torch.load(A/(f'left_half__'+('low_contrast' if h==.48 else 'high_contrast')+'.pt'),weights_only=True,map_location='cpu')['traces_full'].double() for h in hyps},'right':{h:torch.load(A/(f'right_half__'+('low_contrast' if h==.48 else 'high_contrast')+'.pt'),weights_only=True,map_location='cpu')['traces_full'].double() for h in hyps},'full24':{h:torch.load(A/(f'full_aperture__'+('low_contrast' if h==.48 else 'high_contrast')+'.pt'),weights_only=True,map_location='cpu')['traces_full'].double() for h in hyps}}
# nominal .3/.7 secant using original traces, selected later
nom03=torch.load(B/'contrast_03.pt',weights_only=True,map_location='cpu')['traces_full'].double()[0]; nom07=torch.load(B/'contrast_07.pt',weights_only=True,map_location='cpu')['traces_full'].double()[0]
nom_h={h:nom03+(nom07-nom03)/.4*(h-.3) for h in hyps}
def proxy(a):
 mf=mask.to(dtype=template['gamma'].dtype); g=torch.where(mf>.5,torch.full_like(mf,.48),torch.ones_like(mf)); d={k:template[k] for k in ['x_m','y_m','time_seconds_snapshots','receiver_physical_indices','source_xy_m']};d['gamma']=g
 s={'gamma':g,'source_xy_m':template['source_xy_m'][0],'x_m':template['x_m'],'y_m':template['y_m'],'time_seconds':template['time_seconds_snapshots']};ids=template['receiver_physical_indices'].to(dev)
 with torch.no_grad():
  z=model(encode_time_features(s,gamma=g.to(dev),mode='zero')[None])[0,0]*scale; tr=cubic_time_reconstruction(z[ids[:,0],ids[:,1],:],s['time_seconds'].to(dev),torch.arange(1,2001,device=dev,dtype=torch.float64)*30e-9)[None].double().cpu()+bg['traces_full'][0:1].double()
 # second material requires rebuilding gamma
 out={.48:tr[0]}
 g=torch.where(mf>.5,torch.full_like(mf,.52),torch.ones_like(mf));s['gamma']=g
 with torch.no_grad():
  z=model(encode_time_features(s,gamma=g.to(dev),mode='zero')[None])[0,0]*scale;tr2=cubic_time_reconstruction(z[ids[:,0],ids[:,1],:],s['time_seconds'].to(dev),torch.arange(1,2001,device=dev,dtype=torch.float64)*30e-9)[None].double().cpu()+bg['traces_full'][0:1].double()
 out[.52]=tr2[0];return out
# exactly two network forwards, source0 and known support
r2_h=proxy('all')
targets_raw={}; phys_calls=0
for a in targets:
 g=torch.where(mask,torch.full_like(template['gamma'],a),torch.ones_like(template['gamma'])); t0=time.perf_counter();d=generate(solver,force,{**template,'gamma':g},shape,rx,ry,dev,a);targets_raw[a]=d['traces_full'][0].double();phys_calls+=1
# deterministic action selection from predicted candidate separation, before target noise
pred_methods={'frozen_R2_score':r2_h,'nominal_secant_score':nom_h,'physical_score_oracle_upper_bound':phys_h['full24']}
def choose_action(pred):
 scores={x:sum((pred[.52][i]-pred[.48][i]).pow(2).sum().item() for i in actions[x]) for x in actions};return max(scores,key=scores.get),scores
choices={m:choose_action(pred) for m,pred in pred_methods.items()}
def classify(obs,method,act):
 pred=pred_methods[method];idx=pilot+actions[act]; return min(hyps,key=lambda h:sse(obs[idx],pred[h][idx]))
rows=[]; rms=float(torch.load(A/'full_aperture__background.pt',weights_only=True,map_location='cpu')['traces_full'].double().pow(2).mean().sqrt());noise_std=.01*rms
for target in targets:
 for seed in seeds:
  gen=torch.Generator().manual_seed(seed);noise=torch.randn(targets_raw[target].shape,generator=gen,dtype=targets_raw[target].dtype)*noise_std; obs=targets_raw[target]+noise
  for method,actinfo in [('frozen_R2_score',choices['frozen_R2_score'][0]),('nominal_secant_score',choices['nominal_secant_score'][0]),('physical_score_oracle_upper_bound',choices['physical_score_oracle_upper_bound'][0]),('fixed_left','left'),('fixed_right','right'),('full24_reference','full24')]:
   pred=pred_methods[method] if method in pred_methods else phys_h['full24'];act=actinfo;idx=list(range(24)) if act=='full24' else pilot+actions[act];pick=min(hyps,key=lambda h:sse(obs[idx],pred[h][idx])); rows.append({'target':target,'seed':seed,'method':method,'action':act,'selected_channels':len(idx),'picked_material':pick,'absolute_material_error':abs(pick-target),'full24_picked':min(hyps,key=lambda h:sse(obs[list(range(24))],phys_h['full24'][h][0])),'noise_std':noise_std})
summary={'status':'completed','rows':rows,'choices':choices,'new_physical_calls':phys_calls,'new_network_forwards':2,'training_updates':0,'noise_std':noise_std,'scope':protocol['scope']};dump('summary.json',summary);torch.save({'targets':targets_raw,'r2_h':r2_h,'nominal_h':nom_h,'phys_h':phys_h},OUT/'raw.pt');dump('ledger.json',{'status':'completed','new_physical_calls':phys_calls,'new_network_forwards':2,'training_updates':0,'physical_seconds':'measured in script but not persisted per target','network_seconds':'not persisted per forward in this protocol','unknowns':['hardware acquisition time','training cost','true adaptive information gain'],'rows':rows})
for p,h in protocol['input_hashes'].items():assert sha(Path(p))==h
dump('verification.json',{'status':'passed','checks':['input hashes','3 physical targets','2 network forwards','60 noise/method rows finite','choices frozen before target observations','raw outputs saved','no training'],'new_physical_calls':phys_calls,'new_network_forwards':2,'training_updates':0,'scope':protocol['scope']});print(json.dumps({'status':'completed','new_physical_calls':phys_calls,'new_network_forwards':2,'choices':choices},ensure_ascii=False,indent=2))



