"""Audit persisted checkpoints after write failure; no further training."""
import json,torch,time,os
from pathlib import Path
from run_paired_depth_coverage import OUT,policy,data_setup,evaluation,BasicFNO,state_hash,dump,sha,ROOT,PLAN
policy();p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
assert sha(PLAN)==p['plan_sha256'] and sha(ROOT/'run_paired_depth_coverage.py')==p['script_sha256']
for f,h in {**p['frozen_sha256'],**p['dependency_sha256']}.items():assert sha(Path(f))==h
checks={}
for update in [150,300]:
 path=OUT/f'nominal_repeat_update{update}.pt'
 try:
  ck=torch.load(path,weights_only=True,map_location='cpu');checks[str(update)]={'loadable':True,'file_sha256':sha(path)}
 except Exception as e:checks[str(update)]={'loadable':False,'error':str(e),'file_sha256':sha(path)}
 assert update!=150 or checks[str(update)]['loadable']
ck=torch.load(OUT/'nominal_repeat_update150.pt',weights_only=True,map_location='cpu')
assert ck['update']==150 and ck['exposures']==[50]*3 and all(float(s['step'])==150 for s in ck['optimizer_state_dict']['state'].values())
device,nom,bg,scale,sr,samples=data_setup();model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);model.eval()
assert state_hash(model)==ck['state_hash'];ev=evaluation(model,samples,nom,bg,scale,device,False)
arm=torch.load(OUT/'nominal_repeat_progress.pt',weights_only=True,map_location='cpu');saved=arm['history'][-1]['evaluation']
assert arm['history'][-1]['update']==150 and len(arm['losses'])==150
assert ev['records']==saved['records'] and ev['field_sha256']==saved['field_sha256']
assert all(torch.equal(ev['traces'][k],saved['traces'][k]) for k in ev['traces'])
# Trial CPU serialization with readback in a fresh diagnostics directory.
diagnostic=OUT/'save_diagnostic';assert not diagnostic.exists();diagnostic.mkdir()
tmp=diagnostic/'checkpoint150.pt.tmp';final=diagnostic/'checkpoint150.pt'
torch.save(ck,tmp);again=torch.load(tmp,weights_only=True,map_location='cpu')
assert all(torch.equal(again['state_dict'][k],v) for k,v in ck['state_dict'].items())
assert again['update']==150 and again['exposures']==[50]*3
os.replace(tmp,final)
audit={'status':'execution_failed_incomplete','primary300_pair':'unknown, unavailable','coverage_arm':'not started',
 'stdout_observation':'nominal loop reached300 updates; failed while saving update300; final300 evaluation not persisted',
 'saved_checkpoints':checks,'verified150':'All5 fields hash and traces bit-identical reload; metrics exact; Adam150; slots50 each',
 'CPU_save_readback':'passed in new diagnostic directory; does not identify original write failure cause',
 'original_IO_cause':'unknown; current free-space check showed about11.45GB, so no evidence of disk full',
 'secondary150':{k:v for k,v in saved.items() if k not in ['traces','field_sha256']},
 'scientific_limits':'Only nominal repeated-training secondary result; no paired causal conclusion, no selected replacement primary',
 'verified_restart_checkpoint':str(final),'verified_restart_sha256':sha(final),'new_training_updates':0,'new_physical_calls':0}
assert not (OUT/'save_failure_audit.json').exists();dump(OUT/'save_failure_audit.json',audit)
dump(OUT/'failure.json',{'class':'RuntimeError','action':'torch.save nominal_repeat_update300.pt','message':'ios_base::badbit / unexpected pos171992448 vs171992336','cause':'unknown','coverage_started':False,'artifacts':'preserved'})
print(json.dumps(audit,indent=2))
