"""Verify all saved checkpoints, including a budget-limited partial run."""
import json,torch
from run_paired_depth_coverage import OUT,PLAN,policy,data_setup,evaluation,report,ROOT,BasicFNO,state_hash,dump,sha
policy();p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
assert sha(PLAN)==p['plan_sha256']
assert sha(ROOT/'run_paired_depth_coverage.py')==p['script_sha256']
for f,h in {**p['frozen_sha256'],**p['dependency_sha256']}.items():assert sha(__import__('pathlib').Path(f))==h
device,nom,bg,scale,sr,samples=data_setup();arms={};checks=[]
for name in p['plan']['arms']:
 f=OUT/f'{name}_results.pt'
 arm=torch.load(f if f.exists() else OUT/f'{name}_progress.pt',weights_only=True,map_location='cpu');arms[name]=arm
 for h in arm['history']:
  update=h['update']
  if update==0:continue
  path=OUT/f'{name}_update{update}.pt'
  ck=torch.load(path,weights_only=True,map_location='cpu');model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);model.eval()
  assert state_hash(model)==ck['state_hash']
  assert ck['update']==update and ck['exposures']==[update//3]*3
  assert all(float(s['step'])==update for s in ck['optimizer_state_dict']['state'].values())
  ev=evaluation(model,samples,nom,bg,scale,device,update==300);saved=h['evaluation']
  assert ev['field_sha256']==saved['field_sha256'] and ev['records']==saved['records'] and ev['response']==saved['response']
  assert all(torch.equal(ev['traces'][k],saved['traces'][k]) for k in ev['traces'])
  checks.append({'arm':name,'update':update,'checkpoint_sha256':sha(path),'field_and_trace':'bit-identical reload','metrics':'exact'})
  del model,ck;torch.cuda.empty_cache()
 for i,l in enumerate(arm['losses']):assert l['update']==i+1 and l['slot']==i%3 and l['depth']==p['plan']['arms'][name][i%3]
 assert len(arm['losses'])==arm['history'][-1]['update']
a,b=arms.values();assert a['initial_state_hash']==b['initial_state_hash']
ea,eb=a['history'][0]['evaluation'],b['history'][0]['evaluation']
assert ea['field_sha256']==eb['field_sha256'] and all(torch.equal(ea['traces'][k],eb['traces'][k]) for k in ea['traces'])
common=sorted(set(h['update'] for h in a['history'])&set(h['update'] for h in b['history']))
def small(ev):return {k:v for k,v in ev.items() if k not in ['traces','field_sha256']}
paired={str(i):{n:small(next(h['evaluation'] for h in arm['history'] if h['update']==i)) for n,arm in arms.items()} for i in common}
complete=300 in common
summary={'status':'complete' if complete else 'budget_limited_incomplete','primary_update':300,'primary_result_available':complete,
 'latest_saved_updates':{n:arm['history'][-1]['update'] for n,arm in arms.items()},'paired_evaluations':paired,
 'saved_loss_records':{n:len(arm['losses']) for n,arm in arms.items()},
 'unsaved_updates':'unknown if budget stopped between checkpoint writes; do not report saved count as all executed updates',
 'failure':json.loads((OUT/'failure.json').read_text()) if (OUT/'failure.json').exists() else None,
 'limits':'150 is a predeclared secondary checkpoint, cannot replace the primary300 or be selected as best checkpoint.'}
assert not (OUT/'checkpoint_audit.json').exists();dump(OUT/'checkpoint_audit.json',summary)
dump(OUT/'checkpoint_verification.json',{'status':'passed','checks':checks,'initial':'same parameter/field hashes and trace bits','source_hashes':'unchanged','primary_result_available':complete})
print(json.dumps(summary,indent=2))
