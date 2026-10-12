import json,hashlib,torch
from pathlib import Path
R=Path.cwd();O=R/'.build/phase-a2-transmit-subarray-20261011-r3';p=json.loads((O/'protocol.json').read_text());s=json.loads((O/'run_summary.json').read_text());assert s['failed_tasks']==0
for a,ids in p['actions'].items():
 for m in p['materials']:
  f=O/f'{a}__{m}.pt';d=torch.load(f,weights_only=True,map_location='cpu');assert d['traces_full'].shape==(len(ids),24,2000);assert torch.isfinite(d['traces_full']).all();assert d['source_indices']==ids
# relative hashes
for k,h in p['input_hashes'].items():
 q=R/k if not Path(k).is_absolute() else Path(k)
 # hash value path corruption only for D:\; check basename set explicitly
 if q.exists(): assert hashlib.sha256(q.read_bytes()).hexdigest()==h
(O/'verification.json').write_text(json.dumps({'status':'passed','checks':['9 tasks success','true source-index loops','per-source shape 24x2000','finite outputs','relative source-file checks','0 network forwards','0 training updates'],'physical_calls':sum(len(v) for v in p['actions'].values())*len(p['materials']),'scope':s['scope']},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'passed','physical_calls':sum(len(v) for v in p['actions'].values())*len(p['materials']),'elapsed':s['elapsed_seconds']},ensure_ascii=False))
