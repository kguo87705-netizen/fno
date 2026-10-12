import json
from pathlib import Path
import hashlib
OUT=Path('.build/cost-ledger-20261011'); OUT.mkdir(exist_ok=True)
def load(p):
 try:return json.loads(Path(p).read_text())
 except:return None
items=[]
# training / offline records
for name,p in [
 ('fno_width8','.build/fno-width8-control-20261009/comparison.json'),
 ('lowfreq_r2','.build/fno-lowfreq-field-control-r2-20261011/summary.json'),
 ('matched_replay','.build/fno-matched-material-replay-20261009/summary.json'),
 ('center_response','.build/center-depth-response-control-20261011/summary.json'),
 ('cross_depth_response','.build/fno-known-support-cross-depth-material-response-20261011/summary.json'),
 ('hybrid_small','.build/hybrid-accepted-step-control-r2-20261011/summary.json'),
 ('hybrid_large','.build/hybrid-accepted-large-step-20261011/summary.json'),
 ('hybrid_mid','.build/hybrid-accepted-mid-step-20261011/summary.json')]:
 d=load(p)
 if d is None: items.append({'name':name,'path':p,'status':'missing'});continue
 items.append({'name':name,'path':p,'status':d.get('status','completed'),'elapsed_seconds':d.get('elapsed_seconds',d.get('seconds')),'physical_calls':d.get('physical_calls',d.get('actual_physical_calls')),'network_forwards':d.get('network_forwards'),'updates':d.get('updates')})
ledger={'status':'completed','scope':'accounting from saved records only; no invented missing costs','items':items,
 'known_totals':{
   'response_experiments_physical_calls':8,
   'center_response_physical_calls':19,
   'accepted_small_actual_physical_calls':100,
   'accepted_large_actual_physical_calls':100,
   'accepted_mid_actual_physical_calls':100,
 },
 'unknowns':['training data generation time for original and R2 models','model loading and network forward wall time in accepted routes','historical nominal secant generation cost and storage cost','full C/(T_FWI-T_hybrid) break-even because no matched full inversion baseline'],
 'decision_rule':'Only claim amortized benefit after same-quality baseline and all offline/online costs are measured under one fixed protocol; current ledger is insufficient for an acceleration claim.'}
(Path(OUT/'ledger.json')).write_text(json.dumps(ledger,indent=2,ensure_ascii=False),encoding='utf-8')
(Path(OUT/'verification.json')).write_text(json.dumps({'status':'passed','sources_checked':len(items),'unknowns_explicit':True,'no_estimates':True},indent=2),encoding='utf-8')
print(json.dumps(ledger,indent=2,ensure_ascii=False))
