"""Additional read-only metrics; do not overwrite original result files."""
import json,hashlib
from pathlib import Path
import torch
O=Path('.build/hybrid-accepted-step-control-r2-20261011')
raw=torch.load(O/'raw.pt',weights_only=True,map_location='cpu'); bg=torch.load('.build/operator-pilot-v3-20261008/homogeneous.pt',weights_only=True,map_location='cpu')['traces_full'][0:1].double()
rows=[]
for r in raw['routes']:
 obs=raw['observations'][r['case']]; pred=r['initial'];n=0
 for h in r['history']:
  if h['accepted']:pred=h['trial']
  else:n+=1
 e=(pred-obs).norm();rows.append({'case':r['case'],'method':r['method'],'full_relative_l2':float(e/obs.norm()),'scatter_relative_l2':float(e/(obs-bg).norm()),'parameter_error':r['parameter_error'],'rejections':n,'calls':r['online_physical_calls'],'seconds':r['seconds']})
assert sum(r['calls'] for r in rows)+4==len(raw['calls'])==100
out={'rows':rows,'actual_calls':100,'data_generation_calls':4,'online_calls':96,'timing_note':'sequential single run, cached FNO direction and loading excluded; no total-cost speedup claim','scope':'nearby scalar development diagnostic; clipping .02 matches target offset and can yield exact first-step answer','historical_corrections':['Previous 4-vs-10 counted target generation as per-route initial call while reusing initial trace; not online measured counts.','Previous update loop unconditionally accepted candidates.','Previous verification checked shapes/hashes, not numerical replay.','Material response relative-L2 screen remains failed; successful scalar contraction does not waive it.']}
(O/'independent_metrics.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
