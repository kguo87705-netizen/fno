"""Wide finite material responses from existing nominal-geometry predictions."""
import json,torch
from run_depth_coverage_storage_retry import OUT
from run_paired_depth_coverage import BASE
from run_contrast_derivative_audit import metrics,dump
from run_reduced_material_scan import sha
torch.set_num_threads(4)
raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu')
physical={}
for a,k in [(3,'contrast03'),(5,'depth0'),(7,'contrast07')]:physical[k]=torch.load(BASE/f'contrast_{a:02d}.pt',weights_only=True,map_location='cpu')['traces_full'].double()
providers={'original':raw['arms']['nominal_repeat']['history'][0]['evaluation']['traces'],**{n:v['history'][-1]['evaluation']['traces'] for n,v in raw['arms'].items()}}
rows=[]
for name,traces in providers.items():
 for lo,hi in [('contrast03','depth0'),('depth0','contrast07'),('contrast03','contrast07')]:
  rows.append({'method':name,'interval':[lo,hi],**metrics(traces[hi]-traces[lo],physical[hi]-physical[lo])})
result={'status':'completed','new_forward_calls':0,'new_training_updates':0,'rows':rows,'raw_sha256':sha(OUT/'raw.pt'),
 'limits':'Nominal geometry, wide finite differences across saved .3/.5/.7; not a local Jacobian or unseen-depth material derivative. Development diagnostics, not final test.'}
assert not (OUT/'cached_nominal_material_response.json').exists();dump(OUT/'cached_nominal_material_response.json',result)
print(json.dumps(result,indent=2))
