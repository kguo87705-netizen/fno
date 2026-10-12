"""Saved prediction decomposition; no forward or training calls."""
from pathlib import Path
import json,torch
from run_contrast_derivative_audit import metrics,dump,BASE
from run_frozen_depth_shift import OUT,policy
from time_reconstruction import cubic_time_reconstruction
from run_reduced_material_scan import sha
policy();raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu')
d=torch.load(BASE/'contrast_05.pt',weights_only=True,map_location='cpu');ids=d['receiver_physical_indices'];states=raw['states']
p=states['2']['prediction']['field']-states['-2']['prediction']['field']
f=states['2']['physical']['field'].double()-states['-2']['physical']['field'].double()
ps=p[ids[:,0],ids[:,1],:];fs=f[ids[:,0],ids[:,1],:]
true=states['2']['physical']['trace'].double()-states['-2']['physical']['trace'].double()
device='cuda';times=torch.arange(1,2001,device=device,dtype=torch.float64)*30e-9
# Apply the existing reconstruction to the physical snapshot difference itself.
reconstructed=cubic_time_reconstruction(fs.float().to(device),d['time_seconds_snapshots'].to(device),times)[None].double().cpu()
r={'status':'completed','new_forward_calls':0,'new_training_calls':0,
 'full_field_finite_geometry_response':metrics(p,f),
 'receiver_snapshot_finite_geometry_response':metrics(ps,fs),
 'physical_snapshot_reconstruction_vs_full_time_physical_response':metrics(reconstructed,true),
 'interpretation':'Evaluate response before temporal reconstruction and reconstruction on physical snapshots. Background cancels between paired states; a geometry-independent additive correction cannot alter this response.',
 'raw_sha256':sha(OUT/'raw.pt'),'script_sha256':sha(Path(__file__))}
assert not (OUT/'response_decomposition.json').exists();dump(OUT/'response_decomposition.json',r)
print(json.dumps(r,indent=2))
