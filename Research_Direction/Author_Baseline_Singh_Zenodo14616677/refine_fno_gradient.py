"""Double-precision FD audit of the frozen model's weak float32 AD direction."""
import argparse
import json
from pathlib import Path
import torch
from basic_fno import BasicFNO
from operator_data import OperatorPilotDataset,encode_fno_input
from time_reconstruction import cubic_time_reconstruction
from discrete_reference import time_cost

parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint=torch.load(args.run/'best.pt',weights_only=True,map_location=device)
model=BasicFNO(width=checkpoint['config']['width'],modes=tuple(checkpoint['config']['modes']),layers=checkpoint['config']['layers']).to(device)
model.load_state_dict(checkpoint['state_dict']);model.double()
# Module.double() handles real parameters; preserve both parts of complex weights.
for layer in model.spectral:
 for p in layer.weights:p.data=p.data.to(torch.complex128)
for p in model.parameters():p.requires_grad_(False)
model.eval()
dataset=OperatorPilotDataset(args.data,'test');sample=dataset[0]
grad=torch.load(args.run/'gradient_fields.pt',weights_only=True)['neural_gradient']
direction=(grad.double()/grad.double().norm()).to(device)
projection=float((grad.double()*direction.cpu()).sum())
observed=sample['traces_full'].double().unsqueeze(0).to(device)
times=torch.arange(1,2001,dtype=torch.float64,device=device)*dataset.manifest['dt_seconds']
indices=sample['receiver_indices'].to(device)
dx=float(sample['x_m'][-1])/255;dy=float(sample['y_m'][-1])/127;dt=dataset.manifest['dt_seconds']
def objective(gamma):
 field=model(encode_fno_input(sample,gamma).unsqueeze(0))[0,0]
 sparse=field[indices[:,0],indices[:,1],:]*dataset.scale
 full=cubic_time_reconstruction(sparse,sample['time_seconds'].to(device),times)
 return float(time_cost(full.unsqueeze(0),observed,dx,dy,dt))
gamma=torch.ones_like(direction);checks=[]
with torch.no_grad():
 initial=objective(gamma)
 for epsilon in (.1,.05,.02):
  plus,minus=objective(gamma+epsilon*direction),objective(gamma-epsilon*direction)
  fd=(plus-minus)/(2*epsilon)
  checks.append({'epsilon':epsilon,'float64_finite_difference':fd,'float32_ad_projection':projection,'relative_difference':abs(fd/projection-1)})
report={'scope':'Same frozen checkpoint and first test shot as evaluation; real weights promoted to float64 and complex weights to complex128 without changing values. FD compared with saved float32 AD direction.','initial_float64_cost':initial,'checks':checks,'passes_2_percent':all(r['relative_difference']<.02 for r in checks),'model_weights_changed':False}
(args.run/'gradient_precision_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
