"""Frozen constant-input model: alternating versus mean-target parameter gradients."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from basic_fno import BasicFNO
from operator_data import OperatorPilotDataset
from time_feature_input import encode_time_features
from run_receiver_loss_control import supervised_losses


def inner(left,right):
    return sum(float((a.conj()*b).real.double().sum()) for a,b in zip(left,right))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu';ds=OperatorPilotDataset(a.data,'train');samples=[ds[0],ds[4]]
    c=torch.load(a.run/'final_arm1.pt',weights_only=True,map_location=device);assert c['arm']=='gamma_replaced_by_ones' and c['updates']==800
    model=BasicFNO(input_channels=8).to(device);model.load_state_dict(c['state_dict']);model.eval()
    bg=torch.load(a.data/'homogeneous.pt',weights_only=True,map_location='cpu')['wavefield_snapshots'][0].to(device)/ds.scale
    targets=[s['target'][0].to(device)-bg for s in samples];targets.append((targets[0]+targets[1])/2)
    x=encode_time_features(samples[0],torch.ones_like(samples[0]['gamma']),mode='zero')[None].to(device);ids=samples[0]['receiver_indices'].to(device)
    gradients=[];losses=[]
    for target in targets:
        pred=model(x);lf,lr=supervised_losses(pred,target[None,None],ids,ds.scale,ds.manifest['normalization']['trace_rms']);loss=lf+lr
        gradient=torch.autograd.grad(loss,tuple(model.parameters()));assert all(torch.isfinite(g).all() for g in gradient)
        gradients.append([g.detach() for g in gradient]);losses.append(float(loss.detach()));del pred,loss,lf,lr
    ga,gb,gm=gradients;average=[(a+b)/2 for a,b in zip(ga,gb)];noise=[(a-b)/2 for a,b in zip(ga,gb)];difference=[a-b for a,b in zip(average,gm)]
    norm=lambda g:np.sqrt(max(inner(g,g),0.));na,nb,nm,nn=map(norm,[ga,gb,gm,noise])
    numerical_error=norm(difference)/max(nm,1e-30);assert numerical_error<.002
    result={'scope':'Frozen final constant-input model, parameter gradients of supervised training loss only; not material/physical gradients or a proof of optimization failure.',
        'target_losses':dict(zip(['workpiece000','workpiece001','mean_target'],losses)),
        'parameter_gradient_norms':{'workpiece000':na,'workpiece001':nb,'mean_target':nm,'half_difference':nn},
        'alternating_half_difference_over_mean_gradient_norm':nn/max(nm,1e-30),
        'workpiece_gradient_cosine':inner(ga,gb)/max(na*nb,1e-30),
        'average_of_pair_gradients_vs_mean_target_relative_error':numerical_error,
        'interpretation':'The mean target has exactly the averaged gradient for these identical inputs. Replacing alternating labels by their mean preserves the average objective up to a constant, while removing this deterministic label alternation. An actual matched training run is needed to assess its effect.'}
    (a.run/'alternating_gradient_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
