"""Compare dense and sparse/cubic objectives and material gradients on 8 states."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import FiniteDifferencePyTorchConv as FiniteDifference
import Utilities
from discrete_reference import DiscreteReference,time_cost,variable_mask
from time_reconstruction import cubic_time_reconstruction


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stride',type=int,default=4)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    cfg=pd.read_csv(root/'settings2D.csv').iloc[0]
    nx,ny,steps,shots=(int(cfg[k]) for k in ('Nx','Ny','N','numberOfSources'))
    dx,dy,dt=float(cfg['Lx']/nx),float(cfg['Ly']/ny),float(cfg['dt'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float64)
    solver=FiniteDifference.FiniteDifference(dt,dx,dy,float(cfg['c']),float(cfg['rho']),device).double()
    sx,sy=Utilities.getSourceLocations(nx,ny,int(cfg['distanceBetweenSources']),shots)
    rx,ry=Utilities.getSensorLocations(nx,ny,int(cfg['distanceBetweenSensors']),sx)
    reference=DiscreteReference(solver,rx,ry)
    force=torch.tensor(pd.read_hdf(root/'data_casestudy/source.h5').values).reshape(shots,1,nx+3,ny+3,steps).to(device=device,dtype=torch.float64)
    full_times=torch.arange(1,steps+1,device=device)*dt
    indices=torch.arange(args.stride-1,steps,args.stride,device=device)
    assert indices[-1]==steps-1
    results=[]
    for case in (1,2,3,4):
        observed=torch.tensor(pd.read_hdf(root/f'data_casestudy/measurement{case}.h5').values).reshape(shots,len(rx),steps).to(device=device,dtype=torch.float64)
        history=pd.read_hdf(f'.build/author-singh-case{case}-35epochs-20261008/GammaHistory/conventionalFWI_{case}.h5').values.reshape(36,nx+3,ny+3)
        for update in (0,5):
            gamma=torch.tensor(history[update],device=device,dtype=torch.float64)[None,None].requires_grad_(True)
            mask=variable_mask(gamma,rx,ry,sx,sy)
            prediction=reference.traces(gamma,force)
            restored=cubic_time_reconstruction(prediction[...,indices],full_times[indices],full_times)
            dense_loss=time_cost(prediction,observed,dx,dy,dt)
            sparse_loss=time_cost(restored,observed,dx,dy,dt)
            dense_grad=torch.autograd.grad(dense_loss*1e12,gamma,retain_graph=True)[0]/1e12*mask
            sparse_grad=torch.autograd.grad(sparse_loss*1e12,gamma)[0]/1e12*mask
            cosine=float((dense_grad*sparse_grad).sum()/dense_grad.norm()/sparse_grad.norm())
            row={'case':case,'update':update,'waveform_relative_error':float(((prediction-restored).norm()/prediction.norm()).detach()),'relative_cost_change':abs(float((sparse_loss/dense_loss).detach())-1),'gradient_relative_l2':float((dense_grad-sparse_grad).norm()/dense_grad.norm()),'gradient_angle_degrees':float(np.degrees(np.arccos(np.clip(cosine,-1,1))))}
            results.append(row);print(json.dumps(row),flush=True)
            del prediction,restored,dense_grad,sparse_grad,gamma
    args.out.parent.mkdir(parents=True,exist_ok=True)
    report={'stride':args.stride,'reconstruction':'Fixed local cubic Lagrange, known zero initial displacement','dtype':'float64','scope':'Four author cases at initial and author update 5; same scalar solver and actual full-time objective. No neural approximation included.','results':results,'screening_thresholds':{'waveform':.001,'relative_cost_change':.01,'gradient_relative_l2':.02},'screening_passed':all(r['waveform_relative_error']<.001 and r['relative_cost_change']<.01 and r['gradient_relative_l2']<.02 for r in results)}
    args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Sparse-time screening passed:',report['screening_passed'],flush=True)


if __name__=='__main__':main()
