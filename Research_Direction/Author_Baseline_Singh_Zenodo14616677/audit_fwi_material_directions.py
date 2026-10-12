"""Finite material-direction responses and feasible trial-step checks for case 1."""
from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd, torch
import FiniteDifferencePyTorchConv as FiniteDifference
import Utilities
from discrete_reference import DiscreteReference, time_cost, variable_mask

def main():
    torch.set_num_threads(4); root=Path(__file__).resolve().parent; cfg=pd.read_csv(root/'settings2D.csv').iloc[0]
    nx,ny,steps,shots=(int(cfg[k]) for k in ('Nx','Ny','N','numberOfSources')); dx,dy=float(cfg['Lx']/nx),float(cfg['Ly']/ny); dt=float(cfg['dt']); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    solver=FiniteDifference.FiniteDifference(dt,dx,dy,float(cfg.c),float(cfg.rho),device); sx,sy=Utilities.getSourceLocations(nx,ny,int(cfg.distanceBetweenSources),shots); rx,ry=Utilities.getSensorLocations(nx,ny,int(cfg.distanceBetweenSensors),sx); ref=DiscreteReference(solver,rx,ry)
    force=torch.tensor(pd.read_hdf(root/'data_casestudy/source.h5').values).reshape(shots,1,nx+3,ny+3,steps).to(device); observed=torch.tensor(pd.read_hdf(root/'data_casestudy/measurement1.h5').values).reshape(shots,len(rx),steps).to(device)
    truth=torch.ones((1,1,nx+3,ny+3),device=device); truth[0,0,1:-1,1:-1]=torch.tensor(pd.read_hdf(root/'data_casestudy/material1.h5').values,device=device); mask=variable_mask(truth,rx,ry,sx,sy); base=float(time_cost(ref.traces(truth,force,False),observed,dx,dy,dt))
    xx,yy=torch.meshgrid(torch.linspace(0,1,nx+3,device=device),torch.linspace(0,1,ny+3,device=device),indexing='ij'); dirs={
        'uniform':torch.ones_like(truth),
        'smooth_x':torch.cos(torch.pi*xx)[None,None],
        'smooth_y':torch.cos(torch.pi*yy)[None,None],
        'local_center':torch.exp(-((xx-.5)**2+(yy-.5)**2)/.03)[None,None]}
    records={}
    for name,d in dirs.items():
        d=d*mask; d=d/(d.square().sum().sqrt()+1e-30); vals=[]
        for eps in [.005,.02]:
            plus=torch.clamp(truth+eps*d,float(cfg.gamma0),1); minus=torch.clamp(truth-eps*d,float(cfg.gamma0),1)
            cp=float(time_cost(ref.traces(plus,force,False),observed,dx,dy,dt)); cm=float(time_cost(ref.traces(minus,force,False),observed,dx,dy,dt)); vals.append({'epsilon':eps,'plus_cost_over_truth':cp/base,'minus_cost_over_truth':cm/base,'central_difference_over_eps':(cp-cm)/(2*eps),'plus_material_half_mse':float(.5*(plus-truth).square().mean())})
        records[name]={'direction_norm':float(d.norm()),'steps':vals}
    out=Path('.build/author-fwi-material-directions-20261008');out.mkdir(parents=True,exist_ok=True); report={'status':'completed','case':1,'scope':'Truth-state finite material-direction response and feasible trial-step checks; no test data or optimizer update.','base_cost':base,'directions':records,'data_source_sha256':hashlib.sha256((root/'data_casestudy/measurement1.h5').read_bytes()).hexdigest(),'settings_sha256':hashlib.sha256((root/'settings2D.csv').read_bytes()).hexdigest()};(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__': main()
