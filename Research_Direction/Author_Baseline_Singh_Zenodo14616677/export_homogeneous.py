"""Fixed-material no-defect prediction for honest operator comparison."""
import argparse
from pathlib import Path
import pandas as pd
import torch
import FiniteDifferencePyTorchConv as FiniteDifference
import Utilities

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
root=Path(__file__).resolve().parent;cfg=pd.read_csv(root/'settings2D.csv').iloc[0]
nx,ny,steps,shots=(int(cfg[k]) for k in ('Nx','Ny','N','numberOfSources'))
dx,dy,dt=float(cfg['Lx']/nx),float(cfg['Ly']/ny),float(cfg['dt'])
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sx,sy=Utilities.getSourceLocations(nx,ny,int(cfg['distanceBetweenSources']),shots)
rx,ry=Utilities.getSensorLocations(nx,ny,int(cfg['distanceBetweenSensors']),sx)
force=torch.tensor(pd.read_hdf(root/'data_casestudy/source.h5').values).reshape(shots,1,nx+3,ny+3,steps).to(device)
solver=FiniteDifference.FiniteDifference(dt,dx,dy,float(cfg['c']),float(cfg['rho']),device)
zeros=torch.zeros((shots,1,nx+3,ny+3),device=device)
with torch.no_grad():
 wave=solver.forwardNSteps(zeros.clone(),zeros.clone(),torch.ones((1,1,nx+3,ny+3),device=device),force,nx,ny,steps,shots,device)
 traces=wave[:,rx,ry,1:].cpu();snapshots=wave[:,1:-1,1:-1,4::4].cpu()
 assert snapshots.shape[-1]==500
 args.out.parent.mkdir(parents=True,exist_ok=True);torch.save({'traces_full':traces,'wavefield_snapshots':snapshots},args.out)
print('Homogeneous reference exported')
