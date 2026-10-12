import sys,torch,pandas as pd
sys.path.insert(0,'Research_Direction/Author_Baseline_Singh_Zenodo14616677'); import FiniteDifferencePyTorchConv as FD
from run_fno_local_contrast import physical_setup
_,_,template,shape,rx,ry,device=physical_setup(); nx,ny,n=shape; cfg=pd.read_csv('Research_Direction/Author_Baseline_Singh_Zenodo14616677/settings2D.csv').iloc[0]; f=torch.tensor(pd.read_hdf('Research_Direction/Author_Baseline_Singh_Zenodo14616677/data_casestudy/source.h5').values).reshape(4,1,nx+3,ny+3,n).clone().to(device); mask=torch.load('.build/fno-frozen-depth-shift-audit-20261009-r2/state_0.pt',weights_only=True)['gamma']<.99; mat=torch.where(mask,torch.full_like(mask,.5,dtype=template['gamma'].dtype),torch.ones_like(template['gamma'])); gamma=torch.ones(1,1,nx+3,ny+3,device=device); gamma[0,0,1:-1,1:-1]=mat.to(device)
for k in range(3):
 s=FD.FiniteDifference(float(cfg['dt']),float(cfg['Lx']/nx),float(cfg['Ly']/ny),float(cfg['c']),float(cfg['rho']),device); z=torch.zeros_like(gamma); w=s.forwardNSteps(z,z.clone(),gamma,f[1:2],nx,ny,n,1,device); print(k,torch.sqrt(torch.mean(w[:,rx,ry,1:]**2)).item())
