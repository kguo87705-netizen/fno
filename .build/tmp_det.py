import sys,torch,pandas as pd
sys.path.insert(0,'Research_Direction/Author_Baseline_Singh_Zenodo14616677')
from run_fno_local_contrast import physical_setup
solver,_,template,shape,rx,ry,device=physical_setup(); nx,ny,n=shape
root='Research_Direction/Author_Baseline_Singh_Zenodo14616677'; force_all=torch.tensor(pd.read_hdf(root+'/data_casestudy/source.h5').values).reshape(4,1,nx+3,ny+3,n).clone().to(device)
mask=torch.load('.build/fno-frozen-depth-shift-audit-20261009-r2/state_0.pt',weights_only=True)['gamma']<.99
a=.5; mat=torch.where(mask,torch.full_like(mask,a,dtype=template['gamma'].dtype),torch.ones_like(template['gamma'])); gamma=torch.ones(1,1,nx+3,ny+3,device=device); gamma[0,0,1:-1,1:-1]=mat
zs=[]
for rep in range(3):
 z=torch.zeros_like(gamma); w=solver.forwardNSteps(z,z.clone(),gamma,force_all[2:3],nx,ny,n,1,device); zs.append(w[:,rx,ry,1:].cpu())
for i in range(2): print(i,torch.max(torch.abs(zs[i]-zs[0])).item())
