import sys,torch,pandas as pd
sys.path.insert(0,'Research_Direction/Author_Baseline_Singh_Zenodo14616677'); from run_fno_local_contrast import physical_setup
solver,_,template,shape,rx,ry,device=physical_setup(); nx,ny,n=shape; root='Research_Direction/Author_Baseline_Singh_Zenodo14616677'; f=torch.tensor(pd.read_hdf(root+'/data_casestudy/source.h5').values).reshape(4,1,nx+3,ny+3,n).clone().to(device); orig=f.clone(); mask=torch.load('.build/fno-frozen-depth-shift-audit-20261009-r2/state_0.pt',weights_only=True)['gamma']<.99; mat=torch.where(mask,torch.full_like(mask,.5,dtype=template['gamma'].dtype),torch.ones_like(template['gamma'])); gamma=torch.ones(1,1,nx+3,ny+3,device=device); gamma[0,0,1:-1,1:-1]=mat
origg=gamma.clone(); outs=[]
for si in [0,1,2,3]:
 z=torch.zeros_like(gamma); w=solver.forwardNSteps(z,z.clone(),gamma,f[si:si+1],nx,ny,n,1,device); outs.append(w[:,rx,ry,1:].cpu()); print('after',si,'force',torch.max(torch.abs(f-orig)).item(),'gamma',torch.max(torch.abs(gamma-origg)).item())
for si in [2,3]:
 z=torch.zeros_like(gamma); w=solver.forwardNSteps(z,z.clone(),gamma,orig[si:si+1],nx,ny,n,1,device); print('src',si,'diff after sequence',torch.max(torch.abs(w[:,rx,ry,1:].cpu()-outs[si])).item())
