import sys,torch,pandas as pd
sys.path.insert(0,'Research_Direction/Author_Baseline_Singh_Zenodo14616677'); from run_fno_local_contrast import physical_setup
solver,_,template,shape,rx,ry,device=physical_setup(); nx,ny,n=shape; root='Research_Direction/Author_Baseline_Singh_Zenodo14616677'; f=torch.tensor(pd.read_hdf(root+'/data_casestudy/source.h5').values).reshape(4,1,nx+3,ny+3,n).clone().to(device); mask=torch.load('.build/fno-frozen-depth-shift-audit-20261009-r2/state_0.pt',weights_only=True)['gamma']<.99; mat=torch.where(mask,torch.full_like(mask,.5,dtype=template['gamma'].dtype),torch.ones_like(template['gamma'])); gamma=torch.ones(1,1,nx+3,ny+3,device=device); gamma[0,0,1:-1,1:-1]=mat
outs=[]
for si in [2,3]:
 z=torch.zeros_like(gamma); w=solver.forwardNSteps(z,z.clone(),gamma,f[si:si+1],nx,ny,n,1,device); outs.append(w[:,rx,ry,1:].cpu())
for i,si in enumerate([2,3]):
 old=torch.load(f'.build/tx-action-material-scan-20261011/full_tx__gamma_0.50.pt',weights_only=False)['traces_full'][si]
 print('full',si,torch.max(torch.abs(outs[i][0]-old)).item())
 old2=torch.load(f'.build/tx-action-material-scan-20261011/right_tx__gamma_0.50.pt',weights_only=False)['traces_full'][i]
 print('right',si,torch.max(torch.abs(outs[i][0]-old2)).item())
