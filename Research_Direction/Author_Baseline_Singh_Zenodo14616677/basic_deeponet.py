"""Ordinary CNN-branch / coordinate-MLP-trunk DeepONet, signed scalar output.
No Fourier features, PDE loss, homogeneous subtraction constraint or pretrained model.
"""
import torch
from torch import nn

class BasicDeepONet(nn.Module):
    def __init__(self,grid=(256,128),latent=64,hidden=128):
        super().__init__();nx,ny=grid
        if nx%8 or ny%8:raise ValueError('Fixed branch grid must be divisible by eight')
        self.config={'grid':list(grid),'latent':latent,'hidden':hidden,'branch_channels':[16,32,64],'trunk_input_channels':5,'activation':'GELU','output':'signed scalar normalized residual','coordinate_order':['x/Lx','y/Ly','t/T','sx/Lx','sy/Ly']}
        self.branch=nn.Sequential(nn.Conv2d(1,16,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(32,64,3,stride=2,padding=1),nn.GELU(),nn.Flatten(),nn.Linear(64*(nx//8)*(ny//8),hidden),nn.GELU(),nn.Linear(hidden,latent))
        self.trunk=nn.Sequential(nn.Linear(5,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,latent))
        self.bias=nn.Parameter(torch.zeros(()))

    def encode_material(self,gamma):
        if gamma.ndim!=3 or list(gamma.shape[1:])!=self.config['grid']:raise ValueError('gamma must have shape B,X,Y on the fixed grid')
        return self.branch(gamma[:,None])

    def predict_from_encoding(self,encoding,coordinates):
        if coordinates.shape[-1]!=5:raise ValueError('Five normalized coordinate channels required')
        basis=self.trunk(coordinates)
        if basis.ndim==2:return torch.einsum('bp,np->bn',encoding,basis)+self.bias
        if basis.ndim==3:return torch.einsum('bp,bnp->bn',encoding,basis)+self.bias
        raise ValueError('Coordinates must have shape N,5 or B,N,5')

    def forward(self,gamma,coordinates):return self.predict_from_encoding(self.encode_material(gamma),coordinates)


def coordinate_batch(sample,indices):
    """indices N,3 in x/y/time order; normalization matches existing CPU encoder."""
    x=sample['x_m'].to(sample['gamma']);y=sample['y_m'].to(sample['gamma']);t=sample['time_seconds'].to(sample['gamma']);source=sample['source_xy_m'].to(sample['gamma'])
    ix,iy,it=indices.unbind(-1)
    return torch.stack([(x/x[-1])[ix],(y/y[-1])[iy],(t/t[-1])[it],(source[0]/x[-1]).expand(ix.shape),(source[1]/y[-1]).expand(ix.shape)],dim=-1)
