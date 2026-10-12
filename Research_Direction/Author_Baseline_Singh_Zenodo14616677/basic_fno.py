"""Ordinary dense Fourier layers over x,y,t, without physics/residual losses."""
import torch
from torch import nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    def __init__(self,width,modes):
        super().__init__()
        self.modes=tuple(modes)
        self.weights=nn.ParameterList([nn.Parameter(torch.randn(width,width,*modes,dtype=torch.cfloat)/(width*width)) for _ in range(4)])

    def forward(self,x):
        mx,my,mt=self.modes
        assert 2*mx<=x.shape[-3] and 2*my<=x.shape[-2] and mt<=x.shape[-1]//2+1
        spectrum=torch.fft.rfftn(x,dim=(-3,-2,-1),norm='ortho')
        out=torch.zeros_like(spectrum)
        quadrants=[(slice(0,mx),slice(0,my)),(slice(-mx,None),slice(0,my)),(slice(0,mx),slice(-my,None)),(slice(-mx,None),slice(-my,None))]
        for weight,(sx,sy) in zip(self.weights,quadrants):
            out[:,:,sx,sy,:mt]=torch.einsum('bixyt,ioxyt->boxyt',spectrum[:,:,sx,sy,:mt],weight)
        return torch.fft.irfftn(out,s=x.shape[-3:],dim=(-3,-2,-1),norm='ortho')


class BasicFNO(nn.Module):
    def __init__(self,width=4,modes=(16,8,64),layers=4,input_channels=6):
        super().__init__()
        self.config={'width':width,'modes':list(modes),'layers':layers,'input_channels':input_channels,'output_channels':1,'padding':0}
        self.lift=nn.Conv3d(input_channels,width,1)
        self.spectral=nn.ModuleList([SpectralConv3d(width,modes) for _ in range(layers)])
        self.local=nn.ModuleList([nn.Conv3d(width,width,1) for _ in range(layers)])
        self.projection=nn.Sequential(nn.Conv3d(width,width*2,1),nn.GELU(),nn.Conv3d(width*2,1,1))

    def forward(self,x):
        x=self.lift(x)
        for i,(spectral,local) in enumerate(zip(self.spectral,self.local)):
            x=spectral(x)+local(x)
            if i<len(self.spectral)-1:x=F.gelu(x)
        return self.projection(x)
