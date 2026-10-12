"""Matched seven-channel DeepONet with zero or fixed physical-time features."""
import torch
from torch import nn
from basic_deeponet import BasicDeepONet,coordinate_batch

class TimeFeatureDeepONet(BasicDeepONet):
    def __init__(self):
        super().__init__()
        self.trunk[0]=nn.Linear(7,self.config['hidden'])
        self.config.update(trunk_input_channels=7,coordinate_order=self.config['coordinate_order']+['time_sin','time_cos'])

    def predict_from_encoding(self,encoding,coordinates):
        if coordinates.shape[-1]!=7:raise ValueError('Seven coordinate channels required')
        basis=self.trunk(coordinates)
        if basis.ndim==2:return torch.einsum('bp,np->bn',encoding,basis)+self.bias
        if basis.ndim==3:return torch.einsum('bp,bnp->bn',encoding,basis)+self.bias
        raise ValueError('Coordinates must be N,7 or B,N,7')

def feature_table(sample,mode):
    if mode=='zero':return torch.zeros(sample['time_seconds'].numel(),2,dtype=sample['gamma'].dtype)
    if mode!='sincos':raise ValueError('mode must be zero or sincos')
    phase=2*torch.pi*500000*sample['time_seconds'].double()
    return torch.stack([phase.sin(),phase.cos()],-1).to(sample['gamma'])

def coordinate_batch7(sample,indices,mode):
    return torch.cat([coordinate_batch(sample,indices),feature_table(sample,mode)[indices[:,2]]],-1)

def grid_predict(model,sample,gamma,device,mode,chunk=262144):
    nx,ny=gamma.shape;nt=sample['time_seconds'].numel()
    x=sample['x_m'].to(sample['gamma']);y=sample['y_m'].to(sample['gamma']);t=sample['time_seconds'].to(sample['gamma']);source=sample['source_xy_m'].to(sample['gamma'])
    sx=(source[0]/x[-1]).to(device);sy=(source[1]/y[-1]).to(device);x=(x/x[-1]).to(device);y=(y/y[-1]).to(device);t=(t/t[-1]).to(device);features=feature_table(sample,mode).to(device)
    output=torch.empty(nx*ny*nt,device=device)
    with torch.no_grad():
        b=model.encode_material(gamma[None].to(device))
        for start in range(0,output.numel(),chunk):
            ids=torch.arange(start,min(start+chunk,output.numel()),device=device);ix=ids//(ny*nt);iy=(ids//nt)%ny;it=ids%nt
            c=torch.cat([torch.stack([x[ix],y[iy],t[it],sx.expand(ids.shape),sy.expand(ids.shape)],dim=-1),features[it]],-1)
            output[start:start+ids.numel()]=model.predict_from_encoding(b,c)[0]
    return output.reshape(nx,ny,nt)
