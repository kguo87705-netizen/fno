"""131-channel DeepONet trunk for the preregistered temporal-feature control."""
import torch
from torch import nn
from basic_deeponet import BasicDeepONet, coordinate_batch

class MultiFrequencyDeepONet(BasicDeepONet):
    def __init__(self):
        super().__init__()
        self.trunk[0] = nn.Linear(131, self.config['hidden'])
        self.config.update(trunk_input_channels=131, coordinate_order=self.config['coordinate_order'] + [f'time_sin_{k}' for k in range(1,64)] + [f'time_cos_{k}' for k in range(1,64)])

    def predict_from_encoding(self, encoding, coordinates):
        if coordinates.shape[-1] != 131:
            raise ValueError('131 coordinate channels required')
        basis = self.trunk(coordinates)
        if basis.ndim == 2:
            return torch.einsum('bp,np->bn', encoding, basis) + self.bias
        if basis.ndim == 3:
            return torch.einsum('bp,bnp->bn', encoding, basis) + self.bias
        raise ValueError('Coordinates must be N,131 or B,N,131')

def feature_table(sample, mode):
    n = sample['time_seconds'].numel()
    if mode == 'zero126':
        return torch.zeros(n, 126, dtype=sample['gamma'].dtype)
    if mode != 'fourier126':
        raise ValueError('mode must be zero126 or fourier126')
    tau = sample['time_seconds'].double() / sample['time_seconds'][-1].double()
    k = torch.arange(1, 64, dtype=torch.float64)
    angle = 2 * torch.pi * tau[:, None] * k[None]
    return torch.cat([angle.sin(), angle.cos()], -1).to(sample['gamma'])

def coordinate_batch131(sample, indices, mode):
    return torch.cat([coordinate_batch(sample, indices), feature_table(sample, mode)[indices[:, 2]]], -1)

def grid_predict(model, sample, gamma, device, mode, chunk=262144):
    nx, ny = gamma.shape; nt = sample['time_seconds'].numel()
    x = sample['x_m'].to(sample['gamma']); y = sample['y_m'].to(sample['gamma']); t = sample['time_seconds'].to(sample['gamma']); source = sample['source_xy_m'].to(sample['gamma'])
    sx=(source[0]/x[-1]).to(device); sy=(source[1]/y[-1]).to(device); x=(x/x[-1]).to(device); y=(y/y[-1]).to(device); t=(t/t[-1]).to(device); features=feature_table(sample,mode).to(device)
    output=torch.empty(nx*ny*nt,device=device)
    with torch.no_grad():
        b=model.encode_material(gamma[None].to(device))
        for start in range(0,output.numel(),chunk):
            ids=torch.arange(start,min(start+chunk,output.numel()),device=device); ix=ids//(ny*nt); iy=(ids//nt)%ny; it=ids%nt
            c=torch.cat([torch.stack([x[ix],y[iy],t[it],sx.expand(ids.shape),sy.expand(ids.shape)],-1),features[it]],-1)
            output[start:start+ids.numel()]=model.predict_from_encoding(b,c)[0]
    return output.reshape(nx,ny,nt)
