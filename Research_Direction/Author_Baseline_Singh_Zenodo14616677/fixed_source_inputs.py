"""Stream bit-identical eight-channel inputs for a fixed source and grid."""
import torch
from time_feature_input import encode_time_features


class FixedSourceInputs:
    def __init__(self,samples,device):
        self.samples=samples
        first=encode_time_features(samples[0],mode='zero')
        self.shape=tuple(first.shape[1:]);self.gamma=[s['gamma'].to(device) for s in samples]
        self.x=first[1,:,0,0].clone().to(device);self.y=first[2,0,:,0].clone().to(device)
        self.t=first[3,0,0,:].clone().to(device);self.sx=first[4,0,0,0].clone().to(device);self.sy=first[5,0,0,0].clone().to(device)
        self.zero=self.sx.new_zeros(())
        for s in samples:
            for key in ['x_m','y_m','time_seconds','source_xy_m','receiver_indices']:
                assert torch.equal(samples[0][key],s[key])

    def __getitem__(self,index):
        shape=self.shape
        return torch.stack([self.gamma[index][...,None].expand(shape),self.x[:,None,None].expand(shape),
            self.y[None,:,None].expand(shape),self.t[None,None,:].expand(shape),self.sx.expand(shape),self.sy.expand(shape),
            self.zero.expand(shape),self.zero.expand(shape)])[None]

    def __len__(self):return len(self.samples)
