"""Fixed source-frequency features; no learned or validation-selected frequency."""
import torch
from operator_data import encode_fno_input

SOURCE_FREQUENCY_HZ = 500000.


def encode_time_features(sample, gamma=None, mode='zero'):
    base=encode_fno_input(sample,gamma)
    if mode=='zero':
        extra=base.new_zeros((2,*base.shape[1:]))
    elif mode=='sincos':
        phase=sample['time_seconds'].to(device=base.device,dtype=torch.float64)*(2*torch.pi*SOURCE_FREQUENCY_HZ)
        extra=torch.stack([phase.sin(),phase.cos()]).to(base)[:,None,None,:].expand(2,*base.shape[1:])
    else:
        raise ValueError(f'Unknown time feature mode: {mode}')
    return torch.cat([base,extra],dim=0)
