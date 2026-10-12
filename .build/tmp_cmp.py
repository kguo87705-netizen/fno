import torch
for m in ['0.46','0.50','0.54']:
 a=torch.load(f'.build/tx-action-material-scan-20261011/full_tx__gamma_{m}.pt',weights_only=False)['traces_full']
 b=torch.load(f'.build/tx-action-material-scan-20261011/right_tx__gamma_{m}.pt',weights_only=False)['traces_full']
 for i in range(2): print(m,i,torch.max(torch.abs(a[i+2]-b[i])).item(),torch.sqrt(torch.mean((a[i+2]-b[i])**2)).item())
