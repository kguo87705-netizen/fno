import torch
p='.build/tx-action-material-scan-20261011/full_tx__gamma_0.50.pt'; d=torch.load(p,weights_only=False); x=d['traces_full']
for i in range(4): print(i,torch.sqrt(torch.mean(x[i]**2)).item())
