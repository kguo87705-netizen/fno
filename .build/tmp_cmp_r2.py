import torch
root='.build/tx-action-material-scan-20261011-r2';
for m in ['0.46','0.50','0.54']:
 a=torch.load(f'{root}/full_tx__gamma_{m}.pt',weights_only=False)['traces_full']; b=torch.load(f'{root}/right_tx__gamma_{m}.pt',weights_only=False)['traces_full']; print(m,[round(torch.max(torch.abs(a[i+2]-b[i])).item(),8) for i in range(2)])
