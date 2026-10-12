import torch
root='.build/tx-action-material-scan-20261011-r4';
for m in ['0.46','0.50','0.54']:
 a=torch.load(f'{root}/full_tx__gamma_{m}.pt',weights_only=False)['traces_full']; b=torch.load(f'{root}/right_tx__gamma_{m}.pt',weights_only=False)['traces_full']; print(m,[torch.max(torch.abs(a[i+2]-b[i])).item() for i in range(2)])
for a in ['full_tx','left_tx','right_tx']:
 d=torch.load(f'{root}/{a}__gamma_0.50.pt',weights_only=False); print(a,[round(torch.sqrt(torch.mean(x**2)).item(),6) for x in d['traces_full']])
