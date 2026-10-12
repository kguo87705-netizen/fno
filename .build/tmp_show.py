import torch
for a in ['full_tx','left_tx','right_tx']:
 d=torch.load(f'.build/tx-action-material-scan-20261011/{a}__gamma_0.50.pt',weights_only=False)
 print(a,d['source_indices'],d['source_xy_m'])
