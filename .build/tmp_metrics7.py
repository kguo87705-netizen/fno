import torch,numpy as np
root='.build/tx-action-material-scan-20261011-r7'; vals=[.46,.48,.50,.52,.54]
for action in ['full_tx','left_tx','right_tx']:
 xs=[]
 for v in vals: xs.append(torch.load(f'{root}/{action}__gamma_{v:.2f}.pt',weights_only=False)['traces_full'].numpy())
 xs=np.array(xs); print('\n'+action)
 for i,s in enumerate(torch.load(f'{root}/{action}__gamma_0.50.pt',weights_only=False)['source_indices']):
  mid=np.sqrt(np.mean(xs[2,i]**2)); dr=np.sqrt(np.mean((xs[4,i]-xs[0,i])**2)); d48_52=np.sqrt(np.mean((xs[3,i]-xs[1,i])**2)); print(s,'mid_rms',round(float(mid),6),'delta46-54',round(float(dr),6),'center_slope',round(float(d48_52/.04),6))
