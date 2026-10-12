import torch, numpy as np
root='.build/tx-action-material-scan-20261011'; vals=[.46,.48,.50,.52,.54]
for action in ['full_tx','left_tx','right_tx']:
 xs=[]
 for v in vals:
  d=torch.load(f'{root}/{action}__gamma_{v:.2f}.pt',map_location='cpu',weights_only=False); xs.append(d['traces_full'].numpy())
 xs=np.array(xs)
 print('\n'+action, xs.shape)
 for i,s in enumerate(d['source_indices']):
  center=xs[2,i]; dr=np.sqrt(np.mean((xs[4,i]-xs[0,i])**2)); print('src',s,'mid_rms',round(float(np.sqrt(np.mean(center**2))),8),'delta_rms',round(float(dr),8),'slope',round(float(dr/.08),8))
 for i in range(xs.shape[1]):
  y=xs[:,i].reshape(5,-1); cor=[]
  for j in range(y.shape[1]): cor.append(np.corrcoef(vals,y[:,j])[0,1] if np.std(y[:,j])>1e-12 else 0)
  print('src',d['source_indices'][i],'median_corr',round(float(np.median(cor)),4),'frac_abs_gt09',round(float(np.mean(np.abs(cor)>0.9)),4))
