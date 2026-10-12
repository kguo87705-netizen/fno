"""Four fixed-budget inversion trajectories and common-scale reconstruction plot."""
from pathlib import Path
import json,torch,numpy as np,matplotlib.pyplot as plt
out=Path('.build/author-observation-control-20261008');r=json.loads((out/'summary.json').read_text());frozen=torch.load(out/'frozen_observations.pt',weights_only=True,map_location='cpu');fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
fields=[frozen['truth'],torch.load(out/'released_uniform/states.pt',weights_only=True,map_location='cpu')['history'][-1:],torch.load(out/'self_consistent_uniform/states.pt',weights_only=True,map_location='cpu')['history'][-1:]]
for ax,f,title in zip(axes[0],fields,['Truth','Released observations, uniform init','Self-consistent observations, uniform init']):
    im=ax.imshow(f[0,0,1:-1,1:-1].T,origin='lower',extent=(0,100,0,50),vmin=0,vmax=1);ax.set(title=title,xlabel='x (mm)',ylabel='y (mm)')
fig.colorbar(im,ax=list(axes[0]),label='gamma',shrink=.8)
for key,arm in r['results'].items():
    label=key.replace('self_consistent','self').replace('_',' ');x=[z['update'] for z in arm['records']];axes[1,0].plot(x,[z['waveform_relative_l2'] for z in arm['records']],label=label);axes[1,1].plot(x,[z['physical_material_half_mse'] for z in arm['records']],label=label)
axes[1,0].set(title='Waveform error against each frozen target',xlabel='Updates',ylabel='Relative L2');axes[1,0].legend(fontsize=8);axes[1,1].set(title='Physical-grid material error',xlabel='Updates',ylabel='Half MSE',yscale='log');axes[1,1].legend(fontsize=8)
axes[1,2].axis('off');lines=['Case 1, fixed 35 updates, Adam lr0.02','Same solver, source, mask, clip and bounds','Near truth = 0.95*truth + 0.05 (diagnostic)','Main result is final35; no truth-based stopping','']
for k,a in r['results'].items():lines+=[k.replace('self_consistent','self'),f"cost ratio {a['final_cost_over_initial']:.4f}; material ratio {a['final_material_mse_over_initial']:.4f}"]
axes[1,2].text(0,1,'\n'.join(lines),va='top',fontsize=9);fig.savefig(out/'observation_consistency_control.png',dpi=180);plt.close(fig)
