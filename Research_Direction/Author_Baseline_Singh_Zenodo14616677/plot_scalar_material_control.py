"""Plot scalar contrast trajectories and corresponding free-pixel controls."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path('.build/author-scalar-material-control-20261008')
r = json.loads((out/'summary.json').read_text())
old = json.loads(Path('.build/author-observation-control-20261008/summary.json').read_text())
fig,axes = plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for row,name in enumerate(['released','self_consistent']):
    for initial,color in [('uniform','tab:blue'),('near_truth','tab:orange')]:
        key = name+'_'+initial
        records = r['results'][key]['records']
        before = old['results'][key]['records']
        x = [z['update'] for z in records]
        axes[row,0].plot(x,[z['contrast'] for z in records],color=color,label=initial)
        axes[row,1].plot(x,[z['waveform_relative_l2']*100 for z in records],color=color,label=initial+' scalar')
        axes[row,1].plot(x,[z['waveform_relative_l2']*100 for z in before],color=color,ls='--',label=initial+' free pixels')
        axes[row,2].plot(x,[z['physical_material_half_mse'] for z in records],color=color,label=initial+' scalar')
        axes[row,2].plot(x,[z['physical_material_half_mse'] for z in before],color=color,ls='--',label=initial+' free pixels')
    axes[row,0].axhline(1e-5,color='gray',ls=':',label='true contrast')
    axes[row,0].set(yscale='log',ylabel='Contrast a',title=name+': known-support scalar')
    axes[row,1].set(ylabel='Waveform relative L2 (%)',title=name+': own observation target')
    axes[row,2].set_yscale('symlog',linthresh=1e-10)
    axes[row,2].set(ylabel='Physical-grid half MSE',title=name+': material distance')
    for ax in axes[row]:
        ax.set(xlabel='Updates')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
fig.suptitle('Fixed35 Adam lr0.02; same initial fields and targets; geometry is oracle information',fontsize=13)
fig.savefig(out/'scalar_material_control.png',dpi=180)
plt.close(fig)
