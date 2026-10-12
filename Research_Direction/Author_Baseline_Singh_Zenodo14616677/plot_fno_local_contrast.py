"""All fixed evaluations, simple references and a predetermined receiver."""
from pathlib import Path
import json
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('.build/fno-local-contrast-20261009')
r=json.loads((out/'comparison.json').read_text());h=r['history']
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for split,label,color in [('training','training','tab:blue'),('development_interpolation','contrast interpolation','tab:orange')]:
    axes[0,0].plot([x['update'] for x in h],[x[split]['scatter_trace_relative_l2']*100 for x in h],'o-',label=label,color=color)
    axes[0,1].plot([x['update'] for x in h],[x[split]['scatter_field_relative_l2']*100 for x in h],'o-',label=label,color=color)
axes[0,0].set(title='Full2000 scatter receiver error',xlabel='Updates',ylabel='Relative L2 (%)')
axes[0,1].set(title='Scatter field error, stored500 frames',xlabel='Updates',ylabel='Relative L2 (%)')
for ax in axes[0]:ax.legend();ax.grid(alpha=.25)
names=['FNO','zero','mean','nearest','linear']
refs=[r['final_interpolation']]+[r['references'][k]['development_interpolation'] for k in ['zero','training_mean','contrast_nearest','piecewise_linear']]
axes[1,0].bar(names,[x['scatter_trace_relative_l2']*100 for x in refs],color=['tab:orange','gray','tab:green','tab:purple','tab:blue'])
axes[1,0].set(title='Fixed1200 interpolation vs train-only references',ylabel='Scatter receiver relative L2 (%)')
example=torch.load(out/'interpolation_example.pt',weights_only=True,map_location='cpu')
t=torch.arange(1,2001)*.03
axes[1,1].plot(t,example['reference_scatter_traces'][12],label='FD truth')
axes[1,1].plot(t,example['prediction_scatter_traces'][12],label='FNO final1200',alpha=.8)
axes[1,1].set(title='Predetermined a=0.4, receiver12',xlabel='Time (microseconds)',ylabel='Scalar scattered wave')
axes[1,1].legend();axes[1,1].grid(alpha=.25)
fig.suptitle('Fixed training000 geometry/source0; three training contrasts, two interpolation contrasts',fontsize=12)
fig.savefig(out/'fno_local_contrast.png',dpi=180);plt.close(fig)
