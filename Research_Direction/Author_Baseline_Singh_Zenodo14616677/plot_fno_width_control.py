"""Width4/8 fixed-budget curves, strong references and predetermined trace."""
from pathlib import Path
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('.build/fno-width8-control-20261009');base=Path('.build/fno-local-contrast-20261009')
r=json.loads((out/'comparison.json').read_text());old=json.loads((base/'comparison.json').read_text())
initial=json.loads((out/'initial_model.json').read_text())
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for report,width,style in [(old,4,'--'),(r,8,'-')]:
    for split,color,short in [('training','tab:blue','train'),('development_interpolation','tab:orange','interp')]:
        x=[z['update'] for z in report['history']]
        for ax,key in [(axes[0,0],'scatter_trace_relative_l2'),(axes[0,1],'scatter_field_relative_l2')]:
            ax.plot(x,[z[split][key]*100 for z in report['history']],marker='o',ls=style,color=color,label=f'w{width} {short}')
axes[0,0].set(title='Scatter receivers, full2000',ylabel='Relative L2 (%)',xlabel='Updates')
axes[0,1].set(title='Scatter field, stored500',ylabel='Relative L2 (%)',xlabel='Updates')
for ax in [axes[0,0],axes[0,1]]:ax.legend(fontsize=8);ax.grid(alpha=.25)
names=['w4','w8','mean','nearest','linear']
items=[old['final_interpolation'],r['final_interpolation']]+[r['references'][k]['development_interpolation'] for k in ['training_mean','contrast_nearest','piecewise_linear']]
x=np.arange(len(names))
axes[0,2].bar(x-.18,[z['scatter_field_relative_l2']*100 for z in items],.36,label='field')
axes[0,2].bar(x+.18,[z['scatter_trace_relative_l2']*100 for z in items],.36,label='receiver')
axes[0,2].set(xticks=x,xticklabels=names,title='Final1200 interpolation vs references',ylabel='Relative L2 (%)');axes[0,2].legend()
new_example=torch.load(out/'interpolation_example.pt',weights_only=True,map_location='cpu')
old_example=torch.load(base/'interpolation_example.pt',weights_only=True,map_location='cpu')
t=torch.arange(1,2001)*.03
axes[1,0].plot(t,new_example['reference_scatter_traces'][12],label='FD',lw=1.4)
axes[1,0].plot(t,old_example['prediction_scatter_traces'][12],label='width4',alpha=.8)
axes[1,0].plot(t,new_example['prediction_scatter_traces'][12],label='width8',alpha=.8)
axes[1,0].set(title='Predetermined a=0.4, receiver12',xlabel='Time (microseconds)',ylabel='Scattered wave');axes[1,0].legend(fontsize=8)
for report,width,offset in [(old,4,-.18),(r,8,.18)]:
    records=report['final_training']['records']+report['final_interpolation']['records']
    axes[1,1].bar(np.arange(5)+offset,[100*(z['field_error']/z['scatter_field_true'])**.5 for z in records],.36,label=f'width{width}')
axes[1,1].set(xticks=np.arange(5),xticklabels=['0.3 T','0.5 T','0.7 T','0.4 D','0.6 D'],title='All individual scatter field errors',ylabel='Relative L2 (%)');axes[1,1].legend()
axes[1,2].axis('off')
lines=['Same five frozen files, source0, scales','Same Adam/loss, 1200 updates, 400 per material',
       'Seed2; different-dimensional initialization','Equal updates, not equal compute','',
       f"Width4 real params: {initial['width4_counts']['real_parameters']:,}",
       f"Width8 real params: {initial['counts']['real_parameters']:,}",
       f"Width4 run time: {old['elapsed_seconds']:.1f}s",f"Width8 run time: {r['elapsed_seconds']:.1f}s"]
if r['cuda_memory']['peak_allocated_bytes'] is not None:
    lines += [f"Width8 peak allocated: {r['cuda_memory']['peak_allocated_bytes']/2**30:.2f}GiB"]
lines += ['','No new geometry or material derivative check','No end-to-end inversion speedup claim']
axes[1,2].text(0,1,'\n'.join(lines),va='top',fontsize=10)
fig.suptitle('One width diagnostic, fixed final1200 main result',fontsize=13)
fig.savefig(out/'fno_width_control.png',dpi=180);plt.close(fig)
