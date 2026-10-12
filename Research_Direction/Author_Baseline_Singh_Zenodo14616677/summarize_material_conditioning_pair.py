"""Plot pair-conditioned predictions against the optimal shared-target reference."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=json.loads((a.run/'comparison.json').read_text());ex=torch.load(a.run/'example.pt',weights_only=True,map_location='cpu')
    summary={'scope':'One seed, two fitted training workpieces at source0, fixed800 updates each arm. No held-out accuracy or inverse claim.',
        'mean_oracle':r['mean_oracle'],'zero_scatter':r['zero_scatter'],'final':r['final']}
    (a.run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(2,3,figsize=(16,8),layout='constrained');labels=['True material','Constant material'];colors=['tab:blue','tab:orange']
    for arm,(label,color) in enumerate(zip(labels,colors)):
        ax[0,0].plot([x['update'] for x in r['history']],[x['arms'][arm]['mean_total_objective'] for x in r['history']],marker='o',label=label,color=color)
        ax[0,1].plot([x['update'] for x in r['history']],[x['arms'][arm]['pair_trace_response']['relative_l2'] for x in r['history']],marker='o',label=label,color=color)
    ax[0,0].axhline(r['mean_oracle']['mean_total_objective'],color='k',linestyle='--',label='Optimal shared target');ax[0,0].set_yscale('log');ax[0,0].set(title='Mean training objective',xlabel='Updates',ylabel='Field + receiver MSE');ax[0,0].legend(fontsize=8)
    ax[0,1].axhline(1,color='k',linestyle='--',label='Zero pair response');ax[0,1].set(title='Two-material receiver difference error',xlabel='Updates',ylabel='Relative L2');ax[0,1].legend(fontsize=8)
    t=np.arange(1,2001)*.03;ref=ex['reference_scatter_traces']
    ax[0,2].plot(t,(ref[0,12]-ref[1,12]).numpy(),color='k',label='Reference',linewidth=1)
    for arm,(label,color) in enumerate(zip(labels,colors)):
        pred=ex['prediction_scatter_traces'][arm];ax[0,2].plot(t,(pred[0,12]-pred[1,12]).numpy(),label=label,color=color,alpha=.8)
    ax[0,2].set(title='Workpiece000 - 001, receiver12',xlabel='Time (us)',ylabel='Scalar pair response');ax[0,2].legend(fontsize=8)
    for sample in [0,1]:
        ax[1,sample].plot(t,ref[sample,12].numpy(),color='k',label='Reference',linewidth=1)
        ax[1,sample].plot(t,ex['mean_oracle_traces'][sample,12].numpy(),color='tab:green',label='Shared-target oracle',linestyle=':')
        for arm,(label,color) in enumerate(zip(labels,colors)):ax[1,sample].plot(t,ex['prediction_scatter_traces'][arm][sample,12].numpy(),label=label,color=color,alpha=.8)
        ax[1,sample].set(title=f'Workpiece00{sample}, source0, receiver12: scatter',xlabel='Time (us)',ylabel='Scalar scattering u');ax[1,sample].legend(fontsize=8)
    positions=np.arange(3)
    for i,label in enumerate(labels+['Shared-target oracle']):
        z=r['final'][i] if i<2 else r['mean_oracle'];values=[z['scatter_field_relative_l2'],z['scatter_trace_relative_l2'],z['pair_trace_response']['relative_l2']]
        ax[1,2].bar(positions+(i-1)*.25,values,width=.25,label=label)
    ax[1,2].set_xticks(positions,['Scatter field','Scatter receiver','Pair response'],rotation=20,ha='right');ax[1,2].set(title='Final fitted-sample comparison',ylabel='Relative L2');ax[1,2].legend(fontsize=8)
    fig.savefig(a.run/'material_conditioning_pair.png',dpi=160);plt.close(fig);print(json.dumps({'final':r['final'],'mean_oracle_objective':r['mean_oracle']['mean_total_objective']}),flush=True)


if __name__=='__main__':main()
