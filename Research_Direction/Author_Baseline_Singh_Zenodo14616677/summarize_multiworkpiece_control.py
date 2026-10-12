"""Plot final-budget validation tradeoffs, without selecting a checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    report=json.loads((args.run/'comparison.json').read_text(encoding='utf-8'))
    histories=report['histories'];val=report['final_validation'];labels=['Field only','Field + receiver'];colors=['tab:blue','tab:orange']
    examples=[torch.load(args.run/f'validation_example_weight{i}.pt',weights_only=True,map_location='cpu') for i in [0,1]]
    fig,ax=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for arm in [0,1]:
        x=[r['epoch'] for r in histories[arm]]
        ax[0,0].plot(x,[r['validation']['field_relative_l2'] for r in histories[arm]],label=labels[arm],color=colors[arm])
        ax[0,1].plot(x,[r['validation']['trace_relative_l2'] for r in histories[arm]],label=labels[arm],color=colors[arm])
    ax[0,0].axhline(val[0]['homogeneous_field_relative_l2'],color='k',linestyle='--',label='Homogeneous')
    ax[0,1].axhline(val[0]['homogeneous_trace_relative_l2'],color='k',linestyle='--',label='Homogeneous')
    ax[0,0].set(title='Validation field error',xlabel='Epoch',ylabel='Relative L2');ax[0,0].legend()
    ax[0,1].set(title='Validation full receiver error',xlabel='Epoch',ylabel='Relative L2');ax[0,1].legend()
    groups=sorted({r['group_id'] for r in val[0]['records']})
    group_rows=[]
    for arm in [0,1]:
        errors=[]
        for group in groups:
            rec=[r for r in val[arm]['records'] if r['group_id']==group]
            error=np.sqrt(sum(r['trace_error'] for r in rec)/sum(r['trace_true'] for r in rec))
            errors.append(error);group_rows.append({'weight':arm,'group_id':group,'trace_relative_l2':error})
        ax[0,2].bar(np.arange(6)+(arm-.5)*.35,errors,width=.35,label=labels[arm])
    ax[0,2].set_xticks(np.arange(6),[g[-3:] for g in groups]);ax[0,2].set(title='Final validation workpieces: all 4 shots',xlabel='Workpiece ID',ylabel='Receiver relative L2');ax[0,2].legend(fontsize=8)
    t=np.arange(1,2001)*.03
    ax[1,0].plot(t,examples[0]['reference_traces'][12].numpy(),label='Reference',color='k',linewidth=1)
    for arm in [0,1]:ax[1,0].plot(t,examples[arm]['prediction_traces'][12].numpy(),label=labels[arm],color=colors[arm],alpha=.8)
    ax[1,0].set(title=f"Validation {examples[0]['group_id'][-3:]}, shot 0, receiver 12",xlabel='Time (us)',ylabel='Scalar u');ax[1,0].legend(fontsize=8)
    residuals=[e['prediction_field'][:,:,250]-e['truth_field'][:,:,250] for e in examples];vmax=max(float(r.abs().max()) for r in residuals)
    for arm in [0,1]:
        im=ax[1,arm+1].imshow(residuals[arm].T.numpy(),origin='lower',extent=(0,100,0,50),vmin=-vmax,vmax=vmax,cmap='RdBu_r')
        ax[1,arm+1].set(title=f'{labels[arm]}: residual at 30.12 us',xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=list(ax[1,1:]),label='Normalized field error',shrink=.7)
    fig.savefig(args.run/'multiworkpiece_comparison.png',dpi=160);plt.close(fig)
    summary={'protocol':report['protocol'],'final_training':[{k:v for k,v in r.items() if k!='records'} for r in report['final_training']],
        'final_validation':[{k:v for k,v in r.items() if k!='records'} for r in val],'validation_workpieces':group_rows,
        'primary_epoch':20,'checkpoint_selection':'none, final fixed-budget only'}
    (args.run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
