"""Final fixed-budget paired temporal-feature results and diagnostic figure."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.run/'comparison.json').read_text());response=json.loads((a.run/'material_response.json').read_text())
    labels=['Zero extra channels','Source sin/cos features'];colors=['tab:blue','tab:orange']
    summary={'scope':'One seed, fixed epoch20, train12 validation6 workpieces, same eight-channel initialization and order; no test or gradient/FWI evaluation.',
        'training':[{k:v for k,v in x.items() if k!='records'} for x in r['final_training']],
        'validation':[{k:v for k,v in x.items() if k!='records'} for x in r['final_validation']],
        'material_response':[{k:v for k,v in x.items() if k!='records'} for x in response['models']]}
    if (a.run/'scatter_target_spectrum.json').exists():summary['scatter_target_spectrum']=json.loads((a.run/'scatter_target_spectrum.json').read_text())['summary']
    if (a.run/'feature_use.json').exists():summary['feature_use']={k:v for k,v in json.loads((a.run/'feature_use.json').read_text()).items() if k!='records'}
    (a.run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for arm,(label,color,h) in enumerate(zip(labels,colors,r['history'])):
        for col,key in enumerate(['scatter_field_relative_l2','scatter_trace_relative_l2']):
            ax[0,col].plot([x['epoch'] for x in h],[x['validation'][key] for x in h],label=label,color=color)
    for col,title in enumerate(['Validation scatter field error','Validation scatter receiver error']):
        ax[0,col].axhline(1,color='k',linestyle='--',label='Zero scattering');ax[0,col].set(title=title,xlabel='Epoch',ylabel='Relative L2');ax[0,col].legend(fontsize=8)
    examples=[torch.load(a.run/f'validation_example{i}.pt',weights_only=True,map_location='cpu') for i in range(2)]
    t=np.arange(1,2001)*.03
    ax[0,2].plot(t,examples[0]['reference_scatter_traces'][12].numpy(),color='k',label='Reference',linewidth=1)
    for label,color,ex in zip(labels,colors,examples):ax[0,2].plot(t,ex['prediction_scatter_traces'][12].numpy(),color=color,label=label,alpha=.8)
    ax[0,2].set(title='Validation004, shot0, receiver12: scatter',xlabel='Time (us)',ylabel='Scalar scattering u');ax[0,2].legend(fontsize=8)
    errors=[x['prediction_residual_field'][:,:,250]-x['reference_residual_field'][:,:,250] for x in examples];vmax=max(float(x.abs().max()) for x in errors)
    for i,label in enumerate(labels):
        im=ax[1,i].imshow(errors[i].T.numpy(),origin='lower',extent=(0,100,0,50),vmin=-vmax,vmax=vmax,cmap='RdBu_r')
        ax[1,i].set(title=f'{label}: error at30.12us',xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=list(ax[1,:2]),shrink=.7,label='Normalized scatter field error')
    positions=np.arange(3)
    for i,(label,color,v,m) in enumerate(zip(labels,colors,r['final_validation'],response['models'])):
        ax[1,2].bar(positions+(i-.5)*.35,[v['scatter_field_relative_l2'],v['scatter_trace_relative_l2'],m['secant_response_relative_l2']],width=.35,label=label,color=color)
    ax[1,2].axhline(1,color='k',linestyle='--',label='Zero scatter/response');ax[1,2].set_xticks(positions,['Scatter field','Scatter receiver','Material response'],rotation=20,ha='right')
    ax[1,2].set(title='Final validation',ylabel='Relative L2');ax[1,2].legend(fontsize=8)
    fig.savefig(a.run/'time_feature_comparison.png',dpi=160);plt.close(fig);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
