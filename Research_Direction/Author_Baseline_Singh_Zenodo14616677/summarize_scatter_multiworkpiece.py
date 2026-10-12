"""Compare scattering-target final validation with matched direct-target arm."""
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
    r=json.loads((args.run/'comparison.json').read_text());control=Path(r['protocol']['control'])
    old=json.loads((control/'comparison.json').read_text());response=json.loads((args.run/'material_response.json').read_text())
    old_response=json.loads((control/'material_response.json').read_text())['models'][1]
    direct=old['final_validation'][1];scatter=r['final_validation'];h=r['history'];oh=old['histories'][1]
    ex=torch.load(args.run/'validation_example.pt',weights_only=True,map_location='cpu')
    oe=torch.load(control/'validation_example_weight1.pt',weights_only=True,map_location='cpu')
    direct_summary={'full_field_relative_l2':direct['field_relative_l2'],'full_trace_relative_l2':direct['trace_relative_l2'],
        'scatter_field_relative_l2':direct['field_relative_l2']/direct['homogeneous_field_relative_l2'],
        'scatter_trace_relative_l2':direct['scatter_normalized_trace_error']}
    summary={'scope':'Fixed20 epoch, same seed/order/model/scales, validation6 groups24 shots. Not an independent test.',
        'direct_target_validation':direct_summary,'scatter_target_validation':{k:v for k,v in scatter.items() if k!='records'},
        'scatter_target_training':{k:v for k,v in r['final_training'].items() if k!='records'},
        'direct_material_response':{k:v for k,v in old_response.items() if k!='records'},
        'scatter_material_response':{k:v for k,v in response['models'][0].items() if k!='records'}}
    (args.run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(2,3,figsize=(16,8),layout='constrained');epochs=[x['epoch'] for x in h]
    ax[0,0].plot(epochs,[x['validation']['full_trace_relative_l2'] for x in h],label='Scatter target')
    ax[0,0].plot(epochs,[x['validation']['trace_relative_l2'] for x in oh],label='Direct target')
    ax[0,0].axhline(scatter['homogeneous_trace_relative_l2'],color='k',linestyle='--',label='Homogeneous')
    ax[0,0].set(title='Validation full receiver error',xlabel='Epoch',ylabel='Relative L2');ax[0,0].legend(fontsize=8)
    ax[0,1].plot(epochs,[x['validation']['scatter_field_relative_l2'] for x in h],label='Scatter field')
    ax[0,1].plot(epochs,[x['validation']['scatter_trace_relative_l2'] for x in h],label='Scatter receiver')
    ax[0,1].axhline(1,color='k',linestyle='--',label='Zero scattering')
    ax[0,1].set(title='Validation scattering errors',xlabel='Epoch',ylabel='Relative L2');ax[0,1].legend(fontsize=8)
    t=np.arange(1,2001)*.03
    ax[0,2].plot(t,ex['reference_scatter_traces'][12].numpy(),label='Reference',color='k',linewidth=1)
    ax[0,2].plot(t,ex['prediction_scatter_traces'][12].numpy(),label='Scatter target',alpha=.8)
    ax[0,2].plot(t,(oe['prediction_traces'][12]-ex['homogeneous_traces'][12]).numpy(),label='Direct minus base',alpha=.7)
    ax[0,2].set(title='Validation004, shot0, receiver12: scatter',xlabel='Time (us)',ylabel='Scalar scattering u');ax[0,2].legend(fontsize=8)
    residuals=[oe['prediction_field'][:,:,250]-oe['truth_field'][:,:,250],ex['prediction_residual_field'][:,:,250]-ex['reference_residual_field'][:,:,250]]
    vmax=max(float(x.abs().max()) for x in residuals)
    for i,label in enumerate(['Direct target','Scatter target']):
        im=ax[1,i].imshow(residuals[i].T.numpy(),origin='lower',extent=(0,100,0,50),vmin=-vmax,vmax=vmax,cmap='RdBu_r')
        ax[1,i].set(title=f'{label}: field error at30.12us',xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=list(ax[1,:2]),shrink=.7,label='Normalized field error')
    categories=['Scatter field','Scatter receivers','Material response'];positions=np.arange(3)
    for i,(label,values) in enumerate([('Direct target',[direct_summary['scatter_field_relative_l2'],direct_summary['scatter_trace_relative_l2'],old_response['secant_response_relative_l2']]),('Scatter target',[scatter['scatter_field_relative_l2'],scatter['scatter_trace_relative_l2'],response['models'][0]['secant_response_relative_l2']])]):
        ax[1,2].bar(positions+(i-.5)*.35,values,width=.35,label=label)
    ax[1,2].axhline(1,color='k',linestyle='--',label='Zero scatter/response');ax[1,2].set_xticks(positions,categories,rotation=20,ha='right')
    ax[1,2].set(title='Final validation: scattering and response',ylabel='Relative L2');ax[1,2].legend(fontsize=8)
    fig.savefig(args.run/'scatter_multiworkpiece_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
