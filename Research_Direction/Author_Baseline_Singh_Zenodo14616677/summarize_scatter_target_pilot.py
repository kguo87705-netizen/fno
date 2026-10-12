"""Plot fixed-budget direct/scattering-target controls on their one training shot."""
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
    r=json.loads((args.run/'comparison.json').read_text());control=Path(r['protocol']['baseline_control']).parent
    old=json.loads((control/'history.json').read_text());h=r['history']
    example=torch.load(args.run/'example.pt',weights_only=True,map_location='cpu')
    direct=torch.load(control/'example.pt',weights_only=True,map_location='cpu')
    field_baseline=h[-1]['homogeneous_field_relative_l2'];trace_baseline=h[-1]['homogeneous_trace_relative_l2']
    summary={'scope':'Same supervised train workpiece 000, shot0, fixed400 updates, same raw initial FNO, old scales and receiver weight1.',
        'zero_scatter_predictor':{'scatter_field_relative_l2':1.,'scatter_trace_relative_l2':1.,'full_field_relative_l2':field_baseline,'full_trace_relative_l2':trace_baseline},
        'direct_target':{'scatter_field_relative_l2':old[-1]['field_relative_l2']/field_baseline,
            'scatter_trace_relative_l2':old[-1]['trace_relative_l2']/trace_baseline,
            'full_field_relative_l2':old[-1]['field_relative_l2'],'full_trace_relative_l2':old[-1]['trace_relative_l2']},
        'scatter_target':h[-1],
        'caution':'Low full-wave error can follow from fixed homogeneous base alone. Judge learning by scatter error, not the base signal.'}
    (args.run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    x=[p['update'] for p in h]
    ax[0,0].plot(x,[p['scatter_field_relative_l2'] for p in h],label='Scatter target: field')
    ax[0,0].plot(x,[p['scatter_trace_relative_l2'] for p in h],label='Scatter target: receivers')
    ax[0,0].axhline(1,color='k',linestyle='--',label='Zero scattering')
    ax[0,0].set(title='Scatter fitting on one training shot',xlabel='Update',ylabel='Scatter relative L2');ax[0,0].legend(fontsize=8)
    ax[0,1].plot(x,[p['full_trace_relative_l2'] for p in h],label='Scatter target')
    ax[0,1].plot([p['update'] for p in old],[p['trace_relative_l2'] for p in old],label='Direct target')
    ax[0,1].axhline(trace_baseline,color='k',linestyle='--',label='Homogeneous')
    ax[0,1].set(title='Full receiver error',xlabel='Update',ylabel='Relative L2');ax[0,1].legend(fontsize=8)
    t=np.arange(1,2001)*.03
    true_scatter=example['reference_scatter_traces'][12].numpy();pred_scatter=example['prediction_scatter_traces'][12].numpy()
    direct_scatter=(direct['prediction_traces'][12]-example['homogeneous_traces'][12]).numpy()
    for trace,label in [(true_scatter,'Reference scatter'),(pred_scatter,'Scatter target'),(direct_scatter,'Direct target minus base')]:ax[1,0].plot(t,trace,label=label,alpha=.8)
    ax[1,0].set(title='Training receiver 12: scattering',xlabel='Time (us)',ylabel='Scalar scattering u');ax[1,0].legend(fontsize=8)
    categories=['Scatter field','Scatter receivers','Full field','Full receivers'];positions=np.arange(4)
    for i,key in enumerate(['zero_scatter_predictor','direct_target','scatter_target']):
        metrics=summary[key];vals=[metrics[k] for k in ['scatter_field_relative_l2','scatter_trace_relative_l2','full_field_relative_l2','full_trace_relative_l2']]
        ax[1,1].bar(positions+(i-1)*.25,vals,width=.25,label=['Zero scatter','Direct target','Scatter target'][i])
    ax[1,1].set_xticks(positions,categories,rotation=15,ha='right');ax[1,1].set(title='Final fixed-budget errors',ylabel='Relative L2');ax[1,1].legend(fontsize=8)
    fig.savefig(args.run/'scatter_target_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
