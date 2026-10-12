"""Compare fixed-budget field-only and receiver-supervised train-shot controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator_data import OperatorPilotDataset


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--control',type=Path,required=True);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    sample=OperatorPilotDataset(args.data,'train')[0]
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    runs=[args.control,args.run];histories=[read(r/'history.json') for r in runs]
    examples=[torch.load(r/'example.pt',weights_only=True,map_location='cpu') for r in runs]
    boundaries=[read(r/'boundary_error.json') for r in runs];spectra=[read(r/'receiver_spectrum.json') for r in runs]
    homogeneous=torch.load(args.data/'homogeneous.pt',weights_only=True,map_location='cpu')
    observation=sample['traces_full'];h=homogeneous['traces_full'][sample['shot']]
    homogeneous_error=float((h-observation).norm()/observation.norm())
    report={'group_id':sample['group_id'],'shot':sample['shot'],'split':'train',
        'homogeneous_full_trace_relative_l2':homogeneous_error,
        'models':[{'label':label,'final_metrics':history[-1],'boundary':boundary,'receiver_spectrum':spectrum}
                  for label,history,boundary,spectrum in zip(['field_only','field_plus_receiver'],histories,boundaries,spectra)],
        'scope':'One supervised training shot, fixed 400 updates. No validation/test prediction or material-gradient comparison.'}
    (args.run/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    labels=['Field only','Field + receiver'];colors=['tab:blue','tab:orange']
    for label,color,history in zip(labels,colors,histories):
        ax[0,0].plot([r['update'] for r in history],[r['field_relative_l2'] for r in history],label=label,color=color)
        ax[0,1].plot([r['update'] for r in history],[r['trace_relative_l2'] for r in history],label=label,color=color)
    ax[0,0].set(title='Training field error',xlabel='Update',ylabel='Relative L2');ax[0,0].legend()
    ax[0,1].axhline(homogeneous_error,color='k',linestyle='--',label='Homogeneous reference')
    ax[0,1].set(title='Training receiver error',xlabel='Update',ylabel='Relative L2');ax[0,1].legend()
    t=np.arange(1,2001)*.03
    ax[0,2].plot(t,observation[12].numpy(),color='k',label='Reference',linewidth=1)
    for label,color,example in zip(labels,colors,examples):ax[0,2].plot(t,example['prediction_traces'][12].numpy(),label=label,color=color,alpha=.8)
    ax[0,2].set(title='Training receiver 12',xlabel='Time (us)',ylabel='Scalar u');ax[0,2].legend(fontsize=8)
    residuals=[ex['prediction_field'][:,:,250]-sample['target'][0,:,:,250] for ex in examples]
    vmax=max(float(r.abs().max()) for r in residuals)
    for i,(r,label) in enumerate(zip(residuals,labels)):
        im=ax[1,i].imshow(r.T.numpy(),origin='lower',extent=(0,100,0,50),vmin=-vmax,vmax=vmax,cmap='RdBu_r')
        ax[1,i].set(title=f'{label}: field residual, 30.12 us',xlabel='x (mm)',ylabel='y (mm)')
    fig.colorbar(im,ax=list(ax[1,:2]),label='Normalized field error',shrink=.7)
    categories=['Whole field','Interior','Receiver-side rows','Receiver traces'];positions=np.arange(4)
    for i,(label,history,b) in enumerate(zip(labels,histories,boundaries)):
        values=[history[-1]['field_relative_l2'],b['regions']['interior_y_rows']['relative_l2'],b['regions']['last_two_y_rows']['relative_l2'],history[-1]['trace_relative_l2']]
        ax[1,2].bar(positions+(i-.5)*.35,values,width=.35,label=label)
    ax[1,2].set_xticks(positions,categories,rotation=20,ha='right');ax[1,2].set(title='Final errors: fitted training shot',ylabel='Relative L2');ax[1,2].legend(fontsize=8)
    fig.savefig(args.run/'receiver_loss_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
