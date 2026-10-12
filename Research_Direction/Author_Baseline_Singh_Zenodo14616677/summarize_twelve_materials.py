"""Fixed-final twelve-material training and six-material source0 validation figure."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator_data import OperatorPilotDataset
from time_reconstruction import cubic_time_reconstruction


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    r=json.loads((a.run/'comparison.json').read_text());mr=json.loads((a.run/'material_response.json').read_text());summary={k:r[k] for k in ['status','protocol','references','final_training','final_validation']};summary['material_response']=mr
    (a.run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    ex=torch.load(a.run/'validation_example.pt',weights_only=True,map_location='cpu');mean=torch.load(a.run/'training_mean.pt',weights_only=True,map_location='cpu')['normalized_residual'];s=OperatorPilotDataset(a.data,'validation')[0];ids=s['receiver_indices']
    mean_trace=cubic_time_reconstruction(mean[ids[:,0],ids[:,1],:]*s['wavefield_scale'],s['time_seconds'],torch.arange(1,2001,dtype=torch.float64)*30e-9)
    fig,ax=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for col,key,title in [(0,'scatter_trace_relative_l2','Scatter receiver errors'),(1,'scatter_field_relative_l2','Scatter field errors')]:
        for split,color in [('training','tab:blue'),('validation','tab:orange')]:ax[0,col].plot([x['update'] for x in r['history']],[x[split][key] for x in r['history']],marker='o',label=split,color=color)
        ax[0,col].axhline(1,color='k',linestyle='--',label='Zero scattering');ax[0,col].axhline(r['references']['training_mean']['validation'][key],color='tab:green',linestyle=':',label='Frozen train mean (val)');ax[0,col].set(title=title,xlabel='Updates',ylabel='Relative L2');ax[0,col].legend(fontsize=8)
    val=r['final_validation'];mean_val=r['references']['training_mean']['validation'];positions=np.arange(6)
    for i,(label,records) in enumerate([('FNO',val['records']),('Frozen train mean',mean_val['records'])]):ax[0,2].bar(positions+(i-.5)*.35,[np.sqrt(z['trace_error']/z['scatter_trace_true']) for z in records],width=.35,label=label)
    ax[0,2].axhline(1,color='k',linestyle='--',label='Zero scattering');ax[0,2].set_xticks(positions,[z['group_id'][-3:] for z in val['records']]);ax[0,2].set(title='Six validation workpieces: scatter receivers',xlabel='Workpiece ID (source0)',ylabel='Relative L2');ax[0,2].legend(fontsize=8)
    t=np.arange(1,2001)*.03
    for label,v,color in [('Reference',ex['reference_scatter_traces'],'k'),('FNO',ex['prediction_scatter_traces'],'tab:orange'),('Frozen train mean',mean_trace,'tab:green')]:ax[1,0].plot(t,v[12].numpy(),label=label,color=color,linewidth=1,alpha=.85)
    ax[1,0].set(title='Validation004, source0, receiver12: scatter',xlabel='Time (us)',ylabel='Scalar scattering u');ax[1,0].legend(fontsize=8)
    error=ex['prediction_residual_field'][:,:,250]-ex['reference_residual_field'][:,:,250];vmax=float(error.abs().max());im=ax[1,1].imshow(error.T.numpy(),origin='lower',extent=(0,100,0,50),vmin=-vmax,vmax=vmax,cmap='RdBu_r')
    ax[1,1].set(title='Validation004 scatter error at30.12us',xlabel='x (mm)',ylabel='y (mm)');fig.colorbar(im,ax=ax[1,1],shrink=.7,label='Normalized error')
    keys=['opposite_boundary','interior','array_side'];positions=np.arange(3)
    for i,(label,z) in enumerate([('FNO',val),('Frozen train mean',mean_val)]):ax[1,2].bar(positions+(i-.5)*.35,[z['regions'][key]['scatter_field_relative_l2'] for key in keys],width=.35,label=label)
    ax[1,2].axhline(1,color='k',linestyle='--',label='Zero scattering');ax[1,2].set_xticks(positions,['Opposite 2 rows','Interior','Array-side 2 rows'],rotation=20,ha='right');ax[1,2].set(title='Validation regional scatter field errors',ylabel='Relative L2');ax[1,2].legend(fontsize=8)
    fig.savefig(a.run/'twelve_materials_validation.png',dpi=160);plt.close(fig)
    print(json.dumps({'training':{k:v for k,v in r['final_training'].items() if k not in ['records','regions','energy_sums']},'validation':{k:v for k,v in val.items() if k not in ['records','regions','energy_sums']},'material_response':mr['summary']}),flush=True)


if __name__=='__main__':main()
