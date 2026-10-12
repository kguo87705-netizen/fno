"""Matched feature arms at400 and fixed4000: full-grid and waveform evidence."""
import json
from pathlib import Path
import torch,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    torch.set_num_threads(4);short=Path('.build/basic-deeponet-time-control-20261008');long=Path('.build/basic-deeponet-budget-control-20261008');runs={(budget,mode):json.loads(((short if budget==400 else long)/mode/'comparison.json').read_text()) for budget in [400,4000] for mode in ['zero','sincos']}
    for budget in [400,4000]:
        a,b=[runs[budget,m] for m in ['zero','sincos']];assert a['protocol']['initial_state_hash']==b['protocol']['initial_state_hash'];assert a['sampled_indices_sha256']==b['sampled_indices_sha256']
    summary={'scope':'Training000/source0 only; matched seven-channel feature arms and continued budget. No independent validation/test or physical-gradient/FWI claims.','runs':{f'{b}_{m}':{'final_training':r['final_training'],'material_response':r['material_response_training_only'],'sampled_indices_sha256':r['sampled_indices_sha256']} for (b,m),r in runs.items()}}
    (long/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');fig,ax=plt.subplots(2,3,figsize=(16,8),layout='constrained');plt.rcParams['axes.titlesize']=10
    for col,key,title in [(0,'scatter_field_relative_l2','Scatter field relative L2'),(1,'scatter_trace_relative_l2','Scatter receiver relative L2')]:
        for i,mode in enumerate(['zero','sincos']):ax[0,col].bar(np.arange(2)+(i-.5)*.35,[runs[b,mode]['final_training'][key] for b in [400,4000]],width=.35,label=mode)
        ax[0,col].axhline(1,color='k',linestyle='--',label='Zero scatter');ax[0,col].set_xticks(np.arange(2),['400 updates','4000 updates']);ax[0,col].set(title=title,ylabel='Relative L2');ax[0,col].legend(fontsize=8)
    examples={m:torch.load(long/m/'training_example.pt',weights_only=True,map_location='cpu') for m in ['zero','sincos']};t=np.arange(1,2001)*.03
    ax[0,2].plot(t,examples['zero']['reference_scatter_traces'][12],color='k',label='Reference',linewidth=1)
    for m in ['zero','sincos']:ax[0,2].plot(t,examples[m]['prediction_scatter_traces'][12],label=m,linewidth=1)
    ax[0,2].set(title='Train000/source0, receiver12: fixed4000',xlabel='Time (us)',ylabel='Signed scalar scatter u');ax[0,2].legend(fontsize=8)
    ref=examples['sincos']['reference_residual_field'][:,:,250];pred=examples['sincos']['prediction_residual_field'][:,:,250];bound=float(ref.abs().max())
    for col,(name,values) in enumerate([('Reference',ref),('sincos final4000',pred)]):
        im=ax[1,col].imshow(values.T,origin='lower',extent=(0,100,0,50),cmap='RdBu_r',vmin=-bound,vmax=bound);ax[1,col].set(title=f'{name}: scatter at30.12us',xlabel='x (mm)',ylabel='y (mm)');fig.colorbar(im,ax=ax[1,col],shrink=.7,label='Normalized scalar scatter u')
    keys=['opposite_boundary','interior','array_side']
    for i,m in enumerate(['zero','sincos']):ax[1,2].bar(np.arange(3)+(i-.5)*.35,[runs[4000,m]['final_training']['regions'][k]['scatter_field_relative_l2'] for k in keys],width=.35,label=m)
    ax[1,2].axhline(1,color='k',linestyle='--');ax[1,2].set_xticks(np.arange(3),['Opposite2 rows','Interior','Array-side2 rows'],rotation=20,ha='right');ax[1,2].set(title='Fixed4000 regional field errors',ylabel='Relative L2');ax[1,2].legend(fontsize=8)
    fig.savefig(long/'time_features_budget_control.png',dpi=160);plt.close(fig)
    print(json.dumps({f'{b}_{m}':{k:v for k,v in r['final_training'].items() if k not in ['records','energy_sums','regions']} for (b,m),r in runs.items()}),flush=True)

if __name__=='__main__':main()
