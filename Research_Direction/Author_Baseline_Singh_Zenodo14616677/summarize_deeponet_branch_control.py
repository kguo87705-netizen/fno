"""Branch-freeze paired results and independent temporal-basis oracle figure."""
from pathlib import Path
import json,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    torch.set_num_threads(4);root=Path('.build/basic-deeponet-branch-control-20261008');modes=['joint','frozen_initial_branch'];r={m:json.loads((root/m/'comparison.json').read_text()) for m in modes}
    assert r[modes[0]]['protocol']['initial_state_hash']==r[modes[1]]['protocol']['initial_state_hash'];assert r[modes[0]]['sampled_indices_sha256']==r[modes[1]]['sampled_indices_sha256']
    e={m:torch.load(root/m/'training_example.pt',weights_only=True,map_location='cpu') for m in modes};basisroot=Path('.build/receiver-temporal-basis-20261008');basis=json.loads((basisroot/'summary.json').read_text());wave=torch.load(basisroot/'waveforms.pt',weights_only=True,map_location='cpu')
    summary={'scope':'Training000/source0 only, paired branch optimization; basis projection is an in-sample independent-receiver oracle, no prediction or network capacity bound.','runs':r,'temporal_basis':basis};(root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plt.rcParams['axes.titlesize']=10;fig,ax=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for col,key,title in [(0,'scatter_field_relative_l2','Scatter field errors'),(1,'scatter_trace_relative_l2','Scatter receiver errors')]:
        for i,m in enumerate(modes):ax[0,col].bar(np.arange(3)+(i-.5)*.35,[h['metrics'][key] for h in r[m]['history']],width=.35,label=m)
        ax[0,col].axhline(1,color='k',linestyle='--',label='Zero scatter');ax[0,col].set_xticks(np.arange(3),['Initial','400','Fixed4000']);ax[0,col].set(title=title,ylabel='Relative L2');ax[0,col].legend(fontsize=7)
    t=np.arange(1,2001)*.03;ax[0,2].plot(t,e[modes[0]]['reference_scatter_traces'][12],color='k',label='Reference',linewidth=1)
    for m in modes:ax[0,2].plot(t,e[m]['prediction_scatter_traces'][12],label=m,linewidth=1)
    ax[0,2].set(title='Train000/source0 receiver12: fixed4000',xlabel='Time (us)',ylabel='Signed scalar scatter u');ax[0,2].legend(fontsize=7)
    keys=['opposite_boundary','interior','array_side']
    for i,m in enumerate(modes):ax[1,0].bar(np.arange(3)+(i-.5)*.35,[r[m]['final_training']['regions'][k]['scatter_field_relative_l2'] for k in keys],width=.35,label=m)
    ax[1,0].axhline(1,color='k',linestyle='--');ax[1,0].set_xticks(np.arange(3),['Opposite2 rows','Interior','Array-side2 rows'],rotation=20,ha='right');ax[1,0].set(title='Final regional field errors',ylabel='Relative L2');ax[1,0].legend(fontsize=7)
    names=list(basis['bases']);ax[1,1].bar(np.arange(3),[basis['bases'][n]['full2000_relative_l2'] for n in names],color=['tab:gray','tab:green','tab:purple']);ax[1,1].axhline(1,color='k',linestyle='--');ax[1,1].set_xticks(np.arange(3),['Carrier4','Local carrier34','Fourier127'],rotation=20,ha='right');ax[1,1].set(title='In-sample per-receiver linear basis oracle',ylabel='Full trace relative L2')
    ax[1,2].plot(t,wave['reference_full'][12],color='k',label='Reference',linewidth=1)
    for n in names:ax[1,2].plot(t,wave['predictions_full'][n][12],label=n,linewidth=1,alpha=.8)
    ax[1,2].set(title='Receiver12: basis fitted to its full target',xlabel='Time (us)',ylabel='Signed scalar scatter u');ax[1,2].legend(fontsize=6)
    fig.savefig(root/'branch_freeze_and_temporal_basis.png',dpi=160);plt.close(fig)
    print(json.dumps({m:{k:v for k,v in r[m]['final_training'].items() if k not in ['records','energy_sums','regions']} for m in modes}),flush=True)

if __name__=='__main__':main()
