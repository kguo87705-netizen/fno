"""Full-grid and signed-waveform visualization for fitted training000 only."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.titlesize']=10

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    r=json.loads((a.run/'comparison.json').read_text());ex=torch.load(a.run/'training_example.pt',weights_only=True,map_location='cpu');fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    ref=ex['reference_residual_field'][:,:,250];pred=ex['prediction_residual_field'][:,:,250];vmax=float(ref.abs().max())
    for col,(name,v) in enumerate([('Reference scatter',ref),('DeepONet scatter',pred),('Prediction minus reference',pred-ref)]):
        bound=vmax if col<2 else float(v.abs().max());im=axes[0,col].imshow(v.T.numpy(),origin='lower',extent=(0,100,0,50),cmap='RdBu_r',vmin=-bound,vmax=bound);axes[0,col].set(title=f'{name}\nTrain000/source0, t=30.12 us',xlabel='x (mm)',ylabel='y (mm)');fig.colorbar(im,ax=axes[0,col],shrink=.7,label='Normalized scalar scatter u')
    t=np.arange(1,2001)*.03
    for label,key,color in [('Reference','reference_scatter_traces','k'),('DeepONet','prediction_scatter_traces','tab:orange')]:axes[1,0].plot(t,ex[key][12].numpy(),label=label,color=color,linewidth=1)
    axes[1,0].set(title='Train000/source0 receiver12: signed scatter',xlabel='Time (us)',ylabel='Scalar scatter u');axes[1,0].legend()
    keys=['opposite_boundary','interior','array_side'];axes[1,1].bar(np.arange(3),[r['final_training']['regions'][k]['scatter_field_relative_l2'] for k in keys]);axes[1,1].axhline(1,color='k',linestyle='--');axes[1,1].set_xticks(np.arange(3),['Opposite2 rows','Interior','Array-side2 rows'],rotation=20,ha='right');axes[1,1].set(title='Final regional scatter field error',ylabel='Relative L2')
    for i,(label,k) in enumerate([('Field','scatter_field_relative_l2'),('Receivers','scatter_trace_relative_l2')]):axes[1,2].bar(np.arange(2)+(i-.5)*.35,[h['metrics'][k] for h in r['history']],width=.35,label=label)
    axes[1,2].axhline(1,color='k',linestyle='--',label='Zero scattering');axes[1,2].set_xticks(np.arange(2),['Initial','Fixed final400']);axes[1,2].set(title='Only initial/final were evaluated',ylabel='Relative L2');axes[1,2].legend()
    fig.savefig(a.run/'deeponet_single_training.png',dpi=160);plt.close(fig);print('DeepONet figure saved, training-only.')

if __name__=='__main__':main()
