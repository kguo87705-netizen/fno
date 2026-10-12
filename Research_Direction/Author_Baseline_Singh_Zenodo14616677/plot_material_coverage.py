"""Fixed-color material coverage map of train/validation source0 workpieces."""
import argparse,json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from operator_data import OperatorPilotDataset

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    r=json.loads((a.out/'coverage.json').read_text());fig,axes=plt.subplots(3,6,figsize=(17,8),layout='constrained')
    for split,row0 in [('train',0),('validation',2)]:
        ds=OperatorPilotDataset(a.data,split)
        for i,k in enumerate(range(0,len(ds),4)):
            s=ds[k];ax=axes[row0+i//6,i%6];im=ax.imshow(s['gamma'].T.numpy(),origin='lower',extent=(0,100,0,50),cmap='viridis',vmin=.3,vmax=1,aspect='auto')
            title=f"{split} {s['group_id'][-3:]}"
            if split=='validation':title+=f" / nearest {r['records'][i]['nearest_training_group'][-3:]}"
            ax.set(title=title,xlabel='x (mm)',ylabel='y (mm)');ax.tick_params(labelsize=8)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Dimensionless density scaling gamma',shrink=.7)
    fig.savefig(a.out/'material_coverage.png',dpi=150);plt.close(fig);print('Saved material_coverage.png; old test not loaded.')

if __name__=='__main__':main()
