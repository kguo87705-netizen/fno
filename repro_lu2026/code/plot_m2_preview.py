"""Display-only M2 previews: raw FMC and defect-minus-uniform, no dataset edits."""
import json,sys
from pathlib import Path
import numpy as np,h5py,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'));root=Path(cfg['dataset_dir']);rows=json.loads(Path(cfg['manifest_path']).read_text())['samples'];tx=cfg['figure_tx_one_based']-1
with h5py.File(root/'samples/uniform.h5') as f:hom=f['fmc'][tx].astype(float)
for i in cfg['figure_samples']:
 row=rows[i]
 with h5py.File(root/'samples'/(row['sample_id']+'.h5')) as f:c=f['c_200'][:];raw=f['fmc'][tx].astype(float)
 scatter=raw-hom;fig,axes=plt.subplots(1,3,figsize=(13,4));im=axes[0].imshow(c,extent=[0,40,40,0],vmin=1500,vmax=5800,cmap='viridis');fig.colorbar(im,ax=axes[0],label='m/s');axes[0].set(title=f"{row['sample_id']} / {row['letter']}",xlabel='Lateral (mm)',ylabel='Depth (mm)')
 for ax,y,title in zip(axes[1:],[raw,scatter],['Raw full FMC','Defect minus uniform (display only)']):
  lim=float(abs(y).max());im=ax.imshow(y,extent=[0,(cfg['output_count']-1)*cfg['output_dt_s']*1e6,64,1],aspect='auto',vmin=-lim,vmax=lim,cmap='seismic');fig.colorbar(im,ax=ax,label='Pressure (fixed source units)');ax.set(title=title,xlabel='Time (us)',ylabel='Receiver')
 fig.suptitle(f"Tx {cfg['figure_tx_one_based']}; separate color limits, data not normalized");fig.tight_layout();fig.savefig(root/'figures'/(row['sample_id']+'_scattering.png'),dpi=cfg['figure_dpi']);plt.close(fig)
