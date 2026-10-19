"""Read-only residual analysis; no alignment, gain fitting, or propagation."""
import json,sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
run=Path(sys.argv[2])
with h5py.File(run/'uniform_refinement_gate.h5','r') as f:
 a=f['coarse_output'][:];b=f['fine_output'][:];mask=f['direct_mask'][:].astype(bool)
old=json.loads(Path(cfg['previous_config']).read_text(encoding='utf-8-sig'))
base=json.loads(Path(old['base_config']).read_text(encoding='utf-8-sig'));base.update(old)
tx=cfg['gate_transmitter_one_based']-1
pitch=base['array']['pitch_cells']*base['dx_m']
d=np.abs(np.arange(len(a))-tx)*pitch
err=b-a
se=np.sum((err*mask)**2,axis=1);re=np.sum((a*mask)**2,axis=1)
per=[dict(receiver_one_based=i+1,offset_mm=float(d[i]*1e3),direct_l2=float(np.sqrt(se[i]/re[i])),residual_energy_share=float(se[i]/se.sum())) for i in range(len(a))]
groups=[('self',np.arange(len(a))==tx)]
edges=cfg['offset_band_edges_m']
for lo,hi in zip(edges[:-1],edges[1:]):
 groups.append((f'({lo*1e3:g},{hi*1e3:g}] mm',(d>lo+1e-12)&(d<=hi+1e-12)))
off=[]
for name,m in groups:
 off.append(dict(group=name,receivers=int(m.sum()),direct_l2=float(np.sqrt(se[m].sum()/re[m].sum())),residual_energy_share=float(se[m].sum()/se.sum()),coarse_energy_share=float(re[m].sum()/re.sum())))
# Full-record FFT avoids truncating the pulse with different per-receiver windows.
# Rectangular full-record window, no taper or filtering; one-sided Parseval weights.
fa=np.fft.rfft(a,axis=-1);fe=np.fft.rfft(err,axis=-1)
w=np.ones(fa.shape[-1])*2;w[0]=1;w[-1]=1
freq=np.fft.rfftfreq(a.shape[-1],cfg['output_dt_s'])
bands={}
for name,m in [('all',np.ones(len(a),bool)),('nonself',np.arange(len(a))!=tx),('self',np.arange(len(a))==tx)]:
 er=np.sum(abs(fe[m])**2,axis=0)*w;ref=np.sum(abs(fa[m])**2,axis=0)*w
 rows=[]
 fedges=cfg['frequency_band_edges_hz']
 for j,(lo,hi) in enumerate(zip(fedges[:-1],fedges[1:])):
  k=(freq>=lo)&((freq<=hi) if j==len(fedges)-2 else (freq<hi))
  rows.append(dict(low_mhz=lo/1e6,high_mhz=hi/1e6,l2=float(np.sqrt(er[k].sum()/ref[k].sum())),residual_energy_share=float(er[k].sum()/er.sum()),coarse_energy_share=float(ref[k].sum()/ref.sum())))
 assert np.isclose(er.sum()/a.shape[-1],np.sum(err[m]**2),rtol=1e-12)
 bands[name]=rows
out={'offset_direct':off,'receiver_direct':per,'frequency_full_record':bands,'frequency_definition':'Full 1000-point records, rectangular window; one-sided Parseval weights; no fitted gain/lag. Frequency analysis is diagnostic, not a replacement gate.'}
(run/'residual_distribution.json').write_text(json.dumps(out,indent=2)+'\n')
fig,ax=plt.subplots(2,2,figsize=(10,7))
ax[0,0].scatter(d*1e3,np.sqrt(se/re),s=20);ax[0,0].axhline(.03,color='r',linestyle='--');ax[0,0].set(xlabel='Absolute offset (mm)',ylabel='Direct-wave relative L2',title='All receivers, including self')
m=np.arange(len(a))!=tx
ax[0,1].scatter(d[m]*1e3,np.sqrt(se[m]/re[m]),s=20);ax[0,1].axhline(.03,color='r',linestyle='--');ax[0,1].set(xlabel='Absolute offset (mm)',ylabel='Direct-wave relative L2',title='Nonself receivers')
for name in ['all','nonself','self']:
 rows=bands[name];x=np.arange(len(rows))
 ax[1,0].plot(x,[r['l2'] for r in rows],'-o',label=name)
 ax[1,1].plot(x,[100*r['residual_energy_share'] for r in rows],'-o',label=name)
labels=[f"{r['low_mhz']:g}-{r['high_mhz']:g}" for r in bands['all']]
for q in ax[1]:q.set_xticks(x,labels);q.set_xlabel('Frequency band (MHz)');q.legend()
ax[1,0].set_ylabel('Band relative L2');ax[1,0].set_yscale('log')
ax[1,1].set_ylabel('Residual energy share within group (%)')
fig.tight_layout();fig.savefig(run/'residual_distribution.png',dpi=cfg['figure_dpi']);plt.close(fig)
print(json.dumps({'offset_direct':off,'frequency_full_record':bands},indent=2))
