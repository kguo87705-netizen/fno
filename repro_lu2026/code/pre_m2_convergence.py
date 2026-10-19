"""One-shot finer reference, frozen geometric source and calibration."""
import json,sys,copy,time
from pathlib import Path
import numpy as np,h5py,torch
from source_match import waves,simulate
from deepwave_compare import dump,sha
cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'));run=Path(sys.argv[2])
old=json.loads(Path(cfg['previous_config']).read_text(encoding='utf-8-sig'));base=json.loads(Path(old['base_config']).read_text(encoding='utf-8-sig'));base.update(old)
cal=json.loads(Path(cfg['frozen_calibration']).read_text())['forward_difference'];assert cal['lag_s']==0
assert sha(base['reference'])==base['reference_sha256']
dump(run/'config.json',cfg);dump(run/'provenance.json',{'code_sha256':sha(__file__),'config_sha256':sha(sys.argv[1]),'reference_sha256':sha(base['reference']),'previous_scattering_sha256':sha(cfg['previous_scattering'])})
with h5py.File(base['reference']) as f:label=f['label'][:].T;r_old=f['source_wavelet'][:].reshape(-1)
q=copy.deepcopy(base);q.update(dx_m=cfg['check_dx_m'],dt_s=cfg['check_dt_s'],nt=cfg['check_nt'],pml_width=[cfg['check_pml_cells']]*4,batch_shots=1,oom_fallback_shots=1)
ratio=base['dx_m']/q['dx_m']
for k in ['depth_index','first_lateral_index','pitch_cells']:q['array'][k]=int(q['array'][k]*ratio)
model=np.repeat(np.repeat(label,cfg['check_label_repeat'],0),cfg['check_label_repeat'],1);assert list(model.shape)==cfg['check_shape']
t=np.arange(q['nt'])*q['dt_s'];a=np.pi*q['source_frequency_hz']*(t-q['source_shift_s']);r=(1-2*a*a)*np.exp(-a*a)
assert np.allclose(r[::int(ratio)],r_old,rtol=1e-11,atol=1e-13)
w=(waves(q,r)['forward_difference']*(cfg['source_reference_dx_m']/q['dx_m'])).astype(np.float32)
stride=round(cfg['output_dt_s']/q['dt_s']);tx=cfg['check_transmitter_one_based'];timings={};data={}
torch.manual_seed(q['seed']);np.random.seed(q['seed']);dump(run/'solver_config.json',q)
try:
 for medium in ['uniform','defect']:
  v=np.full_like(model,q['max_vel_m_s']) if medium=='uniform' else model
  raw,detail=simulate(q,v,w,[tx]);timings[medium]=detail;data[medium]=raw[...,::stride].astype(float)*cal['gain']
  dump(run/'timings.json',timings);print(medium,json.dumps(detail),flush=True)
except torch.cuda.OutOfMemoryError as e:
 dump(run/'oom_stop.json',{'error':str(e),'stopped':True,'no_retry':True});raise
ref=data['defect'][0]-data['uniform'][0];mask=np.arange(64)!=tx-1
previous={}
with h5py.File(Path(cfg['deepwave_run'])/'adjudication_deepwave.h5') as f:
 ix=list(f['txids'][:]).index(tx)
 previous['Deepwave 0.05 mm']=f['fine/scattering'][ix];previous['Deepwave 0.1 mm']=f['coarse/scattering'][ix]
 assert np.array_equal(model,np.repeat(np.repeat(f['fine/model'][:],2,0),2,1))
with h5py.File(cfg['previous_scattering']) as f:previous['k-Wave 1.25 ns']=f['k-Wave dt1.25'][list(f['txids'][:]).index(tx)]
def metric(x,y):
 x=x[mask].astype(float);y=y[mask].astype(float);xx=x-x.mean();yy=y-y.mean();xc=x-x.mean(-1,keepdims=True);yc=y-y.mean(-1,keepdims=True)
 return dict(relative_l2=float(np.linalg.norm(y-x)/np.linalg.norm(x)),global_pearson=float(np.sum(xx*yy)/np.sqrt(np.sum(xx*xx)*np.sum(yy*yy))),mean_trace_pearson=float(np.mean(np.sum(xc*yc,-1)/np.sqrt(np.sum(xc*xc,-1)*np.sum(yc*yc,-1)))))
metrics={n:metric(ref,x) for n,x in previous.items()};passed=metrics['Deepwave 0.05 mm']['relative_l2']<=cfg['convergence_max_l2']
prior=json.loads(Path(cfg['previous_metrics']).read_text());kw_per=prior['results']['k-Wave dt1.25']['timing']['defect_s']/len(cfg['transmitters_one_based'])*64
per=timings['defect']['propagation_copy_s']*64
summary=dict(metrics=metrics,passed=passed,threshold=cfg['convergence_max_l2'],timings=timings,calibration=cal,channels=63,check_grid=q['dx_m'],no_fit=True,previous_data_reused=True,throughput_executed=False,finer_budget={'single_sample_64_sequential_s':per,'samples':{str(n):{'minutes':n*per/60,'with_margin_minutes':n*per*(1+cfg['budget_margin'])/60} for n in cfg['budget_samples']},'method':'single-shot defect propagation/copy times 64, no throughput batch or saving benchmark'},kwave_cross_solver_budget={'samples':cfg['cross_solver_samples'],'per_sample_solver_s':kw_per,'total_solver_hours':kw_per*cfg['cross_solver_samples']/3600,'with_margin_hours':kw_per*cfg['cross_solver_samples']*(1+cfg['budget_margin'])/3600,'method':'previous three-shot defect solver sum /3*64; excludes saving and startup'})
with h5py.File(run/'convergence.h5','x') as f:
 f['reference_scattering']=ref;f['uniform']=data['uniform'];f['defect']=data['defect'];f['source']=w;f['model']=model
 for n,x in previous.items():f[n]=x
dump(run/'metrics.json',summary)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,1,figsize=(11,8));times=np.arange(cfg['output_nt'])*cfg['output_dt_s']
for ax,rx in zip(axes,[1,40,64]):
 ax.plot(times*1e6,ref[rx-1],color='k',lw=1.2,label='Deepwave 0.025 mm')
 for n,x in previous.items():ax.plot(times*1e6,x[rx-1],lw=.8,label=n)
 ax.set(xlabel='Time (us)',ylabel=f'Tx {tx}, Rx {rx}')
axes[0].legend(fontsize=8,ncol=2);fig.tight_layout();fig.savefig(run/'convergence_waveforms.png',dpi=cfg['figure_dpi']);plt.close(fig)
print(json.dumps(summary,indent=2),flush=True)
