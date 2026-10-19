"""Analyze fixed three-shot scattering relative to the fine Deepwave solution."""
import json,sys,math
from pathlib import Path
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from deepwave_compare import dump,sha

cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'));out=Path(sys.argv[2]);assert out.is_dir()
old=json.loads(Path(cfg['previous_config']).read_text(encoding='utf-8-sig'))
base=json.loads(Path(old['base_config']).read_text(encoding='utf-8-sig'));base.update(old)
ids=np.array(cfg['transmitters_one_based']);times=np.arange(cfg['output_nt'])*cfg['output_dt_s']
off=np.arange(base['array']['count'])[None,:]!=(ids[:,None]-1)
late=(times>=cfg['late_window_s'][0]-1e-15)&(times<cfg['late_window_s'][1])

def measure(a,b,channels,window):
 a=a[channels][:,window].astype(float);b=b[channels][:,window].astype(float)
 e=b-a;ac=a-a.mean(axis=-1,keepdims=True);bc=b-b.mean(axis=-1,keepdims=True)
 corr=(ac*bc).sum(-1)/np.sqrt((ac*ac).sum(-1)*(bc*bc).sum(-1))
 x=a.ravel()-a.mean();y=b.ravel()-b.mean()
 return dict(relative_l2=float(np.linalg.norm(e)/np.linalg.norm(a)),global_pearson=float(x@y/np.sqrt((x@x)*(y@y))),trace_pearson_mean=float(corr.mean()),trace_pearson_median=float(np.median(corr)),channels=int(len(a)),samples_per_trace=int(a.shape[-1]))

def mat_read(path):
 with h5py.File(path,'r') as f:
  tx=np.asarray(f['txids']).reshape(-1).astype(int);ix=[list(tx).index(i) for i in ids]
  a=np.asarray(f['fmc']).transpose(2,1,0)[ix].astype(float)
  assert np.allclose(np.asarray(f['t_output']).reshape(-1),times,rtol=0,atol=1e-15)
 return a

def selected_solver(path):
 m=json.loads(Path(path).read_text(encoding='utf-8-sig'))
 tx=np.atleast_1d(m['transmitters']).astype(int).tolist();secs=np.atleast_1d(m['solver_s'])
 return float(sum(secs[tx.index(int(i))] for i in ids if int(i) in tx))

runs=json.loads(Path(cfg['kwave_series'],'runs.json').read_text(encoding='utf-8-sig'));assert len(runs)==6
sources={};sc={};timings={};checks={};models={}
dwr=Path(cfg['deepwave_run']);dtim=json.loads((dwr/'timings.json').read_text())
with h5py.File(dwr/'adjudication_deepwave.h5','r') as f:
 assert np.array_equal(f['txids'][:],ids)
 ref=f['fine/scattering'][:]
 for tag,name in [('coarse','Deepwave coarse'),('pml20','Deepwave PML20')]:sc[name]=f[tag+'/scattering'][:]
 for tag in ['fine','coarse','pml20']:
  timings['Deepwave '+tag]={'defect_s':dtim[tag+'_defect']['propagation_copy_s'],'uniform_s':dtim[tag+'_uniform']['propagation_copy_s'],'pair_s':dtim[tag+'_defect']['propagation_copy_s']+dtim[tag+'_uniform']['propagation_copy_s'],'timing_definition':'three-shot GPU propagation and CPU copy; no process startup or file save'}
  models[tag]=f[tag+'/model'][:];sources[tag]=f[tag+'/source'][:]
 assert np.array_equal(models['fine'],np.repeat(np.repeat(models['coarse'],2,0),2,1))
 assert np.array_equal(models['pml20'][cfg['top_padding_cells']:],models['coarse'])
 checks['deepwave_model_registration_exact']=True
with h5py.File(cfg['coarse_data'],'r') as f:uniform=f['kwave_uniform'][ids-1][...,::4].astype(float)
sc['k-Wave dt5']=mat_read(base['reference'])-uniform
kwroot=Path(base['reference']).parent
kt=selected_solver(kwroot/'metrics.json')
ku=sum(selected_solver(Path(base[k]).parent/'metrics.json') for k in ['kwave_uniform_single','kwave_uniform_remaining'])
timings['k-Wave dt5']={'defect_s':kt,'uniform_s':ku,'pair_s':kt+ku,'timing_definition':'selected transmitter solver times reused from previous runs; excludes gather/save/startup'}
new={};kw_audit={}
for x in runs:
 path=Path(x['directory'])/'worker_1/fmc.mat';tag=x['variant'];medium=x['medium']
 new[tag,medium]=mat_read(path)
 with h5py.File(path,'r') as f:
  c=np.asarray(f['c']).T;wave=np.asarray(f['source_wavelet']).reshape(-1);t=np.asarray(f['t_internal']).reshape(-1)
  expected=models['pml20' if tag=='pml20' else 'coarse']
  assert np.array_equal(c,expected if medium=='defect' else np.full_like(expected,base['max_vel_m_s']))
  q=np.pi*base['source_frequency_hz']*(t-base['source_shift_s']);r=(1-2*q*q)*np.exp(-q*q)
  assert np.allclose(wave,r,rtol=1e-10,atol=1e-13)
  assert np.isfinite(new[tag,medium]).all()
 kw_audit[tag+'_'+medium]={'sha256':sha(path),'model_exact':True,'ricker_matches':True,'output_times_match':True}
 m=json.loads(Path(x['directory'],'worker_1/metrics.json').read_text())
 x['solver_total_s']=m['solver_total_s']
for tag,name in [('dt25','k-Wave dt2.5'),('dt125','k-Wave dt1.25'),('pml20','k-Wave PML20')]:
 sc[name]=new[tag,'defect']-new[tag,'uniform']
 pair=[x for x in runs if x['variant']==tag];ds=next(x['solver_total_s'] for x in pair if x['medium']=='defect');us=next(x['solver_total_s'] for x in pair if x['medium']=='uniform')
 timings[name]={'defect_s':ds,'uniform_s':us,'pair_s':ds+us,'pair_process_wall_s':sum(x['summary']['process_wall_s'] for x in pair),'timing_definition':'MATLAB solver times for three shots; excludes gather/save/startup; process wall also provided'}
checks['kwave']=kw_audit
order=['Deepwave coarse','k-Wave dt5','k-Wave dt2.5','k-Wave dt1.25','Deepwave PML20','k-Wave PML20']
results={}
for name in order:
 b=sc[name];assert b.shape==ref.shape==(3,64,1000)
 results[name]={}
 for mode,channels in [('all',np.ones_like(off)),('off_diagonal',off)]:
  results[name][mode]={'full':measure(ref,b,channels,np.ones(len(times),bool)),'late':measure(ref,b,channels,late)}
 results[name]['timing']=timings[name if name!='Deepwave PML20' else 'Deepwave pml20']
 results[name]['by_shot_off_diagonal']=[{'tx':int(tx),'relative_l2':float(np.linalg.norm((b-ref)[i,off[i]])/np.linalg.norm(ref[i,off[i]]))} for i,tx in enumerate(ids)]
# Fixed additional diagnostics: differences induced by PML movement, and consecutive k-Wave dt reductions.
pml={}
for baseline,variant in [('Deepwave coarse','Deepwave PML20'),('k-Wave dt5','k-Wave PML20')]:
 pml[variant]={'change_over_baseline_norm':float(np.linalg.norm((sc[variant]-sc[baseline])[off])/np.linalg.norm(sc[baseline][off])),'change_over_ref_norm':float(np.linalg.norm((sc[variant]-sc[baseline])[off])/np.linalg.norm(ref[off]))}
consecutive={}
for a,b in [('k-Wave dt5','k-Wave dt2.5'),('k-Wave dt2.5','k-Wave dt1.25')]:
 consecutive[a+' to '+b]=float(np.linalg.norm((sc[b]-sc[a])[off])/np.linalg.norm(ref[off]))
dwl=results['Deepwave coarse']['off_diagonal']['full']['relative_l2'];kwl=results['k-Wave dt5']['off_diagonal']['full']['relative_l2']
batches=math.ceil(base['array']['count']/cfg['batch_shots']);per=timings['Deepwave fine']['defect_s']*batches
budget={'method':'22 sequential batches at most 3 shots: measured three-shot defect time times ceil(64/3); one homogeneous reference reusable; no full-64 fine benchmark','estimated_defect_64_s':per,'estimated_defect_64_with_margin_s':per*(1+cfg['budget_margin']),'samples':{str(n):{'propagation_minutes':n*per/60,'with_margin_minutes':n*per*(1+cfg['budget_margin'])/60} for n in cfg['budget_samples']},'margin_fraction':cfg['budget_margin'],'uniform_64_once_s':timings['Deepwave fine']['uniform_s']*batches,'output_float32_bytes_per_sample':64*64*1000*4+200*200*4}
summary={'results':results,'reference_timing':timings['Deepwave fine'],'pml_effect':pml,'kwave_consecutive_changes':consecutive,'checks':checks,'decision':{'coarse_deepwave_offdiag_l2':dwl,'coarse_deepwave_threshold':cfg['decision']['coarse_deepwave_max_scattering_l2'],'coarse_deepwave_threshold_pass':dwl<=cfg['decision']['coarse_deepwave_max_scattering_l2'],'kwave_dt5_offdiag_l2':kwl,'deepwave_error_reduction_vs_kwave_dt5':1-dwl/kwl,'fine_budget_triggered':dwl>cfg['decision']['coarse_deepwave_max_scattering_l2']},'fine_budget':budget,'kwave_runs':runs}
dump(out/'adjudication_metrics.json',summary)
with h5py.File(out/'adjudication_scattering.h5','x') as f:
 f['reference']=ref;f['txids']=ids;f['times']=times
 for n,b in sc.items():f[n]=b
# Tables and scientific plots use frozen outputs with no fitting.
for mode in ['all','off_diagonal']:
 lines=['| 方案 | 全窗L2 | 全局相关 | 逐道相关均值 | 12–20 μs L2 | 晚期全局相关 | 三炮缺陷/均匀/合计 秒 |','|---|---:|---:|---:|---:|---:|---:|']
 for n in order:
  x=results[n][mode];z=results[n]['timing']
  lines.append(f"| {n} | {x['full']['relative_l2']:.6f} | {x['full']['global_pearson']:.6f} | {x['full']['trace_pearson_mean']:.6f} | {x['late']['relative_l2']:.6f} | {x['late']['global_pearson']:.6f} | {z['defect_s']:.3f}/{z['uniform_s']:.3f}/{z['pair_s']:.3f} |")
 (out/f'table_{mode}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,window in zip(axes,['full','late']):
 xx=np.arange(len(order));width=.36
 for j,mode in enumerate(['all','off_diagonal']):ax.bar(xx+(j-.5)*width,[results[n][mode][window]['relative_l2'] for n in order],width,label=mode)
 ax.set_xticks(xx,['DW coarse','KW 5 ns','KW 2.5 ns','KW 1.25 ns','DW PML20','KW PML20'],rotation=25,ha='right');ax.set_ylabel('Relative scattering L2');ax.set_title('0-20 us' if window=='full' else '12-20 us');ax.legend()
axes[0].axhline(.15,color='r',ls='--',label='decision threshold')
fig.tight_layout();fig.savefig(out/'adjudication_errors.png',dpi=cfg['figure_dpi']);plt.close(fig)
fig,axes=plt.subplots(len(cfg['plot_pairs_one_based']),1,figsize=(11,10))
for ax,(tx,rx) in zip(axes,cfg['plot_pairs_one_based']):
 i=list(ids).index(tx);ax.plot(times*1e6,ref[i,rx-1],color='k',lw=1.2,label='DW fine reference')
 for name in ['Deepwave coarse','k-Wave dt5','k-Wave dt1.25']:ax.plot(times*1e6,sc[name][i,rx-1],lw=.8,label=name)
 ax.set_ylabel(f'Tx {tx}, Rx {rx}');ax.set_xlabel('Time (us)')
axes[0].legend(ncol=2,fontsize=8);fig.suptitle('Defect minus uniform: fixed gain, no new alignment')
fig.tight_layout();fig.savefig(out/'adjudication_waveforms.png',dpi=cfg['figure_dpi']);plt.close(fig)
print(json.dumps({'decision':summary['decision'],'pml_effect':pml,'kwave_consecutive_changes':consecutive,'fine_budget':budget},indent=2))
print((out/'table_off_diagonal.md').read_text(encoding='utf-8'))
