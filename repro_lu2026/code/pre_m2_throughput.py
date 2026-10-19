"""Full 64-shot fine-grid throughput, gated by the finer-reference check."""
import json,sys,time,os,gc,copy
from pathlib import Path
import numpy as np,h5py,torch
from source_match import waves
from deepwave_compare import dump,sha,propagate,Monitor
cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'));run=Path(sys.argv[2])
gate=json.loads(Path(cfg['convergence_run'],'metrics.json').read_text());assert gate['passed'] and gate['metrics']['Deepwave 0.05 mm']['relative_l2']<=cfg['convergence_max_l2']
old=json.loads(Path(cfg['previous_config']).read_text(encoding='utf-8-sig'));q=json.loads(Path(old['base_config']).read_text(encoding='utf-8-sig'));q.update(old)
cal=json.loads(Path(cfg['frozen_calibration']).read_text())['forward_difference'];assert cal['lag_s']==0
assert sha(q['reference'])==q['reference_sha256']
dump(run/'config.json',cfg);dump(run/'provenance.json',{'code_sha256':sha(__file__),'config_sha256':sha(sys.argv[1]),'reference_sha256':sha(q['reference']),'gate_metrics_sha256':sha(str(Path(cfg['convergence_run'],'metrics.json')))})
sample_start=time.perf_counter()
with h5py.File(q['reference']) as f:label=f['label'][:].T
q.update(dx_m=cfg['fine_dx_m'],dt_s=cfg['fine_dt_s'],nt=cfg['fine_nt'],pml_width=[cfg['fine_pml_cells']]*4)
for k in ['depth_index','first_lateral_index','pitch_cells']:q['array'][k]=int(q['array'][k]*cfg['source_reference_dx_m']/q['dx_m'])
model=np.repeat(np.repeat(label,cfg['fine_label_repeat'],0),cfg['fine_label_repeat'],1);assert list(model.shape)==cfg['fine_shape']
t=np.arange(q['nt'])*q['dt_s'];a=np.pi*q['source_frequency_hz']*(t-q['source_shift_s']);r=(1-2*a*a)*np.exp(-a*a);wave=(waves(q,r)['forward_difference']*(cfg['source_reference_dx_m']/q['dx_m'])).astype(np.float32)
torch.manual_seed(q['seed']);np.random.seed(q['seed']);torch.cuda.reset_peak_memory_stats()
monitor=Monitor(q['memory_poll_s']);monitor.worker.start()
failures=[];gpu_seconds=0;copy_seconds=0

def attempt(batch):
 ac=q['array'];v=torch.tensor(model,dtype=torch.float32,device='cuda');loc=torch.tensor([[ac['depth_index'],ac['first_lateral_index']+j*ac['pitch_cells']] for j in range(ac['count'])],dtype=torch.long,device='cuda')
 output=np.empty((ac['count'],ac['count'],cfg['output_nt']),np.float32);gpu_s=0;cpu_s=0;stride=round(cfg['output_dt_s']/q['dt_s'])
 for first in range(0,ac['count'],batch):
  last=min(first+batch,ac['count']);n=last-first
  w=torch.tensor(wave,device='cuda')[None,None,:].repeat(n,1,1)
  src=loc[first:last,None,:].contiguous();recv=loc[None,:,:].repeat(n,1,1)
  torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
  y=propagate(q,v,w,src,recv);end.record();end.synchronize();gpu_s+=begin.elapsed_time(end)/1000
  tick=time.perf_counter();arr=y[...,::stride].contiguous().cpu().numpy();output[first:last]=(arr.astype(np.float64)*cal['gain']).astype(np.float32);cpu_s+=time.perf_counter()-tick
  del y,arr,w
 assert np.isfinite(output).all()
 return output,gpu_s,cpu_s

prop_start=time.perf_counter()
try:
 for batch in [cfg['throughput_initial_batch']]+cfg['throughput_fallback_batches']:
  tick=time.perf_counter()
  try:
   output,gpu_seconds,copy_seconds=attempt(batch);break
  except torch.cuda.OutOfMemoryError as exc:
   failures.append({'batch':batch,'seconds':time.perf_counter()-tick,'error':str(exc)});dump(run/'oom_attempts.json',failures);gc.collect();torch.cuda.empty_cache()
   if batch==cfg['throughput_fallback_batches'][-1]:raise
 prop_s=time.perf_counter()-prop_start
finally:
 monitor.stop.set();monitor.worker.join(timeout=6)
allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
dump(run/'propagation_checkpoint.json',{'gpu_propagation_event_s':gpu_seconds,'output_decimation_copy_calibration_s':copy_seconds,'propagation_prepare_copy_wall_s':prop_s,'allocated_peak_bytes':allocated,'reserved_peak_bytes':reserved,'batch_size':batch,'oom_attempts':failures})
save_start=time.perf_counter();path=run/'fine_64_fmc.h5'
with h5py.File(path,'x') as f:
 f['fmc']=output;f['label']=label.astype(np.float32);f['t_output']=np.arange(cfg['output_nt'])*cfg['output_dt_s'];f['source']=wave
 f.attrs['axis_order']='tx,rx,time';f.attrs['calibration_gain']=cal['gain'];f.attrs['calibration_lag_s']=cal['lag_s'];f.attrs['config_json']=json.dumps(cfg);f.attrs['source_formula']='-2*h_c/(c_s*h_f**2*dt_f)*diff(r)';f.attrs['internal_waveforms_saved']=False
with open(path,'r+b') as f:os.fsync(f.fileno())
save_s=time.perf_counter()-save_start;sample_s=time.perf_counter()-sample_start
with h5py.File(path) as f:assert f['fmc'].shape==(64,64,1000) and f['fmc'].dtype==np.float32;assert np.array_equal(f['fmc'][:],output)
with h5py.File(Path(cfg['deepwave_run'])/'adjudication_deepwave.h5') as f:previous=f['fine/defect'][:];ids=f['txids'][:]
err=float(np.linalg.norm(output[ids-1].astype(float)-previous)/np.linalg.norm(previous));assert err<1e-6
summary=dict(batch_size=batch,batches=int(np.ceil(64/batch)),oom_attempts=failures,gpu_propagation_event_s=gpu_seconds,output_decimation_copy_calibration_s=copy_seconds,propagation_prepare_copy_wall_s=prop_s,save_close_fsync_s=save_s,sample_load_prepare_propagate_save_s=sample_s,allocated_peak_bytes=allocated,reserved_peak_bytes=reserved,whole_card_peak_mib=max(x[1] for x in monitor.rows),gpu_samples=monitor.rows,file_bytes=path.stat().st_size,fmc_payload_bytes=output.nbytes,previous_three_shot_relative_l2=err,file_sha256=sha(path),budget={str(n):{'sample_total_minutes':n*sample_s/60,'with_20pct_margin_minutes':n*sample_s*(1+cfg['budget_margin'])/60,'disk_bytes':n*path.stat().st_size} for n in cfg['budget_samples']},timing_note='Sample total includes loading model, CUDA setup, propagation, extraction, HDF5 close and fsync; excludes imports, gate/hash preflight, post-save verification. Single measurement, no M2.')
dump(run/'metrics.json',summary);print(json.dumps(summary,indent=2))
