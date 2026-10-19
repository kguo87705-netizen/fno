"""Manifest-only M2 generation with per-sample durable records and resume checks."""
import json,sys,time,os,uuid,gc,traceback,shutil,datetime
from pathlib import Path
import numpy as np,h5py,torch
from deepwave_compare import sha,propagate
from source_match import waves
from m2_common import load_raw,array_sha,model_from_raw,write_new_json

cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'));stage=sys.argv[2];root=Path(cfg['dataset_dir'])
identity=json.loads((root/'identity.json').read_text());assert sha(sys.argv[1])==identity['config_sha256'];assert sha(cfg['manifest_path'])==identity['manifest_sha256'];assert shutil.disk_usage(root).free>=cfg['minimum_free_bytes']
manifest=json.loads(Path(cfg['manifest_path']).read_text(encoding='utf-8-sig'));assert sha(root/'split_manifest.json')==identity['manifest_sha256']
for name,digest in manifest['raw_file_sha256'].items():assert sha(Path(cfg['raw_dir'])/name)==digest
confirm=json.loads(Path(cfg['confirmed_config']).read_text(encoding='utf-8-sig'));old=json.loads(Path(confirm['previous_config']).read_text(encoding='utf-8-sig'));q=json.loads(Path(old['base_config']).read_text(encoding='utf-8-sig'));q.update(old)
cal=json.loads(Path(confirm['frozen_calibration']).read_text())['forward_difference'];assert cal['lag_s']==0
model_cfg=json.loads(Path(cfg['model_config']).read_text(encoding='utf-8-sig'))['model']
raw={p:load_raw(cfg['raw_dir'],p) for p in ['train','test']}
with h5py.File(cfg['confirmed_sample']) as f:reference_max=float(np.max(np.abs(f['fmc'][:])));assert np.array_equal(model_from_raw(raw['train'][0][0],model_cfg),f['label'][:])
ratio=cfg['source_reference_dx_m']/cfg['dx_m'];q.update(dx_m=cfg['dx_m'],dt_s=cfg['dt_s'],nt=cfg['nt'],accuracy=cfg['accuracy'],pml_width=[cfg['pml_cells']]*4)
for k in ['depth_index','first_lateral_index','pitch_cells']:q['array'][k]=int(q['array'][k]*ratio)
a=np.pi*q['source_frequency_hz']*(np.arange(q['nt'])*q['dt_s']-q['source_shift_s']);r=(1-2*a*a)*np.exp(-a*a);wave=(waves(q,r)['forward_difference']*ratio).astype(np.float32)
stride=round(cfg['output_dt_s']/cfg['dt_s']);nt=cfg['output_count'];count=q['array']['count'];ac=q['array'];start=time.perf_counter()
log=Path(cfg['root'])/'logs'/('m2_'+stage+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.jsonl');log.parent.mkdir(exist_ok=True)
def event(kind,**kw):
 rec={'timestamp':datetime.datetime.now().isoformat(),'event':kind,**kw}
 with log.open('a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n');f.flush()
 print(json.dumps(rec),flush=True)
def atomic_json(path,rec):
 temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');write_new_json(temp,rec)
 with temp.open('r+b') as f:os.fsync(f.fileno())
 os.replace(temp,path)

def stats(y):
 x=y.reshape(-1);n=x.size
 return {'count':n,'minimum':float(x.min()),'maximum':float(x.max()),'max_abs':float(np.max(abs(x))),'sum':float(np.sum(x,dtype=np.float64)),'sum_squares':float(np.sum(np.square(x,dtype=np.float64),dtype=np.float64)),'rms':float(np.sqrt(np.mean(np.square(x,dtype=np.float64))))}

def validate(path,row,expected_model):
 with h5py.File(path) as f:
  assert f.attrs['config_sha256']==identity['config_sha256'] and f.attrs['manifest_sha256']==identity['manifest_sha256']
  assert json.loads(f.attrs['sample_json'])==row
  assert f['fmc'].shape==(count,count,nt) and f['fmc'].dtype==np.float32
  assert np.array_equal(f['c_200'][:],expected_model);assert f['label'].id==f['c_200'].id
  y=f['fmc'][:];assert np.isfinite(y).all();v=stats(y)
  assert cfg['amplitude_reference_ratio_bounds'][0]*reference_max<=v['max_abs']<=cfg['amplitude_reference_ratio_bounds'][1]*reference_max,('amplitude',v,reference_max)
  assert expected_model.min()>=model_cfg['c_defect_m_s'] and expected_model.max()<=model_cfg['c_background_m_s']
 return v

def compute(c):
 sim=np.repeat(np.repeat(c,cfg['label_repeat'],0),cfg['label_repeat'],1);assert list(sim.shape)==cfg['shape'];v=torch.tensor(sim,dtype=torch.float32,device='cuda')
 loc=torch.tensor([[ac['depth_index'],ac['first_lateral_index']+j*ac['pitch_cells']] for j in range(count)],dtype=torch.long,device='cuda');torch.cuda.reset_peak_memory_stats()
 yout=np.empty((count,count,nt),np.float32);gpu=0
 for first in range(0,count,cfg['batch_shots']):
  last=min(first+cfg['batch_shots'],count);n=last-first;w=torch.tensor(wave,device='cuda')[None,None,:].repeat(n,1,1)
  torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
  y=propagate(q,v,w,loc[first:last,None,:].contiguous(),loc[None,:,:].repeat(n,1,1));b.record();b.synchronize();gpu+=a.elapsed_time(b)/1000
  yout[first:last]=(y[...,::stride].contiguous().cpu().numpy().astype(np.float64)*cal['gain']).astype(np.float32);del y,w
 return yout,gpu,torch.cuda.max_memory_allocated()

def process(row,c):
 sid=row['sample_id'];path=root/'samples'/(sid+'.h5');record=root/'records'/(sid+'.json')
 if path.exists():
  s=validate(path,row,c);digest=sha(path)
  if record.exists():
   rec=json.loads(record.read_text());assert rec['sha256']==digest and rec['file_bytes']==path.stat().st_size
  else:
   with h5py.File(path) as f:gpu=float(f.attrs['gpu_propagation_s'])
   rec=dict(sample=row,file=str(path),sha256=digest,file_bytes=path.stat().st_size,stats=s,total_s=None,gpu_s=gpu,recovered_record=True);atomic_json(record,rec)
  return rec,True
 for attempt in range(1,cfg['max_consecutive_attempts']+1):
  tick=time.perf_counter()
  try:
   assert shutil.disk_usage(root).free>=cfg['minimum_free_bytes']
   y,gpu,peak=compute(c);s=stats(y);assert np.isfinite(y).all()
   assert cfg['amplitude_reference_ratio_bounds'][0]*reference_max<=s['max_abs']<=cfg['amplitude_reference_ratio_bounds'][1]*reference_max
   temp=path.with_name(sid+'.'+uuid.uuid4().hex+'.partial.h5')
   with h5py.File(temp,'x') as f:
    f['fmc']=y;f['c_200']=c.astype(np.float32);f['label']=f['c_200'];f['letter_id']=row['letter_id'];f['letter']=row['letter'];f['t_output']=np.arange(nt)*cfg['output_dt_s']
    f.attrs['sample_json']=json.dumps(row);f.attrs['config_sha256']=identity['config_sha256'];f.attrs['manifest_sha256']=identity['manifest_sha256'];f.attrs['calibration_gain']=cal['gain'];f.attrs['calibration_lag_s']=0.;f.attrs['normalized']=False;f.attrs['axis_order']='transmitter,receiver,time';f.attrs['gpu_propagation_s']=gpu
   with temp.open('r+b') as f:os.fsync(f.fileno())
   stats_checked=validate(temp,row,c);digest=sha(temp);assert not path.exists();temp.rename(path)
   rec=dict(sample=row,file=str(path),sha256=digest,file_bytes=path.stat().st_size,stats=stats_checked,total_s=time.perf_counter()-tick,gpu_s=gpu,allocated_peak_bytes=peak,attempts=attempt)
   atomic_json(record,rec);del y;return rec,False
  except Exception as exc:
   event('error',sample_id=sid,attempt=attempt,elapsed_s=time.perf_counter()-tick,error=repr(exc),traceback=traceback.format_exc())
   gc.collect();torch.cuda.empty_cache()
   if attempt==cfg['max_consecutive_attempts']:raise

def reciprocity(path):
 with h5py.File(path) as f:y=f['fmc'][:].astype(np.float64)
 return float(np.linalg.norm(y-y.transpose(1,0,2))/np.linalg.norm(y))

def plots(rows):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for index in cfg['figure_samples']:
  row=rows[index]
  with h5py.File(root/'samples'/(row['sample_id']+'.h5')) as f:c=f['c_200'][:];y=f['fmc'][cfg['figure_tx_one_based']-1]
  fig,axes=plt.subplots(1,2,figsize=(10,4));im=axes[0].imshow(c,extent=[0,40,40,0],vmin=1500,vmax=5800,cmap='viridis');fig.colorbar(im,ax=axes[0],label='m/s');axes[0].set(xlabel='Lateral (mm)',ylabel='Depth (mm)',title=f"{row['sample_id']} / {row['letter']}")
  limit=float(np.max(abs(y)));im=axes[1].imshow(y,extent=[0,19.98,64,1],aspect='auto',cmap='seismic',vmin=-limit,vmax=limit);fig.colorbar(im,ax=axes[1]);axes[1].set(xlabel='Time (us)',ylabel='Receiver',title=f"Raw FMC, Tx {cfg['figure_tx_one_based']}");fig.tight_layout();fig.savefig(root/'figures'/(row['sample_id']+'.png'),dpi=cfg['figure_dpi']);plt.close(fig)

torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);event('start',stage=stage,config_sha256=identity['config_sha256'],manifest_sha256=identity['manifest_sha256'],free_bytes=shutil.disk_usage(root).free)
if stage=='bulk':assert (root/'pilot_reviewed.json').exists() and json.loads((root/'pilot_reviewed.json').read_text())['passed']
uniform_row={'sample_id':'uniform','split':'reference','emnist_source':None,'emnist_index_zero_based':None,'letter_id':0,'letter':'uniform'}
rec,skipped=process(uniform_row,np.full(model_cfg['label_shape'],model_cfg['c_background_m_s'],np.float32));event('uniform',skipped=skipped,total_s=rec['total_s'],sha256=rec['sha256'])
rows=manifest['samples'];selected=rows[:cfg['pilot_count']] if stage=='pilot' else rows
for i,row in enumerate(selected):
 raw_image=raw[row['emnist_source']][0][row['emnist_index_zero_based']];assert array_sha(raw_image)==row['raw_image_sha256'];c=model_from_raw(raw_image,model_cfg);assert array_sha(c)==row['c_200_sha256']
 rec,skipped=process(row,c);event('sample',completed=i+1,total=len(selected),sample_id=row['sample_id'],letter=row['letter'],split=row['split'],skipped=skipped,total_s=rec['total_s'],file_bytes=rec['file_bytes'],sha256=rec['sha256'],max_abs=rec['stats']['max_abs'])
 atomic_json(root/'status.json',{'stage':stage,'completed':i+1,'total':len(selected),'elapsed_session_s':time.perf_counter()-start,'last_sample':row['sample_id'],'timestamp':datetime.datetime.now().isoformat()})
if stage=='pilot':
 err=reciprocity(root/'samples'/(rows[0]['sample_id']+'.h5'));assert err<=cfg['reciprocity_max_relative_l2'];plots(rows)
 atomic_json(root/'pilot_checks.json',{'passed':True,'count':cfg['pilot_count'],'finite_shape_amplitude_all_passed':True,'reciprocity_sample':rows[0]['sample_id'],'reciprocity_relative_l2':err,'threshold':cfg['reciprocity_max_relative_l2'],'elapsed_session_s':time.perf_counter()-start,'source_reference_max_abs':reference_max})
else:atomic_json(root/'generation_complete.json',{'samples':len(rows),'elapsed_session_s':time.perf_counter()-start,'finished':datetime.datetime.now().isoformat(),'manifest_sha256':identity['manifest_sha256']})
event('complete',stage=stage,elapsed_s=time.perf_counter()-start)
