"""Audit completed M2 files and summarize raw amplitudes without normalization."""
import json,sys,time,collections,datetime
from pathlib import Path
import numpy as np,h5py
from deepwave_compare import sha
from m2_common import write_new_json,array_sha
cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'));out=Path(sys.argv[2]);out.mkdir(exist_ok=False);root=Path(cfg['dataset_dir']);manifest=json.loads(Path(cfg['manifest_path']).read_text(encoding='utf-8-sig'));identity=json.loads((root/'identity.json').read_text());assert (root/'generation_complete.json').exists()
started=time.perf_counter();records=[];quant={k:[] for k in cfg['split_counts']};errors=[]
for i,row in enumerate(manifest['samples']):
 rec=json.loads((root/'records'/(row['sample_id']+'.json')).read_text());path=Path(rec['file']);assert rec['sample']==row;assert sha(path)==rec['sha256'];assert path.stat().st_size==rec['file_bytes']
 with h5py.File(path) as f:
  assert f.attrs['manifest_sha256']==identity['manifest_sha256'] and f.attrs['config_sha256']==identity['config_sha256'];assert not f.attrs['normalized'];assert f['fmc'].shape==(64,64,cfg['output_count']);assert f['fmc'].dtype==np.float32
  assert int(f['letter_id'][()])==row['letter_id'];assert f['letter'][()].decode()==row['letter'];assert f['label'].id==f['c_200'].id
  assert json.loads(f.attrs['sample_json'])==row;assert f['c_200'].shape==(200,200) and f['c_200'].dtype==np.float32;assert array_sha(f['c_200'][:])==row['c_200_sha256'];assert np.array_equal(f['t_output'][:],np.arange(cfg['output_count'])*cfg['output_dt_s'])
  y=f['fmc'][:].ravel();assert np.isfinite(y).all()
  actual=dict(count=y.size,minimum=float(y.min()),maximum=float(y.max()),max_abs=float(np.max(abs(y))),sum=float(np.sum(y,dtype=np.float64)),sum_squares=float(np.sum(np.square(y,dtype=np.float64),dtype=np.float64)),rms=float(np.sqrt(np.mean(np.square(y,dtype=np.float64)))))
  assert actual==rec['stats'],('statistics mismatch',row['sample_id'])
  step=cfg['quantile_sample_stride'];rng=np.random.default_rng(cfg['seed']+i);ix=np.arange(0,y.size,step);ix=np.minimum(ix+rng.integers(0,step,len(ix)),y.size-1);quant[row['split']].append(y[ix].copy())
 records.append(rec)
 if (i+1)%50==0:print('AUDITED',i+1,flush=True)
uniform=json.loads((root/'records/uniform.json').read_text());assert sha(uniform['file'])==uniform['sha256']
with h5py.File(uniform['file']) as f:
 assert f['fmc'].shape==(64,64,cfg['output_count']) and f['fmc'].dtype==np.float32;assert np.isfinite(f['fmc'][:]).all();assert np.all(f['c_200'][:]==5800);assert f['label'].id==f['c_200'].id;assert int(f['letter_id'][()])==0 and f['letter'][()].decode()=='uniform';assert not f.attrs['normalized'];assert f.attrs['manifest_sha256']==identity['manifest_sha256'] and f.attrs['config_sha256']==identity['config_sha256']

def summarize(selected,values):
 stats=[x['stats'] for x in selected];count=sum(x['count'] for x in stats);total=sum(x['sum'] for x in stats);sq=sum(x['sum_squares'] for x in stats);mean=total/count
 return dict(count=count,minimum=min(x['minimum'] for x in stats),maximum=max(x['maximum'] for x in stats),max_abs=max(x['max_abs'] for x in stats),mean=mean,std_population=float(np.sqrt(max(0,sq/count-mean*mean))),rms=float(np.sqrt(sq/count)),sampled_quantiles={str(p):float(np.quantile(values,p)) for p in [.001,.01,.5,.99,.999]},sampled_abs_quantiles={str(p):float(np.quantile(abs(values),p)) for p in [.5,.9,.99,.999]},quantile_samples=len(values),quantile_method='deterministic stratified one sample per 1000-element block, independently seeded per sample; approximate, not full-data exact quantiles')
qflat={k:np.concatenate(v) for k,v in quant.items()};amplitudes={k:summarize([x for x in records if x['sample']['split']==k],qflat[k]) for k in cfg['split_counts']};amplitudes['all']=summarize(records,np.concatenate(list(qflat.values())))
dist={k:dict(sorted(collections.Counter(x['sample']['letter'] for x in records if x['sample']['split']==k).items())) for k in cfg['split_counts']};assert dist==manifest['letter_counts']
ledger=[{'sample_id':x['sample']['sample_id'],'split':x['sample']['split'],'letter':x['sample']['letter'],'file':str(Path(x['file']).relative_to(root)),'file_bytes':x['file_bytes'],'sha256':x['sha256'],'total_s':x['total_s'],'gpu_s':x['gpu_s'],'attempts':x.get('attempts')} for x in records]
logs=[]
for p in Path(cfg['root'],'logs').glob('m2_*.jsonl'):
 lines=p.read_text(encoding='utf-8').splitlines()
 if not lines or json.loads(lines[0]).get('config_sha256')!=identity['config_sha256']:continue
 for line in lines:
  event=json.loads(line)
  if event.get('event')=='error':errors.append(event)
  if event.get('event') in ['start','complete']:logs.append(event)
summary={'dataset_dir':str(root),'samples':len(records),'split_counts':dict(collections.Counter(x['sample']['split'] for x in records)),'letter_counts':dist,'amplitudes':amplitudes,'file_bytes_samples':sum(x['file_bytes'] for x in records),'file_bytes_uniform':uniform['file_bytes'],'all_h5_bytes':sum(x['file_bytes'] for x in records)+uniform['file_bytes'],'sum_sample_total_s':sum(x['total_s'] for x in records if x['total_s'] is not None),'sum_gpu_s':sum(x['gpu_s'] for x in records),'uniform_total_s':uniform['total_s'],'sample_time_min_s':min(x['total_s'] for x in records if x['total_s'] is not None),'sample_time_max_s':max(x['total_s'] for x in records if x['total_s'] is not None),'sample_time_mean_s':float(np.mean([x['total_s'] for x in records if x['total_s'] is not None])),'peak_allocated_bytes':max(x['allocated_peak_bytes'] for x in records),'sha256_audited_files':len(records)+1,'anomalies':errors,'run_events':logs,'pilot_checks':json.loads((root/'pilot_checks.json').read_text()),'generation_complete':json.loads((root/'generation_complete.json').read_text()),'audit_wall_s':time.perf_counter()-started,'manifest_sha256':identity['manifest_sha256'],'no_normalization':True,'uniform_reference':{'file':'samples/uniform.h5','sha256':uniform['sha256'],'stats':uniform['stats']}}
write_new_json(out/'m2_summary.json',summary);write_new_json(out/'m2_file_ledger.json',ledger);print(json.dumps({k:v for k,v in summary.items() if k not in ['letter_counts','run_events']},indent=2),flush=True)
