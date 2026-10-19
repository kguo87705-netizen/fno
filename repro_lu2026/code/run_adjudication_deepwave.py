"""Three-shot adjudication: fixed source, original model, no optimization."""
import json,sys,copy,time
from pathlib import Path
import numpy as np
import h5py,torch
from deepwave_compare import dump,sha
from source_match import waves,simulate

cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'));run=Path(sys.argv[2])
gate=json.loads(Path(cfg['gate_accepted_run'],'gate_result.json').read_text())
assert cfg['gate_exclude_diagonal'] and gate['direct_excluding_self']['relative_l2_fine_minus_coarse_over_coarse']<cfg['direct_gate_l2_strict_upper']
old=json.loads(Path(cfg['previous_config']).read_text(encoding='utf-8-sig'))
base=json.loads(Path(old['base_config']).read_text(encoding='utf-8-sig'));base.update(old)
cal=json.loads(Path(cfg['frozen_calibration']).read_text())['forward_difference'];assert cal['lag_s']==0
assert sha(cfg['coarse_data'])==cfg['coarse_data_sha256']
assert sha(base['reference'])==base['reference_sha256']
dump(run/'config.json',cfg);dump(run/'provenance.json',{'script_sha256':sha(__file__),'config_sha256':sha(sys.argv[1]),'gate_sha256':sha(str(Path(cfg['gate_accepted_run'],'gate_result.json'))),'reference_sha256':sha(base['reference'])})
with h5py.File(base['reference'],'r') as f:label=f['label'][:].T;c=f['c'][:].T;r_old=f['source_wavelet'][:].reshape(-1)
assert np.array_equal(c,np.repeat(np.repeat(label,2,0),2,1))
ids=cfg['transmitters_one_based'];timings={};audits={}
torch.manual_seed(base['seed']);np.random.seed(base['seed'])
with h5py.File(run/'adjudication_deepwave.h5','x') as out:
 out['txids']=ids;out['t_output']=np.arange(cfg['output_nt'])*cfg['output_dt_s']
 for tag in ['fine','coarse','pml20']:
  q=copy.deepcopy(base);q.update(batch_shots=cfg['batch_shots'],oom_fallback_shots=cfg['oom_fallback_shots'])
  if tag=='fine':
   q.update(dx_m=cfg['fine_dx_m'],dt_s=cfg['fine_dt_s'],nt=cfg['fine_nt'],pml_width=[cfg['fine_pml_cells']]*4)
   model=np.repeat(np.repeat(label,cfg['fine_label_repeat'],0),cfg['fine_label_repeat'],1)
   ratio=base['dx_m']/q['dx_m']
   for k in ['depth_index','first_lateral_index','pitch_cells']:q['array'][k]=int(q['array'][k]*ratio)
  else:
   model=c.copy();ratio=1
   if tag=='pml20':
    pad=cfg['top_padding_cells'];model=np.pad(model,((pad,0),(0,0)),constant_values=q['max_vel_m_s']);q['array']['depth_index']+=pad
    assert q['array']['depth_index']==cfg['pml_variant_array_depth_cells']
  t=np.arange(q['nt'])*q['dt_s'];a=np.pi*q['source_frequency_hz']*(t-q['source_shift_s']);r=(1-2*a*a)*np.exp(-a*a)
  assert np.allclose(r[::int(ratio)],r_old,rtol=1e-11,atol=1e-13)
  wave=(waves(q,r)['forward_difference']*(cfg['source_reference_dx_m']/q['dx_m'])).astype(np.float32)
  out[tag+'/source']=wave;out[tag+'/model']=model
  dump(run/(tag+'_config.json'),q)
  stride=round(cfg['output_dt_s']/q['dt_s'])
  for medium in ['uniform','defect']:
   v=np.full_like(model,q['max_vel_m_s']) if medium=='uniform' else model
   data,detail=simulate(q,v,wave,ids);timings[tag+'_'+medium]=detail
   out[tag+'/'+medium+'_internal']=data
   out[tag+'/'+medium]=data[...,::stride].astype(np.float64)*cal['gain']
   if tag=='coarse':
    with h5py.File(cfg['coarse_data'],'r') as f:old_data=f['forward_difference/'+medium][np.array(ids)-1]
    audits[medium+'_coarse_max_abs_difference']=float(abs(data-old_data).max())
    audits[medium+'_coarse_relative_l2']=float(np.linalg.norm((data-old_data).astype(np.float64))/np.linalg.norm(old_data.astype(np.float64)))
   if tag=='fine' and medium=='uniform':
    with h5py.File(Path(cfg['gate_accepted_run'])/'uniform_refinement_gate.h5','r') as f:prev=f['fine_internal'][0]
    audits['fine_uniform_gate_relative_l2']=float(np.linalg.norm((data[ids.index(cfg['gate_transmitter_one_based'])]-prev).astype(np.float64))/np.linalg.norm(prev.astype(np.float64)))
   dump(run/'timings.json',timings);dump(run/'audits.json',audits)
   print(tag,medium,json.dumps(detail),flush=True)
  out[tag+'/scattering']=out[tag+'/defect'][:]-out[tag+'/uniform'][:]
  out.flush()
dump(run/'success.json',{'success':True,'calibration':cal,'gate_revision':'off_diagonal accepted by user','no_fitting':True})
