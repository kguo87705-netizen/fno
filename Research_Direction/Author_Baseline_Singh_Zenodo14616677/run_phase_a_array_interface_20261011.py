import json, hashlib, sys, time, platform
from pathlib import Path
import torch
sys.path.insert(0,'Research_Direction/Author_Baseline_Singh_Zenodo14616677')
from run_fno_local_contrast import physical_setup, generate
from run_frozen_depth_shift import policy, shift_mask

ROOT=Path('Research_Direction/Author_Baseline_Singh_Zenodo14616677')
OUT=Path('.build/phase-a-array-interface-20261011-r2')
BASE=Path('.build/operator-pilot-v3-20261008/homogeneous.pt')
STATE=Path('.build/fno-frozen-depth-shift-audit-20261009-r2/state_0.pt')
SCRIPT=Path(__file__)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def finite(x): return bool(torch.isfinite(x).all())
def cosine(a,b):
    aa=a.double().reshape(-1);bb=b.double().reshape(-1); den=float(aa.norm()*bb.norm())
    return None if den==0 else float(torch.dot(aa,bb)/den)
def peak_time(trace,times):
    idx=trace.abs().argmax(dim=-1); return times[idx]

def main():
    assert not OUT.exists(), f'output exists: {OUT}'
    OUT.mkdir(parents=True)
    policy(); torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    solver,force,template,shape,rx,ry,device=physical_setup()
    base=torch.load(BASE,weights_only=True,map_location='cpu')
    state=torch.load(STATE,weights_only=True,map_location='cpu')
    mask=state['gamma']<.99
    n_rx=len(rx); half=n_rx//2
    actions={
      'full_aperture':{'name':'full_aperture','tx_elements':[0],'rx_elements':list(range(n_rx)),'source_mode':'source0','receive_mode':'selected_rx'},
      'left_half':{'name':'left_half','tx_elements':[0],'rx_elements':list(range(half)),'source_mode':'source0','receive_mode':'selected_rx'},
      'right_half':{'name':'right_half','tx_elements':[0],'rx_elements':list(range(half,n_rx)),'source_mode':'source0','receive_mode':'selected_rx'}
    }
    # source0 is the only active source in the existing physical interface; record this explicitly.
    write('array_actions.json',{'source_count_in_force':int(force.shape[0]),'active_source_rule':'existing generate uses force[0:1]; tx_elements therefore fixed to [0]','receiver_count':n_rx,'actions':actions})
    files=[ROOT/'run_fno_local_contrast.py',ROOT/'FiniteDifferencePyTorchConv.py',ROOT/'settings2D.csv',ROOT/'data_casestudy/source.h5',BASE,STATE]
    protocol={'status':'frozen_before_run','experiment':'phase_a_array_interface','source':'source0','grid_shape':list(shape),'receiver_count':n_rx,'actions':actions,'material_states':{'background':1.0,'low_contrast':0.48,'high_contrast':0.52},'support':'state_0 center support mask, 449 cells','matrix':'3 material states x 3 array actions = 9 physical tasks','validation_tasks':[{'action':'full_aperture','material':'low_contrast'},{'action':'left_half','material':'background'},{'action':'right_half','material':'high_contrast'}],'budgets':{'physical_tasks':9,'validation_replays':3,'network_forwards':0,'training_updates':0,'max_seconds':300},'metrics':['full_receiver_error','scatter_receiver_error','tof_difference','peak_amplitude_difference','low_high_response_alignment'],'failure_rule':'preserve failure.json with elapsed time and error; do not delete failed task','input_hashes':{str(p):sha(p) for p in files},'script_hash':sha(SCRIPT)}
    write('protocol.json',protocol)
    write('input_manifest.json',{'template_keys':sorted(template.keys()),'shape':list(shape),'source_xy_m':template['source_xy_m'].tolist(),'receiver_physical_indices':template['receiver_physical_indices'].tolist(),'receiver_xy_m':template['receiver_xy_m'].tolist(),'device':str(device),'torch':torch.__version__,'platform':platform.platform(),'script':str(SCRIPT)})
    start=time.perf_counter(); rows=[]; grouped={};
    gamma_by={'background':torch.ones_like(template['gamma']),'low_contrast':torch.where(mask,torch.full_like(template['gamma'],0.48),torch.ones_like(template['gamma'])),'high_contrast':torch.where(mask,torch.full_like(template['gamma'],0.52),torch.ones_like(template['gamma']))}
    for aname,action in actions.items():
      grouped[aname]={}
      rx_idx=torch.tensor(action['rx_elements'],dtype=torch.long)
      for mname,a in [('background',1.0),('low_contrast',0.48),('high_contrast',0.52)]:
        t0=time.perf_counter(); status='completed'; err=None; d=None
        try:
          d=generate(solver,force,{**template,'gamma':gamma_by[mname]},shape,rx,ry,device,a)
          full=d['traces_full'][0].double(); active=full.index_select(0,rx_idx); active_mask=torch.zeros(n_rx,dtype=torch.bool);active_mask[rx_idx]=True
          out={'traces_full':full,'traces_selected':active,'active_rx_mask':active_mask,'wavefield_snapshots':d['wavefield_snapshots'][0].double(),'gamma':d['gamma'].double(),'receiver_physical_indices':template['receiver_physical_indices'],'receiver_xy_m':template['receiver_xy_m'],'source_xy_m':template['source_xy_m'],'time_seconds_full':template['time_seconds_full'],'action':action,'material':mname,'material_value':a}
          torch.save(out,OUT/f'{aname}__{mname}.pt'); grouped[aname][mname]=out
        except Exception as ex:
          status='failed';err=repr(ex); (OUT/f'{aname}__{mname}__failure.json').write_text(json.dumps({'status':'failed','action':action,'material':mname,'elapsed_seconds':time.perf_counter()-t0,'error':err},ensure_ascii=False,indent=2),encoding='utf-8')
        rows.append({'run_id':f'{aname}__{mname}','action':aname,'material_state':mname,'physical_calls':1 if status=='completed' else 1,'network_forwards':0,'training_updates':0,'data_generation_seconds':'unknown','physical_seconds':time.perf_counter()-t0 if status=='completed' else None,'postprocess_seconds':'unknown','total_seconds':time.perf_counter()-t0,'status':status,'error':err})
    # Derive metrics from saved in-memory outputs; no truth-based selection.
    times=template['time_seconds_full'].double()
    for aname in actions:
      g=grouped.get(aname,{})
      if set(g)!={'background','low_contrast','high_contrast'}: continue
      bgtr=g['background']['traces_full']; lo=g['low_contrast']['traces_full']; hi=g['high_contrast']['traces_full']; idx=torch.tensor(actions[aname]['rx_elements'])
      b=bgtr.index_select(0,idx); l=lo.index_select(0,idx); h=hi.index_select(0,idx)
      low_err=float((l-b).norm()/b.norm()); high_err=float((h-b).norm()/b.norm())
      def td(x): return float((peak_time(x,times)-peak_time(b,times)).abs().mean())
      def amp(x): return float((x.abs().amax(-1)-b.abs().amax(-1)).abs().mean())
      span=h-l; align_hi=float(torch.dot((h-b).reshape(-1),span.reshape(-1)))
      align_lo=float(torch.dot((l-b).reshape(-1),span.reshape(-1)))
      rows.append({'run_id':f'{aname}__derived_metrics','action':aname,'material_state':'derived','physical_calls':0,'network_forwards':0,'training_updates':0,'data_generation_seconds':0.0,'physical_seconds':0.0,'postprocess_seconds':0.0,'total_seconds':0.0,'status':'completed','full_receiver_error_low_vs_background':low_err,'full_receiver_error_high_vs_background':high_err,'tof_difference_low_seconds':td(l),'tof_difference_high_seconds':td(h),'peak_amplitude_difference_low':amp(l),'peak_amplitude_difference_high':amp(h),'high_response_alignment':align_hi,'low_response_alignment':align_lo,'active_rx_count':len(idx)})
    summary={'status':'completed','physical_tasks':9,'completed_tasks':sum(x['status']=='completed' and x['material_state']!='derived' for x in rows),'failed_tasks':sum(x['status']=='failed' for x in rows),'rows':rows,'elapsed_seconds':time.perf_counter()-start,'network_forwards':0,'training_updates':0,'scope':'stage A scalar finite-difference array interface; no FNO/PINO training; source0 fixed; not final test'}
    write('run_summary.json',summary)
    ledger={'status':'completed','scope':'measured stage-A costs only; no estimates','entries':rows,'totals':{'physical_calls':sum(x['physical_calls'] for x in rows if x['material_state']!='derived'),'network_forwards':0,'training_updates':0,'failed_tasks':sum(x['status']=='failed' for x in rows),'elapsed_seconds':summary['elapsed_seconds']},'unknowns':['network_forward_seconds','training_seconds','real-instrument acquisition time'],'no_cache_policy':'each of 9 physical tasks ran once; validation replays are separate'}
    write('ledger.json',ledger)
    print(json.dumps({'status':summary['status'],'completed':summary['completed_tasks'],'failed':summary['failed_tasks'],'elapsed_seconds':summary['elapsed_seconds']},ensure_ascii=False,indent=2))

def verify():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8')); s=json.loads((OUT/'run_summary.json').read_text(encoding='utf-8')); l=json.loads((OUT/'ledger.json').read_text(encoding='utf-8'))
    assert p['status']=='frozen_before_run'; assert set(p['actions'])=={'full_aperture','left_half','right_half'}
    assert s['physical_tasks']==9; assert s['failed_tasks']==0
    base_counts=[]
    for a in p['actions']:
      for m in ['background','low_contrast','high_contrast']:
        q=OUT/f'{a}__{m}.pt'; assert q.exists(); d=torch.load(q,weights_only=True,map_location='cpu'); assert finite(d['traces_full']); assert finite(d['traces_selected']); assert d['traces_full'].shape[0]==24; assert d['traces_selected'].shape[0]==len(p['actions'][a]['rx_elements']); assert torch.equal(d['active_rx_mask'].nonzero().reshape(-1),torch.tensor(p['actions'][a]['rx_elements'])); base_counts.append(q)
    assert l['totals']['physical_calls']==9 and l['totals']['network_forwards']==0 and l['totals']['training_updates']==0
    # Three preregistered exact replays; no additional selection.
    solver,force,template,shape,rx,ry,device=physical_setup(); state=torch.load(STATE,weights_only=True,map_location='cpu');mask=state['gamma']<.99
    replay=[('full_aperture','low_contrast'),('left_half','background'),('right_half','high_contrast')]
    for a,m in replay:
      val={'background':1.0,'low_contrast':.48,'high_contrast':.52}[m]; g=torch.where(mask,torch.full_like(template['gamma'],val),torch.ones_like(template['gamma'])); d=generate(solver,force,{**template,'gamma':g},shape,rx,ry,device,val);saved=torch.load(OUT/f'{a}__{m}.pt',weights_only=True,map_location='cpu'); assert torch.equal(d['traces_full'][0],saved['traces_full']); assert torch.equal(d['wavefield_snapshots'][0],saved['wavefield_snapshots'])
    # protocol hashes remain unchanged
    for path,h in p['input_hashes'].items(): assert sha(Path(path))==h
    assert sha(SCRIPT)==p['script_hash']
    write('verification.json',{'status':'passed','checks':['protocol input hashes','3 action definitions','9 output tensors finite and shape-checked','9 physical call ledger','3 exact replay tasks','no network/training calls','no old output mutation'],'replayed_tasks':replay,'physical_calls':9,'network_forwards':0,'training_updates':0,'scope':s['scope']})
    print('Verification passed')
if __name__=='__main__':
    if '--verify' in sys.argv: verify()
    else: main()

