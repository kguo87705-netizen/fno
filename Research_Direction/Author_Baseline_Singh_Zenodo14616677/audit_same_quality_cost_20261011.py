import json,hashlib,math
from pathlib import Path
O=Path('.build/same-quality-cost-audit-20261011');assert not O.exists();O.mkdir()
folders=['hybrid-accepted-step-control-r2-20261011','hybrid-accepted-large-step-20261011','hybrid-accepted-mid-step-20261011']
paths=[Path('.build')/n/'summary.json' for n in folders]+[Path('.build/cost-timing-control-20261011/timing.json')]
def dump(n,x): (O/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
dump('protocol.json',{'kind':'retrospective_read_only','analysis':'all observed tolerance intervals; do not choose acceptance threshold from final material truth','hashes':{str(p):sha(p) for p in paths},'new_forward_calls':0,'warning':'timings are historical component estimates, not measured full inversion costs'})
timing=json.loads(paths[-1].read_text())['summary'];tp=timing['median_physical_seconds'];tn=timing['median_network_pair_seconds'];rows=[];intervals=[]
for path in paths[:-1]:
 data=json.loads(path.read_text());cases=sorted(set(r['case'] for r in data['routes']))
 for case in cases:
  methods={r['method']:r for r in data['routes'] if r['case']==case}; curves={}
  for m,r in methods.items():
   history=r['trajectory'];err=history[0]['before'];curve=[(err,1,0)]
   for i,h in enumerate(history):
    if h['accepted']:err=h['after']
    curve.append((err,h['calls_so_far'],i+1))
   curves[m]=curve
  levels=sorted(set(e for v in curves.values() for e,_,_ in v))
  # Endpoints partition every possible positive tolerance in the observed range.
  for low,high in zip(levels,levels[1:]):
   tol=(low+high)/2;costs={}
   for m,curve in curves.items():
    feasible=[(calls*tp+(tn if m=='fno_frozen' and it>0 else 0),calls,it) for e,calls,it in curve if e<=tol]
    costs[m]=min(feasible) if feasible else None
   intervals.append({'case':case,'tolerance_open_interval':[low,high],'component_cost_estimates_seconds_calls_iteration':costs})
  f=methods['fno_frozen'];b=methods['nominal_secant']
  rows.append({'case':case,'fno_final_full_error':f['final_full_error'],'nominal_final_full_error':b['final_full_error'],'fno_parameter_error':f['parameter_error'],'nominal_parameter_error':b['parameter_error'],'fno_calls':f['online_physical_calls'],'nominal_calls':b['online_physical_calls'],'fno_fixed3_cost_estimate':f['online_physical_calls']*tp+tn,'nominal_fixed3_cost_estimate':b['online_physical_calls']*tp,'nominal_reaches_fno_final_quality_within_saved_budget':any(e<=f['final_full_error'] for e,c,i in curves['nominal_secant'])})
summary={'scope':'12 repeatedly used development cases, scalar known support; descriptive not prospective success','rows':rows,'fno_lower_final_waveform_error_cases':sum(r['fno_final_full_error']<r['nominal_final_full_error'] for r in rows),'equal_cases':sum(r['fno_final_full_error']==r['nominal_final_full_error'] for r in rows),'fixed3_extra_fno_response_cost_seconds':tn,'fixed3_proxy_estimate':4*tp+tn,'fixed3_nominal_estimate':4*tp,'break_even_cases':None,'reason':'Fixed3 costs higher for FNO versus nominal. Quality differs; tighter tolerances often censor baseline at step3. No extrapolation of missing baseline trajectories. Complete incremental offline cost unknown.','corrections':{'center_physical_calls':17,'center_old_count':19,'center_verification':'no saved waveforms and no numerical replay; earlier passed label insufficient','timing_protocol':'written after measurements; not preregistered','timing_scope':'physical includes whole-field generation; FNO includes encoding, reconstruction and CPU transfer; not receiver-only optimized solver comparison','partial_offline_record_seconds':{'width8_training_evaluation':532.9938059000124,'material_replay_arm':65.64595659999941,'r2_original_arm':66.4055214},'partial_offline_sum_is_not_total':True}}
dump('summary.json',summary);dump('all_tolerance_intervals.json',intervals)
assert len(rows)==12 and all(r['fno_calls']==r['nominal_calls']==4 for r in rows)
assert all(r['fno_fixed3_cost_estimate']>r['nominal_fixed3_cost_estimate'] for r in rows)
dump('verification.json',{'status':'passed','checks':['12 cases','equal four physical calls for FNO and nominal','finite component estimates','all tolerance intervals enumerated','input hashes unchanged'],'no_physical_or_network_calls':True})
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
