import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_contrast_derivative_audit import dump
from run_reduced_material_scan import sha
OUT=Path('.build/fno-physical-single-step-control-20261009')
r=json.loads((OUT/'summary.json').read_text())
plan={'status':'planned_not_run','question':'Does surrogate initialization retain a cost/quality advantage once the direct physical scalar baseline gets a fixed sufficient iteration budget?',
 'output_directory':'.build/fno-physical-fixed-budget-control-20261009',
 'targets':[.35,.45,.55,.65],'data_role':'Same now-used development observations, not independent final test',
 'initializations':['frozen original width8 grid-selected a','frozen train-only linear grid-selected a','fixed .5'],
 'fixed':'Same one-step formula, float64 physics, epsilon.002, max absolute step.02, bounds[.3,.7], observations and known support.',
 'primary_change_from_prior':'Exactly8 physical updates per initialization rather than1; all3 methods get the identical update budget.',
 'evaluation_updates':[0,1,2,4,8],
 'selection':'Run all8 updates; final8 is primary result. No stopping or selection from material truth, objective tolerance or best iterate; all cost increases and clips preserved.',
 'cache_policy':'Separate physical cache per target/initialization; reuse the previous F(a1) as the next center only within its own trajectory. No cross-method/cross-target sharing for timing; logical4 calls/update reported alongside actual unique calls.',
 'references':['unchanged initial estimate','direct .5 physical route with same8-update budget','old full201-point physical grid as diagnostic reference only, not new solver call cost'],
 'metrics':['full trajectories of a,cost,raw/clipped delta','material errors scored after fixed updates, full/scatter receiver errors','per-route actual physical calls,cache hits,times',
   'first-stage201-candidate original/linear lookup times and network offline training532.99sec separately; prior training-data generation unknown',
   'quality versus total accounted calls/time at predeclared updates0/1/2/4/8; do not choose a stopping update based on these labels'],
 'decision':{'all_routes_recover_similarly':'Current known-support scalar task offers no neural necessity evidence; favor simplest verified pipeline and preregister a new geometry/parameter axis only if it creates a meaningful operator problem.',
  'surrogate_routes_only_recover':'Initialization benefit at fixed8 steps; require full cost accounting/new validation before neural advantage, and compare linear first.',
  'fixed05_still_clip_limited':'Report optimizer budget limitation, not identifiability failure; no automatic extra updates.',
  'material_worsens_despite_cost_decrease':'No recovery-success claim; audit model/objective mismatch.',
  'late_roundoff_plateau':'Report numerical plateau and label/reference precision, not best-iterate recovery.'},
 'budget':'At most384 logical float64 FD calls, actual within-trajectory cache may reduce calls. Estimated3-5min,cap10min,no training/download,fresh directory,OOM stop.',
 'limits':'Known support scalar optimization; does not validate blind imaging, unknown geometry, noise robustness or an independent final test. Single-run route timings are descriptive, not a formal benchmark.',
 'frozen_sha256':{str(p):sha(p) for p in [OUT/'raw.pt',OUT/'summary.json',OUT/'protocol.json',
   Path('.build/fno-frozen-new-contrast-validation-20261009/observations.pt'),
   Path('.build/fno-frozen-new-contrast-validation-20261009/summary.json'),
   Path('Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_physical_single_step.py')]}}
assert not (OUT/'next_fixed_budget_protocol.json').exists();dump(OUT/'next_fixed_budget_protocol.json',plan)
fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for m,c in [('width8','#2171b5'),('piecewise_linear','#238b45'),('fixed05','#756bb1')]:
    rows=sorted([x for x in r['rows'] if x['method']==m],key=lambda x:x['target']);t=[x['target'] for x in rows]
    axes[0].plot(t,[max(x['contrast_error_before'],1e-10) for x in rows],'o--',color=c,label=m+' before')
    axes[0].plot(t,[max(x['contrast_error_after'],1e-10) for x in rows],'o-',color=c,label=m+' after')
    axes[1].plot(t,[x['cost_ratio'] for x in rows],'o-',color=c,label=m)
axes[0].set(yscale='log',xlabel='Development contrast',ylabel='Absolute contrast error (log)',title='One physical update')
axes[1].set(yscale='log',xlabel='Development contrast',ylabel='Physical cost after / before',title='Cost and material are separate')
for ax in axes:ax.legend(fontsize=6);ax.grid(alpha=.2)
fig.savefig(OUT/'physical_single_step.png',dpi=170)
