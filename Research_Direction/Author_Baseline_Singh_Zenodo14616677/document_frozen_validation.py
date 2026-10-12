import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_contrast_derivative_audit import dump
from run_reduced_material_scan import sha
OUT=Path('.build/fno-frozen-new-contrast-validation-20261009')
r=json.loads((OUT/'summary.json').read_text())
plan={'status':'planned_not_run','question':'Can one fixed physical scalar correction reduce frozen surrogate recovery bias, and does neural initialization add value over linear lookup?',
 'output_directory':'.build/fno-physical-single-step-control-20261009',
 'targets':[.35,.45,.55,.65],'observations':'Reuse frozen new-contrast observations; now development/diagnostic data, not untouched final test',
 'methods':['uncorrected width8 grid estimate followed by one physical update','train-only linear grid estimate followed by one physical update','fixed a=.5 followed by same one physical update'],
 'fixed':'Same geometry/source0, observations, physical float64 stencil, objective, epsilon, max step and number of updates. No fitted bias methods promoted.',
 'primary_factor':'Initial scalar estimate supplied by frozen FNO, train-only linear or fixed background-reference contrast',
 'update':'At a0 compute physical F(a0), F(a0+0.002), F(a0-0.002). J=(Fplus-Fminus)/0.004, r=F(a0)-d. delta=-<J,r>time/<J,J>time. Clamp delta to[-0.02,0.02], clamp a1 to[.3,.7], then compute F(a1). Exactly one step, no line search, acceptance selection, truth stopping or iteration tuning.',
 'precision':'a0 from frozen grid values, physical update endpoints and a1 use float64; record actual a and endpoints. Same original coefficients promoted float64.',
 'report':'All4 targets and3 initializations, even increases. Record raw/clipped delta, true cost before/after, full/scatter receiver errors, material error after completing step, slope norm and boundary flags.',
 'references':['no-update starting estimate','same physical grid optimum from prior cache','linear lookup initialization under identical update budget','fixed .5 direct-physical start, explicitly different starting estimate'],
 'cost_accounting':'Four physical calls per target/initialization,48 calls; original and linear201-point lookup costs are prior recorded first-stage costs and must remain separate. Shared/cached calls reported; no inference of full inversion amortization from one-step timing.',
 'decision':{'both_FNO_and_linear_correct':'Supports hybrid physical verification/correction; if quality comparable, no evidence neural first stage is needed for this task.',
             'FNO_correct_linear_not':'Only one-step known-support evidence; require new frozen validation and full cost before advantage claim.',
             'cost_decreases_but_material_worsens':'Do not label recovery success; revisit physical objective/material relationship.',
             'updates_increase_cost_or_boundary':'Report failure; separately preregister optimizer/step diagnostic, no automatic retuning.'},
 'budget':'48 float64 FD calls, estimated under1min to2min,cap5min,no network training/download, fresh directory, OOM stop.',
 'limits':'Single known-support scalar Gauss-Newton step on used prospective-validation observations. Not blind FWI, final testing, complete optimization or same-quality total-cost advantage.',
 'frozen_sha256':{str(p):sha(p) for p in [OUT/'raw.pt',OUT/'summary.json',OUT/'observations.pt',OUT/'protocol.json',
    Path('Research_Direction/Author_Baseline_Singh_Zenodo14616677/discrete_reference.py'),
    Path('Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_frozen_contrast_validation.py')]}}
assert not (OUT/'next_physical_step_protocol.json').exists();dump(OUT/'next_physical_step_protocol.json',plan)
methods=['width8','constant_width8','affine_width8','piecewise_linear'];colors=['#2171b5','#e6550d','#756bb1','#238b45']
targets=sorted(set(x['target'] for x in r['rows']))
fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
for m,c in zip(methods,colors):
    rows=sorted([x for x in r['rows'] if x['method']==m],key=lambda x:x['target'])
    axes[0].plot(targets,[x['absolute_contrast_error'] for x in rows],'o-',label=m,color=c)
    axes[1].plot(targets,[100*x['true_input_scatter_receiver_error'] for x in rows],'o-',label=m,color=c)
axes[0].set(xlabel='Predeclared new contrast',ylabel='Absolute recovered contrast error',title='Material ranking depends on contrast')
axes[1].set(xlabel='Predeclared new contrast',ylabel='Scatter receiver error (%)',title='Waveform improvement does not guarantee recovery')
for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.savefig(OUT/'frozen_validation.png',dpi=170)
print('Frozen physical-step protocol and plot saved')
