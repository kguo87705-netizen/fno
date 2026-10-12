import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_contrast_derivative_audit import dump
from run_reduced_material_scan import sha
OUT=Path('.build/fno-training-bias-control-20261009')
c=torch.load(OUT/'calibration.pt',weights_only=True,map_location='cpu');es=[x['residual'] for x in c['records']]
def energy(x):return float(torch.trapezoid(x.square(),dx=30e-9,dim=-1).sum())
nonconstant=sum(energy(e-c['bias']) for e in es)/sum(energy(e) for e in es)
diag={'status':'exploratory_training_only_cached_analysis','residual_norms':[float(e.norm()) for e in es],
      'nonconstant_energy_fraction':nonconstant,
      'pairwise_cosines':[float(torch.nn.functional.cosine_similarity(es[i].flatten(),es[j].flatten(),dim=0)) for i,j in [(0,1),(0,2),(1,2)]]}
dump(OUT/'training_residual_diagnostic.json',diag)
plan={'status':'planned_not_run','question':'Does training-derived material dependence of residual improve recovery beyond fixed mean correction?',
 'output_directory':'.build/fno-training-affine-bias-control-20261009',
 'unique_factor':'Constant residual correction versus first-order residual correction in input contrast, fit only from the same3 training states.',
 'formula':'b0=mean(e03,e05,e07); b1=(e07-e03)/(actual_float32_a07-actual_float32_a03); b(a)=b0+(a-mean(actual_training_a))*b1. This is ordinary least squares with centered three actual training contrasts; record exact coefficients. Use exact OLS denominator sum((a-mean_a)^2) if float32 spacing asymmetric.',
 'implementation':'Calculate OLS coefficient tensors from calibration.pt only, save SHA before loading development labels. No neural training, no new predictions, no regularization/strength selection.',
 'baselines':['original width8','constant mean correction','train-only piecewise linear','physical reference','b1=0 must reproduce constant correction'],
 'evaluation':'Same two development observations, frozen37 candidates/common windows, observable time cost argmin and tie rule; all targets and methods reported.',
 'metrics':['material error and true FD cost at selected candidate','full/scatter waveform error at true and selected input','two epsilon material response at a=.4/.6 from frozen IEEE endpoint tensors minus affine offset, report error/cosine against frozen float64 FD',
            'own objective gradient from cached proxy AD and affine slope via time-weighted residual contraction, finite-difference check when feasible; never call FD alone exact network Jacobian'],
 'screens':'Keep original response relative error<=.20/cos>=.9 and numerical1%QA; report both epsilons, no target-based parameter tuning.',
 'decision':{'both_material_improve_and_response_passes':'Supports this development-only low-order receiver calibration; freeze before evaluating predeclared new materials.',
             'mixed_worse_or_response_fails':'Do not promote correction or increase polynomial degree using these targets. Stop calibration escalation and plan fresh frozen material validation with original/linear baselines.',
             'window_edge':'Inconclusive recovery outside window; no automatic expansion.'},
 'budget':'Cached tensor arithmetic only; no network or FD calls/training, expected under1min,cap5min,fresh directory.',
 'limits':'Additional training-data-derived surrogate calibration changes derivatives; must check them. Receiver-only, known-support, development data; no blind/generalization/total-cost claim.',
 'frozen_sha256':{str(p):sha(p) for p in [OUT/'calibration.pt',OUT/'raw.pt',OUT/'summary.json',
    Path('.build/fno-contrast-precision-control-20261009/raw.pt'),Path('.build/fno-scalar-grid-refinement-20261009/raw.pt')]}}
assert not (OUT/'next_affine_bias_protocol.json').exists();dump(OUT/'next_affine_bias_protocol.json',plan)
r=json.loads((OUT/'summary.json').read_text());old=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu')['reference']
fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for ax,(key,grid) in zip(axes,old['windows'].items()):
    for m,color in [('width8','#2171b5'),('corrected_width8','#e6550d'),('piecewise_linear','#238b45'),('physical','#222222')]:
        ax.plot(grid,r['curves'][key][m],'.-',label=m,color=color,markersize=3)
        row=next(x for x in r['rows'] if str(x['target'])==key and x['method']==m)
        ax.scatter(row['selected_a'],row['selected_proxy_cost'],color=color,s=35)
    ax.axvline(float(key),color='gray',linestyle=':');ax.set(xlabel='Contrast a',ylabel='Time objective x 1e12',title=f'Development target a={float(key):.1f}')
    ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.savefig(OUT/'training_bias_control.png',dpi=170)
print(json.dumps(diag))
