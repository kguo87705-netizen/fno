import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_contrast_derivative_audit import dump
from run_reduced_material_scan import sha
OUT=Path('.build/fno-training-affine-bias-control-20261009')
r=json.loads((OUT/'summary.json').read_text())
plan={'status':'planned_not_run','question':'Do frozen original and calibrated receiver surrogates retain material recovery on predeclared new contrasts?',
 'output_directory':'.build/fno-frozen-new-contrast-validation-20261009',
 'targets':[.35,.45,.55,.65],
 'role':'Prospective validation of frozen methods on new contrasts, same previously used geometry. Not an independent final test or unseen-geometry validation.',
 'freeze':'Original width8, constant bias and affine coefficients frozen now. No method selection, refitting, new degree, stopping adjustment or threshold tuning after labels.',
 'methods':['original width8 IEEE','constant training bias IEEE','affine training bias IEEE','train-only piecewise linear','same-stencil float64 physical reference'],
 'primary_factor':'New observation contrast; all five methods scored on exactly the same observations and common candidate grid.',
 'observations':'Generate source0 full2000 traces from same stencil in float32 at four predetermined states; no snapshots/full-field training generation required. Preserve labels and hashes once generated.',
 'candidate_grid':'201 float32 values(0.30+i*0.002), i0..200, full original contrast interval. No target-dependent local windows.',
 'selection':'All201 candidates, observed time-cost minimum, tie lower a; materials score only after selection. Reuse each prediction across all4 observations. No truth-based early stop.',
 'baselines':'Both training-data calibrated methods are retained as diagnostic arms, not promoted remedies; retain original, train-only linear and physical references.',
 'metrics':['per-target selected a, absolute contrast error and physical material halfMSE','true FD cost at selected a relative to initial a=.5',
  'full/scatter receiver error at true input and selected input','all201 objective curves and boundary minima','runtime and call count, offline training/data costs separately unknown where unavailable'],
 'evaluation_rules':'Preserve prior derivative20%/cos0.9 and numerical1% rules, but this experiment does not itself revalidate Jacobian; no new material success threshold or method selection from new labels.',
 'decision':{'calibration_mixed_or_worse':'Keep negative evidence; do not escalate calibration degree or coefficient tuning.',
  'new_material_recovery_consistent':'Only supports the frozen method over this known-geometry contrast interval; next separately freeze one new geometry axis and matched simple baseline.',
  'all_surrogates_biased':'Prioritize physical verification/correction or reduce declared proxy scope; no neural complexity increase from waveform score alone.',
  'boundary_minimum_or_physical_inconsistency':'Report inconclusive/failure, stop interpretation and preregister further audit.'},
 'budget':'201 width8+201 linear+201 float64 physical predictions and4 float32 physical observation calls; calibrated predictions use cached algebra.605 forward calls, estimated3-6min,10min cap, no training/download, OOM stop.',
 'cost_limit':'Single-run timing is descriptive, not same-quality total inversion speedup; candidate-grid recovery remains oracle-known support.',
 'frozen_sha256':{str(p):sha(p) for p in [OUT/'affine.pt',OUT/'summary.json',OUT/'protocol.json',
   Path('.build/fno-training-bias-control-20261009/calibration.pt'),Path('.build/fno-width8-control-20261009/final.pt'),
   Path('.build/fno-local-contrast-20261009/data_manifest.json'),
   *[Path(f'.build/fno-local-contrast-20261009/contrast_{i:02d}.pt') for i in [3,5,7]]]}}
assert not (OUT/'next_frozen_validation_protocol.json').exists();dump(OUT/'next_frozen_validation_protocol.json',plan)
fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for ax,key in zip(axes,r['curves']):
    old=json.loads(Path('.build/fno-scalar-grid-refinement-20261009/summary.json').read_text());grid=old['windows'][key]
    for m,c in [('width8','#2171b5'),('corrected_width8','#e6550d'),('affine_width8','#756bb1'),('physical','#222222')]:
        ax.plot(grid,r['curves'][key][m],'.-',label=m,color=c,markersize=3)
        row=next(x for x in r['rows'] if str(x['target'])==key and x['method']==m)
        ax.scatter(row['selected_a'],row['selected_proxy_cost'],color=c,s=30)
    ax.axvline(float(key),color='gray',linestyle=':');ax.set(xlabel='Contrast a',ylabel='Time objective x1e12',title=f'Development target a={float(key):.1f}')
    ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.savefig(OUT/'affine_bias_control.png',dpi=170)
