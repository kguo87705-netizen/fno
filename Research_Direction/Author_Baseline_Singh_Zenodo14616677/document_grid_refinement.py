import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_contrast_derivative_audit import dump
from run_reduced_material_scan import sha
OUT=Path('.build/fno-scalar-grid-refinement-20261009')
r=json.loads((OUT/'summary.json').read_text())
plan={'status':'planned_not_run','question':'Can a fixed training-only mean residual reduce scalar recovery bias without changing the material derivative?',
 'output_directory':'.build/fno-training-bias-control-20261009',
 'training':'No optimization. Three frozen width8 IEEE forward evaluations at training a=.3,.5,.7.',
 'correction':'b=(sum over three train states of (full reconstructed FNO receiver prediction - frozen full receiver training label))/3. Correct every candidate prediction by subtracting the same b, coefficient1, no fitting or tuning.',
 'unique_factor':'Apply versus do not apply constant training-derived receiver offset; frozen model and all other settings unchanged.',
 'data_separation':'Construct and save b and its SHA using only three training labels, before loading development labels or scoring. No development residual, geometry truth adjustment, threshold tuning or best correction selection.',
 'evaluation':'Reuse all37 frozen refined grid predictions and physical reference; same windows, two development observations, min observable time-cost selection and tie rule. Report every candidate and both targets.',
 'baselines':['uncorrected width8','train-only piecewise linear','physical FD grid','zero correction exactly recovers uncorrected results'],
 'metrics':['selected a and contrast error','physical material half-MSE','physical cost relative to initial .5','full and scatter receiver error at true input and selected input','constant correction norm','finite difference of corrected and uncorrected predictions should match to float64 rounding'],
 'invariant':'Constant offset does not change prediction derivative with respect to a; it can change objective gradient by changing residual. This is a receiver-only calibration, not a corrected full-field operator.',
 'decision':{'both_material_errors_improve':'Supports transferable constant training residual for these development cases only; freeze correction before genuinely new materials/geometry evaluation.',
 'mixed_or_worse':'Reject constant bias as a sufficient remedy; do not tune coefficient using these targets. Inspect material-dependent residual and retain simple baseline.',
 'wave_error_only_improves':'No material-recovery success; do not promote correction.',
 'window_edge_minimum':'Inconclusive outside current windows; save failure and separately preregister expansion.'},
 'screens':'Keep existing derivative and numerical QA unchanged. No new posthoc success thresholds; compare per-target material errors at same0.002 grid.',
 'budget':'3 network calls plus cached CPU arithmetic, no training or new physical simulation, expected under1min, cap5min, fresh directory and OOM stop.',
 'limits':'Same known-support development targets. Not blind FWI, independent test, full-field improvement or total-cost advantage.',
 'frozen_sha256':{str(p):sha(p) for p in [OUT/'raw.pt',OUT/'summary.json',OUT/'protocol.json',Path('.build/fno-width8-control-20261009/final.pt'),
    *[Path(f'.build/fno-local-contrast-20261009/contrast_{i:02d}.pt') for i in [3,5,7]]]}}
assert not (OUT/'next_training_bias_protocol.json').exists();dump(OUT/'next_training_bias_protocol.json',plan)
fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
for ax,(key,grid) in zip(axes,r['windows'].items()):
    for m,c in [('physical','#222222'),('piecewise_linear','#238b45'),('width8','#2171b5')]:
        ax.plot(grid,r['curves'][key][m],'.-',label=m,color=c,markersize=3)
        row=next(x for x in r['selected'] if str(x['target'])==key and x['method']==m)
        ax.scatter(row['selected_a'],row['selected_proxy_cost'],color=c,s=45)
    ax.axvline(float(key),color='gray',linestyle=':');ax.set(xlabel='Contrast a',ylabel='Time objective x 1e12',title=f'Development target {float(key):.1f}, step0.002')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(OUT/'refined_grid.png',dpi=170)
