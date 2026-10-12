import json
from pathlib import Path
import torch
from run_reduced_material_scan import sha
from run_contrast_derivative_audit import dump
OUT=Path('.build/fno-contrast-precision-control-20261009')
raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu')
parity=[]
for m,states in raw['arms']['default'].items():
    for k,v in states.items():
        old=raw['reference']['methods'][m][k]
        parity.append({'method':m,'state':float(k),'center_bit_identical':torch.equal(v['center'],old['center']),
                      'absolute_gradient_difference':abs(v['gradient']-old['gradient'])})
dump(OUT/'parity_detail.json',{'records':parity,'interpretation':'Default center predictions reproduce exactly; small AD reductions differ across processes. Not evidence that all bits reproduce.'})
plan={'status':'planned_not_run','question':'Does a numerically stable frozen proxy recover contrast as well as train-only interpolation on the same known geometry?',
 'output_directory':'.build/fno-scalar-grid-recovery-20261009','training':'None',
 'methods':['frozen width8 IEEE float32','train-only piecewise linear IEEE float32','original-stencil promoted float64 FD'],
 'targets':[.4,.6],'target_source':'Existing contrast_04/06 full traces, already-used development data, not independent final test',
 'fixed':'Known training000 support, source0, scales, observations, full2000 times and correct time cost; no initial condition/optimizer differences',
 'unique_method_factor':'Forward prediction provider; same predeclared candidate grid and objective for all',
 'grid':'41 float32 candidates from0.30 to0.70 inclusive at0.01 intervals, computed as float32(0.30+i*0.01), i=0..40',
 'selection':'Evaluate every candidate; select minimum observed-data time cost, tie by lower a. Never use true material error to choose candidate, stop, or thresholds. Reuse each prediction for both targets.',
 'references':'FD objective grid and training-only linear; constant predictor as uninformative objective reference, not a recovered-material success',
 'metrics':['all 41 objective values per target/method','selected a and absolute contrast error, scoring only after selection',
   'physical material half-MSE on full physical grid','true FD objective at selected candidate divided by initial a=.5 FD objective',
   'full and scatter receiver error at selected candidate','gap to FD-grid selected a','counts and wall times; include prior data generation/training as separate offline costs, unknown costs explicitly unknown'],
 'interpretation':'Grid-resolution-limited known-support scalar recovery, not iterative blind FWI. Do not claim identical-quality total-cost advantage from this diagnostic.',
 'decision':{'FNO_matches_linear_and_FD':'Next preregister one new geometry axis with matched simple reference; no necessity or speedup claim yet',
   'FNO_biased_linear_good':'Diagnose near-target forward residual bias before derivative training; tangent passed only at two states',
   'linear_and_FNO_biased':'Inspect approximation bias and grid resolution using frozen objectives before any model change',
   'FD_reference_inconsistent':'Stop interpreting surrogate recovery and audit target/reference consistency'},
 'budget':'41 predictions per method,123 total, no training or download, estimated1-3min,10min cap, OOM stop. Save protocol before run, reject existing output directory.',
 'frozen_sha256':{str(p):sha(p) for p in [OUT/'summary.json',OUT/'raw.pt',Path('.build/fno-width8-control-20261009/final.pt'),
     *[Path(f'.build/fno-local-contrast-20261009/contrast_{i:02d}.pt') for i in [3,4,5,6,7]]]},
 'numerical_policy':'Disable TF32 for both cuda.matmul and cudnn, highest float32 precision, no AMP; check current0.4/0.6 centers against frozen IEEE predictions before interpretation',
 'screens':'Do not alter existing derivative20%/cos0.9 screens or numerical1% QA. Report material differences at fixed grid resolution, no posthoc success threshold.'}
assert not (OUT/'next_scalar_recovery_protocol.json').exists()
dump(OUT/'next_scalar_recovery_protocol.json',plan)
print(json.dumps(parity))
