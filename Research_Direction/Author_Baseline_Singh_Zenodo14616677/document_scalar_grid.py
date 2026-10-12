"""Exploratory cached residual projection and preregistered refinement plan."""
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_contrast_derivative_audit import dump
from run_reduced_material_scan import sha
OUT=Path('.build/fno-scalar-grid-recovery-20261009')
raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu')
previous=torch.load('.build/fno-contrast-precision-control-20261009/raw.pt',weights_only=True,map_location='cpu')
r=json.loads((OUT/'summary.json').read_text())
def dot(a,b):return torch.trapezoid(a*b,dx=30e-9,dim=-1).sum()
projection=[]
for method in ['width8','piecewise_linear']:
    for key,target in raw['targets'].items():
        v=previous['arms']['ieee'][method][key];lo,hi=v['endpoints'][1]
        j=(v['pairs'][1][1]-v['pairs'][1][0])/(hi-lo)
        e=v['center']-target;step=-float(dot(j,e)/dot(j,j))
        row=next(x for x in r['selected'] if x['method']==method and str(x['target'])==key)
        projection.append({'method':method,'target':float(key),'linearized_bias_prediction_delta_a':step,
                           'observed_coarse_delta_a':row['selected_a']-float(key),
                           'residual_energy_fraction_projected_on_tangent':float(dot(j,e)**2/(dot(j,j)*dot(e,e)))})
dump(OUT/'cached_bias_diagnostic.json',{'status':'exploratory_post_run_no_new_forward_calls',
    'formula':'delta=-<J,e>_time/<J,J>_time, e=proxy_at_target-observation; local Gauss-Newton approximation, not recovered a',
    'records':projection,'limitation':'Uses development target to diagnose error after completed selection; cannot use this as an oracle correction in deployment'})
windows={}
for key in raw['targets']:
    indices=[x['index'] for x in r['selected'] if str(x['target'])==key]
    low=max(0,min(indices)-1);high=min(40,max(indices)+1)
    windows[key]={'coarse_low_index':low,'coarse_high_index':high,
       'grid_formula':f'float32(0.30+j*0.002) for integer j={low*5}..{high*5} inclusive',
       'points':(high-low)*5+1}
plan={'status':'planned_not_run','question':'Separate coarse-grid quantization from persistent scalar proxy bias before expanding geometry or training',
 'output_directory':'.build/fno-scalar-grid-refinement-20261009',
 'unique_primary_change':'Grid step0.01 to0.002 inside common predeclared windows derived only from previous objective-selected candidates',
 'windows':windows,'methods':['width8 IEEE float32','train-only piecewise linear IEEE float32','same float64 physical reference'],
 'fixed':'Same two development observations, known geometry, models, interpolation, cost, IEEE policy and selection rule. No new training.',
 'selection':'All methods use identical candidate window for each observation. Minimize observed-data cost, ties lower a. Score against material only after selection.',
 'metrics':'Save all objective curves, selected a, contrast error, physical material MSE, true FD cost and full/scatter receiver errors, same as coarse protocol. Also compare shifts between grids and cached local-bias prediction without using it for selection.',
 'baseline':'Physical grid and frozen train-only linear; constant predictor remains uninformative',
 'budget':'37 distinct candidates per method,111 forward calls, estimated1-3min, no training/download,10-minute cap and stop on OOM. Fresh directory only.',
 'decision':{'proxy_minimum_inside_and_persistent_bias':'Value residual bias supported. Define only a training-data-derived calibration/control, never subtract test residual; retain linear baseline.',
   'coarse_difference_disappears':'Coarse quantization explained ranking; do not claim precision beyond0.002, then separately plan one geometry axis.',
   'minimum_at_window_edge':'Local refinement inconclusive; report and separately preregister expansion, no automatic window tuning.',
   'physical_reference_not_near_target':'Stop surrogate recovery interpretation and audit reference.'},
 'limits':'Development-only known-geometry scalar diagnostic. No new success threshold or blind recovery/speedup claim.',
 'frozen_sha256':{str(p):sha(p) for p in [OUT/'raw.pt',OUT/'summary.json',OUT/'protocol.json',Path('.build/fno-width8-control-20261009/final.pt')]}}
assert not (OUT/'next_refinement_protocol.json').exists();dump(OUT/'next_refinement_protocol.json',plan)
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for ax,(key,target) in zip(axes,raw['targets'].items()):
    for method,color in [('physical','#222222'),('piecewise_linear','#238b45'),('width8','#2171b5')]:
        y=r['curves'][method][key];ax.plot(raw['grid'],y,label=method,color=color)
        row=next(x for x in r['selected'] if x['method']==method and str(x['target'])==key)
        ax.scatter(row['selected_a'],row['selected_proxy_cost'],color=color,s=35,zorder=4)
    ax.axvline(float(key),color='gray',linestyle=':');ax.set(xlim=(float(key)-.04,float(key)+.04),ylim=(0,.02),
        xlabel='Contrast a',ylabel='Time objective x 1e12',title=f'Development target a={float(key):.1f}')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(OUT/'scalar_grid.png',dpi=170)
print(json.dumps(projection,indent=2))
