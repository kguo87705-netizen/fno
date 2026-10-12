"""Save follow-up protocol and plot without changing completed results."""
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_reduced_material_scan import sha

out=Path('.build/fno-contrast-derivative-20261009')
r=json.loads((out/'summary.json').read_text())
plan={
 'status':'planned_not_run',
 'question':'Does reduced-precision convolution contaminate frozen FNO small contrast differences?',
 'new_output_directory':'.build/fno-contrast-precision-control-20261009',
 'single_primary_factor':'Default versus IEEE float32 backend execution policy; identical frozen weights and float32 tensors',
 'methods':['width4 final1200','width8 final1200','frozen train-only piecewise linear'],
 'states':[.4,.6],'epsilons':[.001,.0005],
 'fixed':'Same geometry, source0, receiver/time reconstruction, original target a=.5, scales, actual float32 endpoints, cost and original physical float64 endpoint tensors',
 'arms':{
   'default':'Fresh-process runtime defaults, record actual backend flags before evaluation; never silently assume TF32 was used',
   'ieee':'Disable cuda.matmul.allow_tf32 and cudnn.allow_tf32, highest float32 matmul precision, no AMP; retain float32 weights/inputs'},
 'replicates':'Two identical forward evaluations per center and endpoint per arm to measure repeat variability; scalar AD once per state/method/arm. Methods run sequentially. No training.',
 'baselines':'Original FD float64 endpoints/AD and train-only piecewise linear, plus zero derivative; report undefined cosine for zero',
 'metrics':['absolute waveform change between policies','response relativeL2, norm ratio and cosine at BOTH epsilons',
            'two-epsilon response change','own scalar AD versus both loss differences','physical scalar gradient ratio','repeat variability'],
 'numerical_qa':'Predeclared follow-up QA: two-epsilon response change <=0.01 and own scalar AD/FD relative error <=0.01 in each state. This is a numerical usability check, not a replacement for research screens.',
 'unchanged_science_screens':'Response error <=0.20 and cosine >=0.9, report each state and both epsilons; cannot select the favorable epsilon. Scalar sign/descent remains separate.',
 'decision':{
   'stable_and_passes':'Reduced-precision explanation supported; do not add derivative training. Next separately preregister fixed-geometry scalar recovery/near-target bias against linear and physical baselines.',
   'stable_but_fails':'A genuine learned tangent discrepancy is supported; then design one derivative-informed correction with matched value-only continuation and retained linear baseline.',
   'still_unstable':'Do not label physical Jacobian accurate or inaccurate. Next use a separately budgeted direct JVP/high-precision diagnostic, not more training.',
   'policies_identical':'If runtime flags or effective outputs do not differ, intervention is uninformative about TF32, not evidence excluding all numerical error.'},
 'budget':'No new data or training; <=60 forward evaluations and 12 scalar reverse evaluations; estimated 1-3 minutes on this GPU, operational cap 10 minutes, stop on OOM; report actual cost.',
 'limits':'Two development states, oracle-known fixed support and one contrast direction; neither independent test nor full Jacobian nor blind recovery nor total-cost advantage.',
 'frozen_files_sha256':{str(p):sha(p) for p in [out/'protocol.json',out/'raw.pt',out/'summary.json',
     Path('.build/fno-local-contrast-20261009/final.pt'),Path('.build/fno-width8-control-20261009/final.pt'),
     Path('Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_contrast_derivative_audit.py')]}
}
assert not (out/'next_precision_protocol.json').exists()
(out/'next_precision_protocol.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
supplement={'verification_note':'Initial verify CLI did not fix CPU threads and strict float equality failed; same original four-thread setting recomputed all metrics exactly. No summary or raw tensors overwritten.',
 'verification_command':'.build/author-baseline-env/Scripts/python.exe -c "import sys,torch; sys.path.insert(0,\'Research_Direction/Author_Baseline_Singh_Zenodo14616677\'); torch.set_num_threads(4); import run_contrast_derivative_audit as a; a.verify()"',
 'fresh_process_runtime_probe_not_retrospective_kernel_evidence':{'torch':torch.__version__,
 'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
 'float32_matmul_precision':torch.get_float32_matmul_precision()},
 'status':'Artifact recomputation passed; FNO small-difference convergence unresolved; planned follow-up not run'}
(out/'audit_notes.json').write_text(json.dumps(supplement,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
colors={'width4':'#8c6bb1','width8':'#2171b5','piecewise_linear':'#238b45'}
for method,color in colors.items():
    rows=[x for x in r['rows'] if x['method']==method]
    for i,row in enumerate(rows):
        axes[0].plot([.001,.0005],[100*x['relative_l2'] for x in row['responses']],marker='o',color=color,linestyle=['-','--'][i],label=f'{method}, a={row["state"]:.1f}')
    axes[1].plot([.4,.6],[100*x['two_epsilon_relative_change'] for x in rows],marker='o',color=color,label=method)
    axes[2].plot([.4,.6],[x['gradient_ratio'] for x in rows],marker='o',color=color,label=method)
axes[0].axhline(20,color='gray',linestyle=':');axes[0].set(xlabel='Absolute difference epsilon',ylabel='Response error (%)',title='Smaller epsilon is not more reliable')
axes[0].invert_xaxis();axes[0].legend(fontsize=7)
axes[1].set(xlabel='Contrast a',ylabel='Two-epsilon response change (%)',title='FNO difference has not converged')
axes[1].legend(fontsize=8)
axes[2].axhline(1,color='gray',linestyle=':');axes[2].set(xlabel='Contrast a',ylabel='Proxy AD / physical AD',title='Scalar gradient alone is insufficient')
for ax in axes:ax.grid(alpha=.2)
fig.savefig(out/'derivative_audit.png',dpi=170)
print('Follow-up protocol, audit notes and plot saved')
