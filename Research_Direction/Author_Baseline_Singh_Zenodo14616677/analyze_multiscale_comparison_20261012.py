"""Compare the completed two-cutoff run with saved controls, without solvers."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = ROOT / '.build/singh-multiscale-author-20261012'
# Reuse the preceding scoring implementation, changing only the input methods.
code = (BASE/'score_false_alarm_aligned_20261012.py').read_text(encoding='utf-8')
code = code[:code.index('# Cross-check the reused definition')]
code = code.replace("METHODS = ['conventional', 'nn', 'tv_k1']", "METHODS = ['conventional', 'nn', 'tv_k1', 'ms250', 'ms350']")
code = code.replace("('tv_k1', OUT / f'k1/case{c}/final_fields.npz')", "('tv_k1', OUT / f'k1/case{c}/final_fields.npz'), ('ms250', ROOT / f'.build/singh-multiscale-author-20261012/fc250k/case{c}/final_fields.npz'), ('ms350', ROOT / f'.build/singh-multiscale-author-20261012/fc350k/case{c}/final_fields.npz')")
ns = {'__file__': str(BASE/'score_false_alarm_aligned_20261012.py'), '__name__': 'score_only'}
exec(compile(code, ns['__file__'], 'exec'), ns)
result, cases = ns['result'], ns['cases']
old = json.loads((ROOT/'.build/singh-tv-author-20261012/false_alarm_aligned_scores.json').read_text(encoding='utf-8'))
for m in ['conventional', 'nn', 'tv_k1']:
    assert result['scan'][m] == old['scan'][m]
    assert result['working_points'][m] == old['working_points'][m]
result['material_mse'] = {m: [float(((a['fields'][m]-a['truth'])**2).mean()) for a in cases.values()]
                          for m in ns['METHODS']}
result['mean_material_mse'] = {m: float(np.mean(vals)) for m,vals in result['material_mse'].items()}
best = min(['ms250', 'ms350'], key=lambda m: result['mean_material_mse'][m])
result['best_multiscale'] = best
result['best_selection'] = 'Lower arithmetic mean material MSE across four cases: retrospective oracle selection of two cutoffs'
result['mse_ratio_to_nn'] = {m: (np.array(vals)/result['material_mse']['nn']).tolist() for m,vals in result['material_mse'].items()}
result['mse_pass_count'] = {m: int(np.sum(np.array(vals) <= 1.25)) for m,vals in result['mse_ratio_to_nn'].items()}
timings = json.loads((OUT/'results.json').read_text(encoding='utf-8'))
assert len(timings) == 8 and {(x['cutoff_hz'], x['case']) for x in timings} == {(f,c) for f in [250000,350000] for c in range(1,5)}
result['case_wall_seconds'] = timings
(OUT/'comparison_scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

labels = {'conventional':'Conventional FWI','nn':'NN','tv_k1':'TV k=1','ms250':'Multiscale 250 kHz','ms350':'Multiscale 350 kHz'}
fig,axes=plt.subplots(6,4,figsize=(15,12),sharex=True,sharey=True)
extent=(-ns['DX']/2,100+ns['DX']/2,50+ns['DY']/2,-ns['DY']/2)
for col,(c,a) in enumerate(cases.items()):
    axes[0,col].imshow(a['truth'].T,cmap='coolwarm',vmin=0,vmax=1,extent=extent,interpolation='nearest')
    axes[0,col].set_title(f'Case {c}')
    for row,m in enumerate(ns['METHODS'],1):
        axes[row,col].imshow(a['fields'][m].T,cmap='coolwarm',vmin=0,vmax=1,extent=extent,interpolation='nearest')
        axes[row,col].set_title(f'MSE {result["material_mse"][m][c-1]:.5f}',fontsize=9)
for row,label in enumerate(['Truth']+[labels[m] for m in ns['METHODS']]):
    axes[row,0].set_ylabel(label+'\ny (mm)',fontsize=9)
for ax in axes.flat:
    ax.set_xlim(0,100);ax.set_ylim(50,0)
for ax in axes[-1]:ax.set_xlabel('x (mm)')
fig.suptitle('35-step saved reconstructions; shared gamma scale [0,1]')
fig.tight_layout(rect=(0,0,1,.98));fig.savefig(OUT/'multiscale_reconstruction_comparison.png',dpi=150);plt.close(fig)

methods=['conventional','nn','tv_k1',best]
fig,axes=plt.subplots(4,4,figsize=(15,10),sharex=True,sharey=True)
for row,m in enumerate(methods):
    wp=result['working_points'][m]['0.01']
    for col,(c,a) in enumerate(cases.items()):
        ax=axes[row,col]
        if wp:
            binary=abs(a['fields'][m]-1)>=wp['threshold']*a['contrast']
            ax.imshow(binary.T,cmap='gray_r',vmin=0,vmax=1,extent=extent,interpolation='nearest')
            s=wp['cases'][str(c)]
            ax.set_title(f'Case {c}: {sum(s["detected"])}/{len(s["detected"])}, FPR {s["false_alarm_fraction"]:.2%}',fontsize=9)
        ax.contour(np.arange(256)*ns['DX'],np.arange(128)*ns['DY'],a['defect'].T,levels=[.5],colors='red',linewidths=.6)
        ax.set_xlim(0,100);ax.set_ylim(50,0)
    threshold_label = f't={wp["threshold"]:.2f}' if wp else 'No feasible threshold'
    axes[row,0].set_ylabel(labels[m]+'\n'+threshold_label+'\ny (mm)',fontsize=9)
for ax in axes[-1]:ax.set_xlabel('x (mm)')
fig.suptitle('1% pooled FPR cap: black abnormal pixels, red truth boundaries')
fig.tight_layout(rect=(0,0,1,.97));fig.savefig(OUT/'multiscale_1pct_detection_comparison.png',dpi=150);plt.close(fig)
print(json.dumps(dict(best=best,mse=result['material_mse'],mse_ratios=result['mse_ratio_to_nn'],mse_pass_count=result['mse_pass_count'],
    working_points={m:{cap:{k:v for k,v in wp.items() if k!='cases'} if wp else None for cap,wp in points.items()}
    for m,points in result['working_points'].items()}),indent=2))
