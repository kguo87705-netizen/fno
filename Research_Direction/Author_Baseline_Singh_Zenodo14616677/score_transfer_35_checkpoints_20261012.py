"""Same threshold scan and pooled FPR caps at four saved update counts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
OUT=ROOT/'.build/singh-transfer35-author-20261012'
STEPS=[5,10,20,35]
code=(BASE/'score_false_alarm_aligned_20261012.py').read_text(encoding='utf-8')
code=code[:code.index('# Cross-check the reused definition')]
code=code.replace("METHODS = ['conventional', 'nn', 'tv_k1']","METHODS = ['conventional','nn','t5','t10','t20','t35']")
a=code.index('    for m, path in [');b=code.index('    defect = truth < .5',a)
code=code[:a]+'''    for m in ['nn','t5','t10','t20','t35']:
        path=ROOT/f'.build/singh-nnfwi-author-20261012/case{c}/final_fields.npz' if m=='nn' else ROOT/f'.build/singh-transfer35-author-20261012/step{m[1:]}/case{c}/final_fields.npz'
        a=np.load(path)
        assert np.array_equal(truth,a['truth'][1:-1,1:-1])
        fields[m]=a['reconstruction'][1:-1,1:-1].astype(float)
'''+code[b:]
ns={'__file__':str(BASE/'score_false_alarm_aligned_20261012.py'),'__name__':'score_only'}
exec(compile(code,ns['__file__'],'exec'),ns)
r,cases=ns['result'],ns['cases']
prior=json.loads((ROOT/'.build/singh-pretrained-author-20261012/comparison_scores.json').read_text(encoding='utf-8'))
for m in ['conventional','nn']:
    assert r['scan'][m]==prior['scan'][m]
assert r['scan']['t5']==prior['scan']['transfer_post5']
r['material_mse']={m:[float(((a['fields'][m]-a['truth'])**2).mean()) for a in cases.values()] for m in ns['METHODS']}
r['checkpoints']=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
assert len(r['checkpoints'])==16
r['selected_defects']={}
selected=[(3,1,'case3:thin_slit'),(4,0,'case4:left_circle_d6'),(4,2,'case4:central_circle_d4'),
          (4,3,'case4:central_circle_d3'),(4,4,'case4:central_circle_d2'),(4,5,'case4:right_circle_d4')]
for cap in ['0.01','0.005']:
    r['selected_defects'][cap]={}
    for case,index,name in selected:
        rows=[]
        for step in STEPS:
            a=r['working_points'][f't{step}'][cap]
            rows.append(dict(step=step,detected=a['cases'][str(case)]['detected'][index],
                recall=a['cases'][str(case)]['recalls'][index],threshold=a['threshold']) if a else dict(step=step,detected=None,recall=None))
        hits=[a['step'] for a in rows if a['detected']]
        r['selected_defects'][cap][name]=dict(first_saved_step=hits[0] if hits else None,history=rows,
            status='Detected at sampled checkpoint(s)' if hits else '始终漏检（四个保存时点）')
(OUT/'comparison_scores.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True)
for ax,cap in zip(axes,['0.01','0.005']):
    counts=[r['working_points'][f't{s}'][cap]['detected_total'] if r['working_points'][f't{s}'][cap] else np.nan for s in STEPS]
    ax.plot(STEPS,counts,'o-',label='Transfer FWI')
    for m,color in [('nn','tab:orange'),('conventional','gray')]:
        ax.axhline(r['working_points'][m][cap]['detected_total'],color=color,linestyle='--',label=f'{m} (35 updates)')
    for x,y in zip(STEPS,counts):ax.annotate(str(int(y)),(x,y),xytext=(0,8),textcoords='offset points',ha='center')
    ax.set(title=f'{100*float(cap):g}% pooled FPR cap',xlabel='Optimizer updates',xticks=STEPS,ylim=(0,12.8),yticks=range(13))
    ax.grid(alpha=.25);ax.legend(fontsize=8)
axes[0].set_ylabel('Detected defects / 12')
fig.suptitle('Threshold selected separately at each checkpoint; only saved steps observed')
fig.tight_layout();fig.savefig(OUT/'detection_vs_steps.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(5,2,figsize=(9,12),sharex=True,sharey=True)
extent=(-ns['DX']/2,100+ns['DX']/2,50+ns['DY']/2,-ns['DY']/2)
for col,c in enumerate([3,4]):
    a=cases[c]
    axes[0,col].imshow(a['truth'].T,vmin=0,vmax=1,cmap='coolwarm',extent=extent,interpolation='nearest');axes[0,col].set_title(f'Case {c}: truth')
    for row,s in enumerate(STEPS,1):
        axes[row,col].imshow(a['fields'][f't{s}'].T,vmin=0,vmax=1,cmap='coolwarm',extent=extent,interpolation='nearest')
        axes[row,col].set_title(f'{s} updates; MSE {r["material_mse"][f"t{s}"][c-1]:.5f}')
for ax in axes.flat:ax.set_xlim(0,100);ax.set_ylim(50,0)
for ax in axes[:,0]:ax.set_ylabel('y (mm)')
for ax in axes[-1]:ax.set_xlabel('x (mm)')
fig.tight_layout();fig.savefig(OUT/'case3_case4_steps.png',dpi=160);plt.close(fig)
print(json.dumps({'mse':r['material_mse'],'working_points':{m:{cap:{'threshold':a['threshold'],'detected':a['detected_total'],'slit':a['thin_slit_recall'],'percase_fp':[a['cases'][str(c)]['false_alarm_fraction'] for c in range(1,5)]} if a else None for cap,a in points.items()} for m,points in r['working_points'].items()},'selected_defects':r['selected_defects']},indent=2))
