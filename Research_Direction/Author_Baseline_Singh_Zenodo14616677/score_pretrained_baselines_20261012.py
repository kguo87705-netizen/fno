"""Re-score saved pretrained baselines with the previous exact metric."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
OUT=ROOT/'.build/singh-pretrained-author-20261012'
code=(BASE/'score_false_alarm_aligned_20261012.py').read_text(encoding='utf-8')
code=code[:code.index('# Cross-check the reused definition')]
code=code.replace("METHODS = ['conventional', 'nn', 'tv_k1']","METHODS = ['conventional','nn','initial','initial_fwi','transfer','transfer_post5']")
a=code.index('    for m, path in [')
b=code.index('    defect = truth < .5',a)
code=code[:a]+'''    for m in ['nn','initial','initial_fwi','transfer','transfer_post5']:
        if m=='nn':
            path=ROOT/f'.build/singh-nnfwi-author-20261012/case{c}/final_fields.npz'
            field_key='reconstruction'
        else:
            folder='initial_fwi' if m in ['initial','initial_fwi'] else 'transfer'
            path=ROOT/f'.build/singh-pretrained-author-20261012/{folder}/case{c}/final_fields.npz'
            field_key='initial' if m=='initial' else ('post5_reconstruction' if m=='transfer_post5' else 'reconstruction')
        a=np.load(path)
        assert np.array_equal(truth,a['truth'][1:-1,1:-1])
        fields[m]=a[field_key][1:-1,1:-1].astype(float)
'''+code[b:]
ns={'__file__':str(BASE/'score_false_alarm_aligned_20261012.py'),'__name__':'score_only'}
exec(compile(code,ns['__file__'],'exec'),ns)
r,cases=ns['result'],ns['cases']
old=json.loads((ROOT/'.build/singh-tv-author-20261012/false_alarm_aligned_scores.json').read_text(encoding='utf-8'))
for m in ['conventional','nn']:
    assert r['scan'][m]==old['scan'][m]
r['material_mse']={m:[float(((a['fields'][m]-a['truth'])**2).mean()) for a in cases.values()] for m in ns['METHODS']}
r['timing']=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
assert len(r['timing'])==8
r['initial_agreement']={str(c):float(np.max(abs(np.load(OUT/f'initial_fwi/case{c}/final_fields.npz')['initial']-np.load(OUT/f'transfer/case{c}/final_fields.npz')['initial']))) for c in range(1,5)}
r['transfer_reporting']='Primary preserves original script output before last optimizer update; supplementary post5 field also scored'
(OUT/'comparison_scores.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
names={'conventional':'Conventional (35)','nn':'NN (35)','initial':'U-Net initial (0)','initial_fwi':'U-Net + conventional (35)','transfer':'Transfer (5 steps; author saved field)','transfer_post5':'Transfer (post 5 updates)'}
fig,axes=plt.subplots(7,4,figsize=(15,14),sharex=True,sharey=True)
extent=(-ns['DX']/2,100+ns['DX']/2,50+ns['DY']/2,-ns['DY']/2)
for col,(c,a) in enumerate(cases.items()):
    axes[0,col].imshow(a['truth'].T,cmap='coolwarm',vmin=0,vmax=1,extent=extent,interpolation='nearest');axes[0,col].set_title(f'Case {c}')
    for row,m in enumerate(ns['METHODS'],1):
        axes[row,col].imshow(a['fields'][m].T,cmap='coolwarm',vmin=0,vmax=1,extent=extent,interpolation='nearest')
        axes[row,col].set_title(f'MSE {r["material_mse"][m][c-1]:.5f}',fontsize=9)
for row,label in enumerate(['Truth']+[names[m] for m in ns['METHODS']]):axes[row,0].set_ylabel(label+'\ny (mm)',fontsize=8)
for ax in axes.flat:ax.set_xlim(0,100);ax.set_ylim(50,0)
for ax in axes[-1]:ax.set_xlabel('x (mm)')
fig.suptitle('Author pretrained U-Net baselines; shared gamma scale [0,1]')
fig.tight_layout(rect=(0,0,1,.98));fig.savefig(OUT/'pretrained_reconstruction_comparison.png',dpi=150);plt.close(fig)
print(json.dumps({'mse':r['material_mse'],'working_points':{m:{cap:{k:v for k,v in wp.items() if k!='cases'} if wp else None for cap,wp in points.items()} for m,points in r['working_points'].items()},'initial_agreement':r['initial_agreement']},indent=2))
