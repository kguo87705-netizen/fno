import json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
OUT=ROOT/'.build/singh-coverage-paired-20261012'
code=(BASE/'score_false_alarm_aligned_20261012.py').read_text(encoding='utf-8')
code=code[:code.index('# Cross-check the reused definition')]
code=code.replace("METHODS = ['conventional', 'nn', 'tv_k1']","METHODS = ['author800','control','experiment']")
a=code.index("    fields = {'conventional':");b=code.index('    defect = truth < .5',a)
code=code[:a]+'''    fields={}
    for m in METHODS:
        path=ROOT/f'.build/singh-transfer35-author-20261012/step5/case{c}/final_fields.npz' if m=='author800' else ROOT/f'.build/singh-coverage-paired-20261012/{m}/case{c}/final_fields.npz'
        saved_fields=np.load(path)
        assert np.array_equal(truth,saved_fields['truth'][1:-1,1:-1])
        key='reconstruction' if m=='author800' else 'post5_reconstruction'
        fields[m]=saved_fields[key][1:-1,1:-1].astype(float)
        assert np.isfinite(fields[m]).all()
'''+code[b:]
ns={'__file__':str(BASE/'score_false_alarm_aligned_20261012.py'),'__name__':'score_only'}
exec(compile(code,'reused_fixed_fpr_scoring','exec'),ns)
r=ns['result'];cases=ns['cases']
old=json.loads((ROOT/'.build/singh-transfer35-author-20261012/comparison_scores.json').read_text(encoding='utf-8'))
assert r['scan']['author800']==old['scan']['t5']
r['material_mse']={m:[float(((a['fields'][m]-a['truth'])**2).mean()) for a in cases.values()] for m in ns['METHODS']}
r['timings']=json.loads((OUT/'timings.json').read_text(encoding='utf-8'))
cw=r['working_points']['control']['0.01'];ew=r['working_points']['experiment']['0.01']
r['decision']={'difference':ew['detected_total']-cw['detected_total'] if cw and ew else None,
               'criterion':'Experimental minus control >=2 detected defects at pooled FPR <=1%',
               'status':'支持数据覆盖是主因之一' if cw and ew and ew['detected_total']-cw['detected_total']>=2 else '未支持' if cw and ew else '无可用工作点，无法判定'}
(OUT/'comparison_scores.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,4,figsize=(14,8),sharex=True,sharey=True)
for row,m in enumerate(ns['METHODS']):
    for col,(c,a) in enumerate(cases.items()):
        ax=axes[row,col]
        ax.imshow(a['fields'][m].T,vmin=0,vmax=1,cmap='coolwarm',extent=(0,100,50,0),interpolation='nearest')
        ax.contour(np.linspace(0,100,256),np.linspace(0,50,128),a['defect'].T,levels=[.5],colors='black',linewidths=.6)
        ax.set_title(f'{m} case {c} MSE={r["material_mse"][m][c-1]:.5f}')
fig.tight_layout();fig.savefig(OUT/'paired_reconstructions.png',dpi=160);plt.close(fig)
print(json.dumps(dict(decision=r['decision'],mse=r['material_mse'],working_points=r['working_points'],timings=r['timings']),indent=2))
record=ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md'
lines=['\n### 同规模覆盖对照：本轮结果',f"结论：{r['decision']['status']}；1%虚警下实验组相对对照组检出差={r['decision']['difference']}。白话：只换训练分布是否补回小缺陷，以预先固定的2个缺陷门槛判断。",'| 方法 | 四例整体MSE | 1%检出/狭缝召回 | 0.5%检出/狭缝召回 | 各例虚警%（1%；0.5%） |','|---|---|---|---|---|']
for m,label in [('author800','作者800样本模型'),('control','单椭圆对照200'),('experiment','覆盖扩展实验200')]:
    pts=r['working_points'][m]
    text=[];fps=[]
    for cap in ['0.01','0.005']:
        p=pts[cap]
        text.append(f"{p['detected_total']}/12 / {100*p['thin_slit_recall']:.2f}%（阈值{p['threshold']:.2f}）" if p else '无工作点')
        fps.append('/'.join(f"{100*p['cases'][str(c)]['false_alarm_fraction']:.3f}" for c in range(1,5)) if p else '无')
    lines.append('| '+label+' | '+' / '.join(f'{x:.6f}' for x in r['material_mse'][m])+' | '+' | '.join(text)+' | '+'；'.join(fps)+' |')
for m in ns['METHODS']:
    for cap in ['0.01','0.005']:
        p=r['working_points'][m][cap]
        if p:
            lines.append(f"{m} {float(cap):.1%}档案例4五孔（左6、中央4、中央3、中央2、右4毫米）："+' / '.join('检出' if p['cases']['4']['detected'][i] else '漏检' for i in [0,2,3,4,5])+f"；合并虚警{p['false_alarm_fraction']:.4%}。")
lines += ['本轮作者设置保持epochs=1、RMSprop lr=5e-4、batchSize=20（N//10）、作者每batch调用调度器且训练前将beta置0的实际行为均保留；不因结果改轮数、超参数或主判据。',
          '作者800样本权重仅为历史参照，训练过程不一定与本地1轮脚本等价；因果比较只看同条件两组200。单轮收敛程度、单随机种子、四个已知合成测试案例及同组真值选阈值限制解释；未支持不等于证明覆盖无作用或U-Net有问题。',
          '完整耗时见`.build/singh-coverage-paired-20261012/timings.json`，指标见同目录comparison_scores.json，图见paired_reconstructions.png。一次计时内存错误已计入本轮；重试后未再失败。已停止，不开始FNO。']
with record.open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
