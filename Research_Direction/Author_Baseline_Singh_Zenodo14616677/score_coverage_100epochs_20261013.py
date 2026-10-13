import json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
OUT=ROOT/'.build/singh-coverage-100epochs-20261013'
old=(BASE/'score_coverage_paired_20261012.py').read_text(encoding='utf-8')
old=old[:old.index('record=ROOT/')]
old=old.replace('singh-coverage-paired-20261012','singh-coverage-100epochs-20261013')
exec(compile(old,'same_coverage_scoring','exec'),globals())
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
for ax,group in zip(axes,['control','experiment']):
    history=np.load(OUT/group/'loss_history.npz')
    assert len(history['validation'])==100
    for key in ['training','validation']:
        ax.plot(np.arange(1,101),history[key],label=key)
    ax.set(title=f'{group}: fixed 100 epochs',xlabel='Epoch',ylabel='Author loss (0.5 * mean squared error)',yscale='log')
    ax.legend();ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(OUT/'validation_loss_curves.png',dpi=170);plt.close(fig)
record=ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md'
lines=['\n### 100轮重做结果（替代上一轮无效结论）',f"结论：{r['decision']['status']}；1%虚警下实验减对照检出差={r['decision']['difference']}，预定支持门槛为≥2。白话：以同样训练到100轮的两组200样本比较，判断补充缺陷类型是否补回漏检。",'| 方法（迁移5步） | 四例整体MSE | 1%检出/狭缝召回 | 0.5%检出/狭缝召回 | 各例虚警%（1%；0.5%） | 最终验证损失 |','|---|---|---|---|---|---|']
for m,label in [('author800','作者800样本模型'),('control','单椭圆对照200，100轮'),('experiment','覆盖扩展200，100轮')]:
    pts=r['working_points'][m];text=[];fps=[]
    for cap in ['0.01','0.005']:
        p=pts[cap]
        text.append(f"{p['detected_total']}/12 / {100*p['thin_slit_recall']:.2f}%（t={p['threshold']:.2f}）" if p else '无工作点')
        fps.append('/'.join(f"{100*p['cases'][str(c)]['false_alarm_fraction']:.3f}" for c in range(1,5)) if p else '无')
    val='未提供，不能补造' if m=='author800' else f"{r['timings'][m]['training']['validation_loss']:.6g}"
    lines.append('| '+label+' | '+'/'.join(f'{x:.6f}' for x in r['material_mse'][m])+' | '+' | '.join(text)+' | '+'；'.join(fps)+' | '+val+' |')
for m in ['control','experiment']:
    s=r['timings'][m]['training'];c=s['convergence']
    lines.append(f"{m}验证曲线：末10轮均值{c['last10_mean']:.6g}，相对前10轮变化{c['relative_change']:+.2%}，末10轮变异系数{c['last10_cv']:.2%}；"+('符合预设末段平台检查（非数学收敛证明）' if c['late_plateau'] else '未满足末段平台检查，未确认收敛')+f"；训练含加载验证{s['seconds']:.2f}秒。")
for cap in ['0.01','0.005']:
    parts=[]
    for m in ns['METHODS']:
        p=r['working_points'][m][cap]
        parts.append(m+':'+('/'.join('检出' if p['cases']['4']['detected'][i] else '漏检' for i in [0,2,3,4,5]) if p else '无工作点'))
    lines.append(f"{float(cap):.1%}档案例4孔（左6、中央4、中央3、中央2、右4毫米）："+'；'.join(parts))
control=r['working_points']['control']['0.01']
lines.append('对照仍只有200样本，作者权重800样本；'+(f"对照检出{control['detected_total']}/12，低于作者7/12，不能将这项差距归于数据覆盖。" if control else '对照无工作点，无法与作者7/12直接比较。')+'两组验证集分布不同，验证损失不可当作同测试集的优劣。')
lines.append(f"本轮正式训练+八次迁移耗时{r['timings']['paired_wall_seconds']/60:.2f}分钟，无新增数据。曲线`.build/singh-coverage-100epochs-20261013/validation_loss_curves.png`、图paired_reconstructions.png、指标comparison_scores.json。保留四合成案例、单种子、同组真值选阈值的局限；若未确认收敛，不能强推覆盖的负结论。")
lines.append('已停止；不增加轮数、不改超参数、不开始FNO。以后脚本与论文/Readme设置不一致，先告知负责人再跑。')
with record.open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print('SUMMARY '+json.dumps(dict(decision=r['decision'],mse=r['material_mse'],points={m:{cap:dict(n=p['detected_total'],slit=p['thin_slit_recall'],threshold=p['threshold'],fpr=[p['cases'][str(c)]['false_alarm_fraction'] for c in range(1,5)],holes=[p['cases']['4']['detected'][i] for i in [0,2,3,4,5]]) if p else None for cap,p in ps.items()} for m,ps in r['working_points'].items()},training={g:r['timings'][g]['training'] for g in ['control','experiment']},seconds=r['timings']['paired_wall_seconds']),ensure_ascii=False))
