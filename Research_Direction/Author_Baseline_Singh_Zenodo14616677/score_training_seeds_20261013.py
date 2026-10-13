"""Score newly completed seeds once, then aggregate five seeds and update table."""
import json,time
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
BUILD=ROOT/'.build/singh-training-seeds-20261013'
scores={'0':json.loads((ROOT/'.build/singh-coverage-100epochs-20261013/comparison_scores.json').read_text(encoding='utf-8'))}
prefix=(BASE/'score_coverage_paired_20261012.py').read_text(encoding='utf-8')
prefix=prefix[:prefix.index('cw=r[')]
for seed in range(1,5):
    folder=BUILD/f'seed{seed}'
    if not (folder/'timings.json').exists():continue
    times=json.loads((folder/'timings.json').read_text(encoding='utf-8'))
    if 'paired_wall_seconds' not in times:continue
    dest=folder/'comparison_scores.json'
    if dest.exists():r=json.loads(dest.read_text(encoding='utf-8'))
    else:
        code=prefix.replace('singh-coverage-paired-20261012',f'singh-training-seeds-20261013/seed{seed}')
        scope={'__file__':str(BASE/'score_coverage_paired_20261012.py'),'__name__':'saved_seed_scoring'}
        exec(compile(code,'unchanged_seed_scoring','exec'),scope)
        r=scope['r'];dest.write_text(json.dumps(r,indent=2),encoding='utf-8')
    scores[str(seed)]=r
print('COMPLETED_SEED_SCORES '+json.dumps({s:{m:r['working_points'][m]['0.01']['detected_total'] if r['working_points'][m]['0.01'] else None for m in ['control','experiment']} for s,r in scores.items()}),flush=True)
if len(scores)!=5:raise SystemExit(0)
aggregate={};details=[]
for m in ['control','experiment']:
    aggregate[m]={}
    aggregate[m]['mse_mean']=np.mean([scores[str(s)]['material_mse'][m] for s in range(5)],axis=0).tolist()
    for cap in ['0.01','0.005']:
        rows=[scores[str(s)]['working_points'][m][cap] for s in range(5)]
        if any(x is None for x in rows):
            aggregate[m][cap]={'unavailable':True};continue
        counts=[r['detected_total'] for r in rows]
        aggregate[m][cap]=dict(counts=counts,mean=float(np.mean(counts)),min=min(counts),max=max(counts),
            slit_recalls=[r['thin_slit_recall'] for r in rows],slit_mean=float(np.mean([r['thin_slit_recall'] for r in rows])),
            fpr_per_case_mean=np.mean([[r['cases'][str(c)]['false_alarm_fraction'] for c in range(1,5)] for r in rows],axis=0).tolist(),
            case4_hole_hits=[sum(r['cases']['4']['detected'][i] for r in rows) for i in [0,2,3,4,5]])
for s in range(5):
    row={'seed':s,'init_seed':30+s}
    for m in ['control','experiment']:
        p=scores[str(s)]['working_points'][m]['0.01']
        row[m]=None if p is None else dict(detected=p['detected_total'],slit=p['thin_slit_recall'],case3_fpr=p['cases']['3']['false_alarm_fraction'],threshold=p['threshold'])
    details.append(row)
c=aggregate['control']['0.01'];e=aggregate['experiment']['0.01']
if c.get('unavailable') or e.get('unavailable'):
    decision=dict(status='未完成判定：有种子无可用1%工作点',mean_difference=None,wins=None)
else:
    diff=e['mean']-c['mean'];wins=sum(x>y for x,y in zip(e['counts'],c['counts']))
    decision=dict(status='对训练种子稳健' if diff>=2 and wins>=4 else '不稳健',mean_difference=diff,wins=wins)
result=dict(seed_mapping='seed0..4 -> Torch initialization 30..34; sample order and split seeds fixed at 2',details=details,aggregate=aggregate,decision=decision,
            timing=json.loads((BUILD/'run_times.json').read_text(encoding='utf-8')))
(BUILD/'five_seed_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
tablepath=ROOT/'.build/singh-teacher-report-20261013/summary_table.md'
text=tablepath.read_text(encoding='utf-8')
newrows=[]
for m,name in [('control','对照组200'),('experiment','扩展组200')]:
    a=aggregate[m];p=a['0.01']
    cells=[f'{x:.6f}' for x in a['mse_mean']]
    cells+=['未测']*3 if p.get('unavailable') else [f"{p['mean']:.2f}/12",f"{100*p['slit_mean']:.2f}%",f"{100*p['fpr_per_case_mean'][2]:.4f}%"]
    row=f'| {name}（100轮，5种子均值） | '+' | '.join(cells)+' |'
    lines=text.splitlines();matches=[i for i,l in enumerate(lines) if l.startswith('| '+name+'（100轮，5种子均值） |')];assert len(matches)==1
    lines[matches[0]]=row;text='\n'.join(lines)+'\n';newrows.append(row)
text=text.replace('本表只摘录已有comparison_scores.json；未执行新的MSE、阈值扫描或种子均值计算。两个200样本组仅有seed0，不能填写5种子均值。','前五行摘录历史评分；两组200样本行已更新为seed0–4均值。每种子独立按原规则选择1%工作点，再平均MSE、检出数、狭缝召回和案例3虚警。')
text=text.replace('现行代码存在未豁免的预训练裁剪/调度及迁移裁剪差异，新增种子复现未启动；这不是科学上的“不稳健”。','裁剪与调度差异已获书面豁免，按发布代码统一执行；五种子判定：'+decision['status']+'。')
tablepath.write_text(text,encoding='utf-8')
record=ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md'
report=['\n### 训练种子复现结果（seed0–4）',f"判定：{decision['status']}；平均检出差={decision['mean_difference']}，实验组更多的种子数={decision['wins']}/5。门槛未变：平均差≥2且至少4/5获胜。",'| 种子（初始化Torch种子） | 对照1%检出 | 扩展1%检出 | 对照/扩展狭缝召回% | 对照/扩展案例3虚警% |','|---|---|---|---|---|']
for d in details:
    a,b=d['control'],d['experiment']
    report.append(f"| seed{d['seed']}（{d['init_seed']}） | {a['detected']}/12 | {b['detected']}/12 | {100*a['slit']:.2f}/{100*b['slit']:.2f} | {100*a['case3_fpr']:.4f}/{100*b['case3_fpr']:.4f} |" if a and b else f"| seed{d['seed']} | 无工作点 | 无工作点 | 未测 | 未测 |")
for m,name in [('control','对照'),('experiment','扩展')]:
    for cap in ['0.01','0.005']:
        p=aggregate[m][cap]
        if not p.get('unavailable'):
            report.append(f"{name} {float(cap):.1%}档：检出均值{p['mean']:.2f}/12、范围{p['min']}–{p['max']}/12；狭缝召回均值{100*p['slit_mean']:.2f}%；各例虚警均值%="+'/'.join(f'{100*x:.4f}' for x in p['fpr_per_case_mean'])+'；案例4孔频次（左6/中央4/中央3/中央2/右4毫米）='+ '/'.join(f'{x}/5' for x in p['case4_hole_hits'])+'。')
report+=['白话：固定训练数据和划分，只换配对网络训练种子，是否多数时候都能多找回缺陷，按上述预定门槛判断；稳健不等于对新数据或独立阈值也稳健。',
         '总表已更新（完整版本如下）；案例4的seed0 PNG未重做。\n'+text,
         f"新增4种子正式训练及32例5步迁移总耗时{result['timing']['wall_seconds']/60:.2f}分钟；seed0复用上轮结果，不计入本轮新增计算。发生1次显存不足（seed2首个迁移案例）；复用已保存权重，独立进程并回收临时张量后继续，未改变计算设置。完整两档工作点及逐例评分见`.build/singh-training-seeds-20261013/five_seed_summary.json`和seed1–4/comparison_scores.json。\n已停止，不开始FNO。"]
with record.open('a',encoding='utf-8') as f:f.write('\n'.join(report)+'\n')
print('FINAL '+json.dumps(dict(decision=decision,aggregate=aggregate,details=details,minutes=result['timing']['wall_seconds']/60),ensure_ascii=False))
