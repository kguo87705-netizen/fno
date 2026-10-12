"""Render verified paired results plus a separately preregistered follow-up."""
import json,re,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_depth_coverage_storage_retry import OUT
from run_reduced_material_scan import sha
V=Path('PINO超声成像知识库');TITLE='配对深度覆盖终态审查 2026-10-09'
r=json.loads((OUT/'summary.json').read_text())
assert json.loads((OUT/'verification.json').read_text())['status']=='passed'
p=json.loads((OUT/'next_experiment_protocol.json').read_text(encoding='utf-8'))
assert p['status']=='planned_not_run'
arms=r['arms'];rows=[];trainrows=[];forgetrows=[]
for name,a in arms.items():
 for h in a['history']:
  e=h['evaluation'];q=e['response']
  rows.append(f"| {name} | {h['update']} | "+' | '.join(f"{100*e['records'][k]['scatter_trace_relative_l2']:.3f}%" for k in ['depth-1','depth1'])+f" | {100*q['relative_l2']:.3f}% | {q['cosine']:.6f} | {'通过' if e['screen_passed'] else '未通过'} |")
 e=a['history'][-1]['evaluation']
 for k in ['depth-2','depth0','depth2','depth-1','depth1']:
  v=e['records'][k];trainrows.append(f"| {name} | {k} | "+' | '.join(f"{100*v[s]:.3f}%" for s in ['full_field_relative_l2','scatter_field_relative_l2','full_trace_relative_l2','scatter_trace_relative_l2'])+' |')
 for k in ['contrast03','contrast07']:
  before=a['history'][0]['evaluation']['records'][k];after=e['records'][k]
  forgetrows.append(f"| {name} | {k} | {100*before['scatter_field_relative_l2']:.3f}% → {100*after['scatter_field_relative_l2']:.3f}% | {100*before['scatter_trace_relative_l2']:.3f}% → {100*after['scatter_trace_relative_l2']:.3f}% |")
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for m in ['nominal_repeat','depth_coverage']:
 ev=arms[m]['history'][-1]['evaluation']
 for ax,space in zip(axes,['field','trace']):ax.plot([-1,1],[100*ev['records'][f'depth{k}']['scatter_'+space+'_relative_l2'] for k in [-1,1]],'o-',label=m+' final300')
for m in ['linear_direct','linear_snapshot']:
 rows0=[x for x in r['linear_baselines']['rows'] if x['method']==m]
 for ax,space in zip(axes,['field','trace']):ax.plot([x['shift_rows'] for x in rows0],[100*x['scatter_'+space+'_relative_l2'] for x in rows0],'o--',label=m)
for ax,space in zip(axes,['field snapshots','receiver traces']):
 ax.axhline(20,color='red',linestyle=':',label='20% frozen screen');ax.set(title=space,xlabel='Used development depth shift (rows)',ylabel='Scattered relative L2 error (%)',xticks=[-1,1]);ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.savefig(OUT/'paired_depth_final.png',dpi=170);shutil.copy2(OUT/'paired_depth_final.png',V/'90附件'/f'{TITLE}.png')
note=f'''---
type: evidence-review
date: 2026-10-09
status: 配对300步终态已核验；下一实验已预注册未运行
tags: [FNO, FWI, 几何覆盖, 简单基线, 材料敏感度]
---
# 配对深度覆盖终态审查

{p['bottleneck_cn']}

承接 [[配对深度覆盖实验存档故障与恢复协议 2026-10-09]]。工作区已迁到`D:\\下载\\FNO_Research_Workspace`，相对结果目录R=`.build/fno-paired-depth-coverage-recovery-r2-20261009`；R/protocol.json、raw.pt、summary.json、verification.json分别提供冻结协议、全损失/评估缓存、机器结果和复核。执行代码`Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_depth_coverage_storage_retry.py`，名义恢复代码为`run_paired_depth_recovery.py`。旧故障及损坏文件保留，未覆盖。

## 1. 事实与结果

| 类型 | 可核查判断 | 来源 |
| --- | --- | --- |
| 已完成 | 名义150步权重/Adam状态/预测回读一致后继续到300；覆盖臂从原checkpoint新建Adam，两臂原初权重及初始预测一致 | R/protocol.json、verification.json |
| 已完成 | 每臂终态Adam300、三槽各100曝光、300条损失；同a=.5、源0、损失、归一化、IEEE、结构；仅训练几何分布变化 | R/raw.pt、verification.json；名义[0,0,0]，覆盖[−2,0,+2] |
| 已完成 | 两臂固定0/150/300评估，final300唯一主终态，无择优轮或开发真值停止 | R/raw.pt、protocol.json |
| 已完成 | 机器判定为 `{r['decision']}`；覆盖预定因果筛查 `{r['causal_screen_passed']}` | R/summary.json，按冻结20%/.9筛查 |
| 已完成 | 所有14个终态场SHA和接收波形逐位重载一致，指标精确重算，输入/依赖SHA未变 | R/verification.json |
| 已完成 | 恢复新450次更新；选定轨迹合计600，但跨故障尝试实际750；本轮没有新物理正演 | R/summary.json、verification.json |
| 未知 | 材料方向导数、新材料泛化、未知几何恢复、独立最终测试和同质量完整反演成本 | 本轮没有执行对应验证，不补猜 |
| 未执行 | {p['name_cn']} | R/next_experiment_protocol.json |

预定检查点的开发散射接收与有限几何响应：

| 支路 | 更新数 | 深度−1接收 | 深度+1接收 | 有限几何响应误差 | 响应余弦 | 整体波形/响应筛查 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

全部final300训练/开发四类误差（前3状态为训练几何，后2已用开发状态）：

| 支路 | 状态 | 完整场 | 散射场 | 完整接收 | 散射接收 |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(trainrows)}

原名义几何a=.3/.7的预定遗忘诊断：

| 支路 | 材料 | 散射场0→300 | 散射接收0→300 |
| --- | --- | --- | --- |
{chr(10).join(forgetrows)}

这些材料只用于诊断，不参与梯度更新或停止轮选择；它们是既有训练材料，不是独立最终测试。名义曝光300 vs100是几何分布干预本身，不能称逐样本完全相同。

![[{TITLE}.png]]

### 成本边界

最终覆盖存储重试耗时{r['elapsed_seconds']:.3f}秒，分支秒数来自不同阶段：名义只记录从150恢复到300的{arms['nominal_repeat']['elapsed_seconds']:.3f}秒，覆盖记录新300步的{arms['depth_coverage']['elapsed_seconds']:.3f}秒。名义原先150步费用不在该分支时间中，不能把两者直接当相同300步训练耗时比较。首个失败尝试已有302.1秒下界，加第二次存储故障、迁移、核验及历史训练/标签生成另计；不能只报选定600步隐藏重算150步。显存峰值分别为名义{arms['nominal_repeat']['peak_allocated_bytes']}字节、覆盖{arms['depth_coverage']['peak_allocated_bytes']}字节。线性参考有已有三深度标签，网络此轮覆盖臂也使用相同三深度训练信息；两者仍不同表示/训练费用，推理与完整成本另需公平核算。

## 2. 竞争解释、反方审查与下一实验

{p['review_markdown_cn']}

## 3. 不变的评价边界

完整与散射、全场与接收分别报告；几何有限变化不是局部Jacobian，波形拟合不是材料恢复。当前a=.5训练几何及已用±1开发状态不能代替未见材料/几何泛化。没有网络AD或物理材料导数验证，不能从几何响应通过推出材料敏感度正确。几何是已知正演输入，不是盲定位；单次推理和训练时间不是完整反演总成本。只有独立最终测试才能支持最终泛化声明，本轮没有该数据角色。
'''
dest=V/'05实践与复习'/f'{TITLE}.md';assert not dest.exists();dest.write_text(note,encoding='utf-8')
targets=[V/'00导航'/'FNO与超声全波形反演科研计划 2026-10-07.md',V/'00导航'/'相控阵FWI与神经算子后续执行路线 2026-10-08.md',V/'04专题'/'相控阵FWI与神经算子阶段总结及知识梳理 2026-10-08.md',V/'05实践与复习'/'配对深度覆盖实验存档故障与恢复协议 2026-10-09.md']
for path in targets:
 lines=path.read_text(encoding='utf-8').splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('# '))
 lines[i+1:i+1]=['', '> [!important] 配对300步主结果已核验（2026-10-09）',f'> [[{TITLE}]]：{p["navigation_cn"]}','']
 for i,l in enumerate(lines):
  if l.startswith('status:'):lines[i]='status: 配对300步终态已复核，下一实验已预注册';break
 path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
names={f.stem for f in V.rglob('*') if f.is_file()}|{f.name for f in V.rglob('*') if f.is_file()}
for path in [dest,*targets]:
 s=path.read_text(encoding='utf-8');assert s.startswith('---\n') and s.count('```')%2==0
 for l in re.findall(r'\[\[([^\]]+)\]\]',s):assert l.split('|')[0].split('#')[0] in names,(path,l)
assert sha(OUT/'paired_depth_final.png')==sha(V/'90附件'/f'{TITLE}.png')
for f,h in p['frozen_sha256'].items():assert sha(Path(f))==h
assert not Path(p['output_directory']).exists()
dump={'status':'passed','notes':'All5 wikilinks/structure checked','tables':'Generated from verified summary','next_protocol':'Not run, hashes match','image':'Copied SHA matches; inspect plot separately','native_Obsidian_rendering':'not performed','files_sha256':{str(path):sha(path) for path in [dest,*targets]}}
(OUT/'delivery_verification.json').write_text(json.dumps(dump,ensure_ascii=False,indent=2),encoding='utf-8');print(dest.resolve())
