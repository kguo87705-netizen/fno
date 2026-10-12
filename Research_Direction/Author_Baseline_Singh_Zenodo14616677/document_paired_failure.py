"""Document incomplete paired experiment without promoting secondary results."""
import json,re
from pathlib import Path
from run_paired_depth_coverage import OUT,PLAN,ROOT
from run_frozen_depth_shift import WIDE
from run_reduced_material_scan import sha
V=Path('PINO超声成像知识库');TITLE='配对深度覆盖实验存档故障与恢复协议 2026-10-09'
a=json.loads((OUT/'save_failure_audit.json').read_text())
old=json.loads(PLAN.read_text(encoding='utf-8'))
restart=Path(a['verified_restart_checkpoint'])
plan={
 'status':'planned_not_run','question':old['question'],
 'output_directory':'.build/fno-paired-depth-coverage-recovery-20261009',
 'scientific_plan':old,'primary_update':300,
 'recovery_only':'Same paired intervention; do not alter architecture, loss, data, scales, optimizer hyperparameters, sample order, evaluation schedule or screens.',
 'nominal_repeat':'Load verified checkpoint150 including optimizer moments and step counters; run exactly updates151..300, same nominal sample cycle. Do not reset optimizer in this resumed arm. Check update150 state/prediction parity before continuation.',
 'depth_coverage':'Load original width8 final state_dict, new Adam as initially planned; run exactly300 updates cycling depth -2/0/+2. Check its initial parameters and predictions against original nominal arm update0.',
 'new_updates':450,
 'data_role':'All existing diagnostics remain development. No new physical labels, no final test, no validation gradients.',
 'evaluations':'Nominal150 parity then300 primary; coverage0/150/300. Keep final300 primary even if150 is better.',
 'storage':'Convert state_dict and optimizer state recursively to CPU; write each checkpoint to a fresh temporary path, load with weights_only and verify tensors and optimizer step/exposure counters, then rename to a unique final path. Never overwrite old broken checkpoint or prior results. Save loss/progress every25 updates; CPU checkpoint at150/300, plus verified restart on timeout if possible.',
 'preflight':'Record free disk space and test CPU save/load before training; GPU/source/dependency hashes match prior run. Any write failure stops the run and is not interpreted scientifically.',
 'resource_accounting':'Prior stdout reached300 nominal updates but only150 safely checkpointed; recovery repeats150 discarded updates. Across attempts expect750 executed optimizer updates (300 prior+450 recovery), not600 free-compute total. Account failed saving, replay/QA and historical data/training separately; elapsed original training log lower bound302.1sec.',
 'budget':'450 new updates, expected7-9min under observed IEEE execution rate; cap10min with storage preflight included, no downloads/label generation. Save partial progress and stop on cap/OOM/nonfinite/write errors. This revised recovery budget is explicit; not executed in the current turn.',
 'decision':'Use unchanged original causal screen and neural utility gate only after both final300 states are safely saved and reloaded; intermediate or single-arm results cannot replace the paired primary.',
 'frozen_sha256':{**old['frozen_sha256'],**{str(p):sha(p) for p in [OUT/'protocol.json',OUT/'failure.json',OUT/'save_failure_audit.json',OUT/'nominal_repeat_progress.pt',OUT/'nominal_repeat_update150.pt',OUT/'nominal_repeat_update300.pt',restart,ROOT/'run_paired_depth_coverage.py']}}
}
assert not (OUT/'next_recovery_protocol.json').exists()
(OUT/'next_recovery_protocol.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
note='''---
type: experiment-record
date: 2026-10-09
status: 配对主实验未完成；150步检查点已核验；恢复协议未运行
tags: [FNO, 几何覆盖, 实验审查, 可复现性]
---
# 配对深度覆盖实验存档故障与恢复协议

**当前瓶颈是终态存档失败，配对300步主结果仍未知。** 对执行故障和150步检查点完整性置信度高，对“几何覆盖能否修复网络”暂不能更新判断。

承接 [[深度插值基线与冻结FNO比较 2026-10-09]]。项目根相对目录R=`.build/fno-paired-depth-coverage-control-20261009`。协议R/protocol.json，故障R/failure.json，详细检查R/save_failure_audit.json；完好名义150步文件R/nominal_repeat_update150.pt及nominal_repeat_progress.pt，损坏300步文件亦保留。代码`Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_paired_depth_coverage.py`，检查代码`audit_paired_save_failure.py`。

## 1. 事实、解释与未知

| 类型 | 判断 | 证据 |
| --- | --- | --- |
| 已完成 | 原协议已在训练前冻结，名义支路从原width8 checkpoint开始，新建Adam，IEEE；数据/依赖SHA未变 | R/protocol.json、save_failure_audit.json |
| 已完成 | 保存的名义150步状态含Adam150计数、三槽各50次曝光、150条训练损失；五几何场SHA及接收逐位重载一致，误差精确重算 | R/save_failure_audit.json、nominal_repeat_update150.pt、nominal_repeat_progress.pt |
| 已观察 | 运行stdout显示名义更新到300，累计约302.1秒，随后保存300步失败；最终评估未写入结果文件 | 本轮命令输出；R/failure.json及save_failure_audit.json记录 |
| 已完成 | 300步文件无法读取，错误为zip archive缺少central directory；原文件保留，未用作结果 | R/save_failure_audit.json saved_checkpoints |
| 已完成 | 新诊断目录中，CPU序列化150步状态并回读检查通过；重启文件另存并记录SHA | R/save_diagnostic/checkpoint150.pt、save_failure_audit.json |
| 未知 | 原始ios_base::badbit写入故障原因 | 当前磁盘空闲约11.45 GB，不支持直接断言磁盘满；一次CPU保存成功也不能证明原故障机制 |
| 未完成 | 覆盖支路尚未启动；两支路原权重/初始预测一致的配对检查尚未执行 | 现有文件及运行失败位置；不能称已通过双支路检查 |
| 未知 | 双支路final300误差、响应、遗忘及神经收益 | 没有完整配对结果，禁止补猜 |
| 未执行 | 从完整150步恢复名义支路，再从原权重运行覆盖支路，保持300步主结果 | R/next_recovery_protocol.json |

首次启动因Windows默认GBK读取含中文路径的UTF-8计划失败，发生在创建实验目录和训练前；显式UTF-8后才启动。随后训练中的存档故障与此不同，原因仍未知。

## 2. 仅可报告的预定150步次要结果

| 状态 | 名义重复150步散射场误差 | 散射接收误差 |
| --- | ---: | ---: |
| 名义深度0 | 17.566% | 2.748% |
| 开发深度−1 | 33.737% | 34.966% |
| 开发深度+1 | 35.357% | 32.786% |

开发有限几何响应误差121.793%、余弦−.128731。此前原网络约120.787%、余弦−.117990；名义拟合变好未使深度响应通过筛查。**150是原协议预先指定的中间评估，不是择优停止点，也不能替代300步主结果。** 这一单支路结果只限制“150步名义重复就足够修复”的解释，不证明覆盖必要或有效。

完整场/完整接收另见R/save_failure_audit.json，不能用散射误差代称完整波形。此轮没有新物理标签或材料优化。

## 3. 三个竞争解释的更新与反证

| 解释 | 当前支持与冲突 | 可证伪表现 |
| --- | --- | --- |
| H1 缺少几何覆盖使网络响应没有受约束 | 先前线性深度参照可靠；本轮150步名义重复没修复。覆盖臂未运行，因果解释仍无法区分 | 相同初值/预算覆盖臂若优于重复臂并达预定筛查，才支持局部覆盖作用；若无优势则削弱 |
| H2 只是额外优化/名义拟合不足 | 名义训练接收改善到2.75%，开发响应仍差，削弱150步版本；300步状态未知 | 完整300步重复臂若也修复，会削弱覆盖特异解释；不能用150步直接否定所有优化可能 |
| H3 原写入失败由存储空间或序列化路径问题引起 | 写入错误可核查；磁盘仍有空间，CPU回读通过。原因未知，不能确认为空间满或GPU序列化缺陷 | 先做CPU临时保存/严格回读，再原子提交；若继续失败则停下查存储环境，不解释为科研失败 |

H3属于执行可靠性，不能与科研机制混为一谈。此前“时间重建是主因”被削弱的证据保留，本轮没有新证据改变它。

## 4. 对继续与修改计划的反方审查

继续原300步计划仍最合适：关键问题尚未得到结果，不应为了故障改变研究方向。直接把150步作为最终比较会改变主终点；只有名义支路也无法做覆盖归因。原300步未保存，不能凭stdout推测终态指标。

修改仅限执行可靠性：使用已验证150步权重与Adam状态恢复，按原槽序继续151–300；覆盖臂仍从原权重新建Adam运行300步。恢复臂不能重置动量，也不能换损失、数据、精度或学习率。CPU临时保存、回读、唯一文件名提交和更频繁进度存档属于存储流程修复，不是科学处理变化；它们仍须记录成本。

未知原写入原因时，不删除旧结果或假定更大磁盘即可解决。已有150步重载和CPU保存测试足以支持检查点恢复方案，但不保证未来存储绝不失败。

## 5. 候选比较与唯一下一步

| 候选 | 固定因素/变化 | 必须参照 | 成本 | 成功/失败/无法判定及下一步 |
| --- | --- | --- | --- | --- |
| A 恢复原配对实验（唯一优先） | 科学协议不变；名义从150含Adam继续，覆盖从原权重0开始；仅修复存储 | 重复臂、冻结原网络、深度线性两版本、最近邻/零散射 | 新450步，预计7–9分钟，显式上限10分钟；不在本轮执行 | 两臂300均安全存档重载，才按旧筛查判断覆盖效应。若存储/预算/有限性失败，保存部分并保持主结果未知 |
| B 双臂从头重跑 | 科学协议不变，但放弃可恢复150步 | 同A | 新600步，比A多150，未增加科学信息 | 同样需存档修复；若150检查点验证失败才转选此方案 |

不能选择“把150提升为主结果”或“只训练覆盖臂150与名义300比较”，会改变终点或预算配对。

## 6. 恢复协议与成本边界

新目录`.build/fno-paired-depth-coverage-recovery-20261009`，先冻结已验证恢复文件、旧故障/协议/进度、原checkpoint和数据SHA。名义恢复检查点150及Adam状态，原曝光50/50/50，继续恰好150步；覆盖原权重+新Adam，循环−2/0/+2恰好300步。双支路final300仍主结果，无真值停止、择优轮或阈值回调。开发数据已用，不称独立测试。

训练前检查剩余磁盘并试CPU保存回读。递归把权重与优化器状态转CPU，保存到新临时文件，安全加载并核查tensor/计数，再提交唯一正式文件。每25步保存损失/进度；150/300保存并校验checkpoint，超时尽可能保存可验证恢复状态。任一写入失败即停，不把损坏文件视为有效状态。

修订恢复预算450步、预计7–9分钟、上限10分钟，已经明确保存为未运行方案，**本轮没有在旧八分钟预算后追加训练**。按stdout原尝试300＋恢复450，跨尝试预计750次优化更新；其中名义150次需重算，不能只报600次而隐藏失败计算。原尝试时间已有302.1秒下界，最终评估/保存/核验另计。没有新增物理正演或下载。

新证据若表明150检查点损坏或恢复初值/动量不一致，应放弃恢复结论；双支路300完成后才更新H1/H2。覆盖臂波形改善不等于材料方向Jacobian正确、材料恢复或同质量总成本收益，线性参照始终保留。所有结果路径保留，没有覆盖原始实验。
'''
dest=V/'05实践与复习'/f'{TITLE}.md';assert not dest.exists();dest.write_text(note,encoding='utf-8')
targets=[V/'00导航'/'FNO与超声全波形反演科研计划 2026-10-07.md',V/'00导航'/'相控阵FWI与神经算子后续执行路线 2026-10-08.md',V/'04专题'/'相控阵FWI与神经算子阶段总结及知识梳理 2026-10-08.md',V/'05实践与复习'/'深度插值基线与冻结FNO比较 2026-10-09.md']
for p in targets:
 lines=p.read_text(encoding='utf-8').splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('# '))
 lines[i+1:i+1]=['', '> [!important] 配对覆盖实验未完成（2026-10-09）',f'> [[{TITLE}]]：名义300步存档写入失败，覆盖支路未启动；150步检查点重载核验通过。配对主结果未知，不改变科学结论；下一步按原协议恢复，新的恢复预算已保存未运行。','']
 for i,l in enumerate(lines):
  if l.startswith('status:'):lines[i]='status: 配对实验存档失败，检查点已核验待恢复';break
 p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
names={p.stem for p in V.rglob('*') if p.is_file()}|{p.name for p in V.rglob('*') if p.is_file()}
for p in [dest,*targets]:
 s=p.read_text(encoding='utf-8');assert s.startswith('---\n') and s.count('```')%2==0
 for l in re.findall(r'\[\[([^\]]+)\]\]',s):assert l.split('|')[0].split('#')[0] in names,(p,l)
for p,h in plan['frozen_sha256'].items():assert sha(Path(p))==h
assert not Path(plan['output_directory']).exists()
(OUT/'delivery_verification.json').write_text(json.dumps({'status':'passed','note':'All5 edited notes parsed, wikilinks resolved','evidence':'150 checkpoint reload verified;300 corruption explicit;coverage unstarted','next_plan':'planned not run;hashes verified','native_Obsidian_rendering':'not performed','files_sha256':{str(p):sha(p) for p in [dest,*targets]}},ensure_ascii=False,indent=2),encoding='utf-8')
print(dest.resolve())
