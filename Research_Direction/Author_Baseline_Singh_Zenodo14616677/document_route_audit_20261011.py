import json,re,hashlib
from pathlib import Path
B=Path('D:/下载/FNO_Research_Workspace');O=B/'.build/route-evidence-audit-20261011';V=B/'PINO超声成像知识库';TITLE='科研路线复盘与证据纠偏 2026-10-11'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
s=json.loads((O/'results.json').read_text(encoding='utf-8'))
plan={'status':'planned_not_run','experiment':'known_support_cross_depth_local_material_response','purpose':'Does nominal anchor replay transfer small material perturbation response to the already-used development depths -1 and+1?','fixed':'Known449-pixel support at each depth;source0,same physical solver and scalar density scaling;central a=.5. Freeze original width8 and original-joint-loss R2 final500,not lowfreq winner selection.','models':[str(B/'.build/fno-width8-control-20261009/final.pt'),str(B/'.build/fno-lowfreq-field-control-r2-20261011/loss_original_update400.pt')],'geometry_rows':[-1,1],'material_center':.5,'epsilons':[.002,.001],'baseline':'Physical nominal-geometry secant (F(.7,0)-F(.3,0))/.4,constant across depth;uses existing nominal training anchors only. Also retain no-change response as zero baseline.','budget':{'new_physical_calls':8,'network_forwards':16,'training_updates':0,'wall_seconds':120},'metrics':'Central directional finite differences in receivers(primary) and field(secondary);both epsilons;relativeL2,amplitude ratio,cosine;complete vs scatter point prediction separately.','gates':'Physical finite-difference stabilization between epsilons <=2% relative,else inconclusive. Existing derivative screen relativeL2<=20%,cosine>=.9 on both depths and both epsilons;network FD stabilization<=2%. Simple baseline remains explicit.','interpretation':'No exact local Jacobian or networkAD claim;this is converged central finite-difference evidence along one known-support direction. Not unseen geometry or independent test;only perturbed material labels newly generated.','outcomes':{'pass':'Preregister fixed-budget known-support single-parameter physical inversion comparison against same simple initializer,then include total offline/online cost. No blind claim.','fail':'Local material response remains bottleneck;compare simple depth/material separable approximation before derivative supervision or architecture change.','reference_unstable':'Fix physical finite-difference precision first,do not train to unreliable labels.'},'frozen_sha256':{str(p):sha(p) for p in [O/'results.json',O/'verification.json',B/'.build/fno-lowfreq-field-control-r2-20261011/loss_original_update400.pt',B/'.build/fno-width8-control-20261009/final.pt']}}
dump(O/'next_experiment_protocol.json',plan)
t='''---
type: research-route-audit
date: 2026-10-11
status: 路线保留；历史解释与核验纠偏；下一项局部材料响应已预注册未运行
tags: [科研复盘, 证据审查, FNO, FWI]
---
# 科研路线复盘与证据纠偏

**结论：可靠正演代理→材料恢复→同质量总成本的总体目标正确，但最近几轮已偏向围绕全场误差连续调试；部分解释由错误轴标注、不同采样率的频段混用、继承的判定代码及不完整核验支撑，应收回过强结论。当前仍是代理可行性与可靠性诊断阶段，不能称阶段即将完成。** 数值局部结果置信度高，物理因果解释置信度低。

本轮实际执行：新目录`.build/route-evidence-audit-20261011`保存冻结文件哈希、独立脚本与结果；读取旧原始结果和优化器状态，并对R2两个模型、两个开发深度做4次冻结正演，0更新、0物理正演。13.26秒为测量阶段，文件哈希预检与文档工作另计。四次预测的全场哈希与接收波形均与原缓存逐位一致，校验空间分区守恒及单边FFT Parseval能量守恒。旧实验文件未改写。

## 1. 可靠事实、假设和未知

| 状态 | 内容 | 来源（D盘工作区下） |
| --- | --- | --- |
| 有结果支持 | 材料回放：名义宽区间接收响应30.30%→6.81%，无材料回放对照34.79%；两个子区间改善 | `.build/fno-matched-material-replay-20261009/raw.pt` |
| 有结果支持 | 同数据移除接收项：全场相对改善5.84%/5.34%，接收误差47.76%/41.87%，几何响应32.61%；存在明显取舍 | 本轮results.json corrected_comparisons；原loss-ablation raw.pt |
| 有结果支持 | 重置Adam：全场相对改善0.865%/0.196%，未达到20%阈值；材料子区间并非全部改善 | 同上；adam-state-control raw.pt |
| 有结果支持 | 低频R2：全场相对改善0.570%/0.682%，远低20%；同Adam的R2可用，R1混入Adam重置不可用于单因素结论 | 本轮checkpoint_audit、corrected_comparisons和四次预测回读 |
| 有结果支持 | 当前原联合损失final500开发深度全场散射28.97%/27.98%，接收散射6.64%/6.67%，名义宽材料响应5.96% | R2 raw.pt；本轮独立重算 |
| 待验解释 | 已用名义材料映射可由回放维护；能否迁移到不同深度的局部材料方向未知 | 回放仅覆盖a=.3/.7的名义几何 |
| 未知 | 新几何/材料联合泛化、多源代理可靠性、独立最终测试、可靠的同质量端到端成本优势 | 当前小实验基本source0，开发数据反复使用 |
| 已完成但范围有限 | 更早做过已知支持单参数、固定八步物理更新/初始化比较；并非从未进入反演实验，但没有证明神经总成本优势 | `.build/fno-physical-fixed-budget-control-20261009`；[[固定八步物理更新与初始化收益审查 2026-10-09]] |

## 2. 必须纠正的历史记录

1. **不是“内域缺少监督”。** `supervised_losses`直接对整个[X,Y,T]网格做MSE；只有额外接收项稀疏。可以讨论各部分权重或优化难度，但不能称内域无标签。
2. **空间轴被写反。** `operator_data.py`明确[X,Y,T]=[256,128,500]；旧诊断把第0轴标成y。91%—94%的“内域”来自x轴252/256，而非文中124/128。真实y轴内域的比例必须另读；阵列侧不能沿错误轴解释。
3. **低频频段不相同。** 2000点接收dt=30ns，旧前10%最后一频点1.65MHz；500点场快照dt=120ns，加权前10%最后一频点0.40MHz。0.50MHz激励被后者排除。原接收残差绝大多数位于1.65MHz以下不等于全场在0.40MHz以下失真。旧单边谱还未做端点/正频率能量权重。
4. **重算后的同Hz证据。** 原联合损失final500全场误差约31%—32%在0–0.4167MHz，约67%—68%在0.4167–1.6667MHz；对应真散射能量约34%—35%和65%，误差频带大致跟随信号。不能据此宣布低频病理或高频容量不足。接收误差在相同两段分别约16%—23%和74%—81%。低频项失败只否定这组截止/权重/100步组合。
5. **Adam方向并非实际下一步。** 旧脚本使用终态m/v构造历史方向，没有纳入当前样本梯度更新到step401。小欧氏余弦本身也可由对角预条件造成，不是优化器失效证据。重置无显著收益只削弱该重置方案，不能普遍排除优化问题。
6. **判定和核验代码复制残留。** 损失消融、Adam控制、低频R2沿用了材料宽响应为主的report()，而各自协议以场误差改善为主。独立重算各自20%场阈值均未通过，数值大方向未翻转，但旧decision字符串不是对应协议的有效判定。R2保存文件名/metadata仍写400，实际Adam是500；raw记录接收权重0，执行代码实际1，summary事后改为1导致旧raw→summary精确相等不再成立。
7. **“核验通过”曾说得过满。** 部分verification.json在失败后人工生成，未覆盖声称的所有重载检查；协议/脚本哈希曾事后修改。此次仅对明确列出的四次预测重新核验，不追认缺失的历史检查。后续新实验必须预检配置、实际优化器步数、独立评价函数与只读协议，不能把重写哈希当作恢复证据。
8. **成本不得只计成功R2。** 最近材料回放、损失消融、Adam控制、低频R1/R2合计1000新更新，其中R1的200步科学上排除但成本保留。原“本轮200步”只代表有效R2，不是两次尝试的400步。预检失败、被删除的零步中断记录及其他未完整计时费用标未知，不能补造。

## 3. 三个竞争解释

| 假设 | 支持 | 冲突/不足 | 可观察预测与反证 |
| --- | --- | --- | --- |
| H1 材料回放仅修复名义几何有限差分，尚未形成跨深度局部响应 | 回放名义结果强，几何训练只a=.5 | 几何接收响应已较好，不能断定跨深度材料必失败 | 固定±1深度、a=.5小扰动与物理参考相符则削弱；不符且参考稳定则支持 |
| H2 在当前监督和100步预算下，全场拟合/优化仍不足 | 训练端点场误差也大；继续原损失有缓慢改善 | 没有收敛证明，也没有固定架构的充分拟合对照 | 不能用小幅loss消融或Adam重置证明容量不足；需以后单独可拟合性对照 |
| H3 诊断/统计实现与反演评价不匹配导致路线偏移 | 轴、频率、report、核验确有问题；把全场指标替代反演目标 | 预测缓存能复现，不能因此把全部实验判无效 | 修复统计后场误差仍大，说明真实误差存在；但其反演重要性须由材料方向/恢复证据判定 |

## 4. 路线选择与反方审查

继续原路线的优点是基线、简单插值和可核查物理参考已建立；问题是连续围绕同一批开发状态改损失，离最终材料恢复与总成本越来越远。直接换网络/加入真值深度编码会改变问题及信息条件；当前无需这样做。

不能因全场约28%就宣布接收代理完全无价值；也不能悄悄降低既定全场20%合格线。保留原“全场代理资格未通过”，另外独立测试反演需要的接收材料局部响应。几何输入仍来自材料图，已知支持仅用于定义诊断扰动，绝不能称盲反演。

| 候选 | 唯一主要变化与基线 | 成本 | 能回答/不能回答 |
| --- | --- | --- | --- |
| 输入编码检查 | 固定网络，仅查材料图/时间通道是否有异；零变化参照 | 无训练，少量前向 | 代码已明确包含gamma、x、y、t；几何响应已有证据，不足以解释剩余材料响应 |
| 重新选频带加权 | 同起点，只改场频带项；原联合损失/线性插值 | 约200更新 | 容易继续开发集调参；目前信号与残差频带相似，缺少优先依据 |
| **不同深度的小扰动材料响应（唯一优先）** | 冻结模型/支持/源/尺度，改变a的小扰动；原FNO、最终联合模型、名义材料线性响应、零响应 | 8物理+16网络前向，0训练，上限120秒 | 直接检查反演需要的方向是否正确；只能支持已知支持单方向，不证明盲识别/总成本 |

## 5. 下一项已冻结、未运行

本轮已经完成证据修复所需的4次新测量；不再依据错误频段继续训练。具体协议见本次实验目录`next_experiment_protocol.json`：两个已用开发深度±1，中心a=.5，固定ε=.002/.001中心差分。先要求物理差分两尺度变化≤2%，再按既有方向误差≤20%、余弦≥.9评价两模型；接收为主，全场单列。参照是已有名义a=.3/.7物理波形斜率在各深度保持不变的简单模型，不使用待测新标签拟合。

若通过，下一步才是同已知支持、固定物理更新预算的材料恢复对照，并纳入离线训练/数据及在线成本；若失败，先比较简单可分离材料/深度近似，再考虑导数监督。若物理差分自身不稳定，先修复数值参考。这个门槛不按新标签改动。

## 6. 总体进度

物理基线与代理初版已建立，名义材料回放有局部受控证据；当前仍在**代理可行性和可靠性审查**。更早的简化材料恢复试验已做过，但多材料/未知几何反演、独立最终测试及同质量成本收益仍未完成。不能用连续完成诊断的数量估算论文完成百分比，也不能称可靠性阶段即将收尾。
'''
n=V/'05实践与复习'/f'{TITLE}.md';assert not n.exists();n.write_text(t,encoding='utf-8')
targets=[V/'00导航/FNO与超声全波形反演科研计划 2026-10-07.md',V/'00导航/相控阵FWI与神经算子后续执行路线 2026-10-08.md',V/'04专题/相控阵FWI与神经算子阶段总结及知识梳理 2026-10-08.md']
names=['Adam预条件方向审查 2026-10-10','Adam状态控制终态审查 2026-10-11','场接收损失消融终态审查 2026-10-10','空间频谱误差覆盖诊断 2026-10-11','低频全场损失加权终态审查 2026-10-11']
targets += [V/'05实践与复习'/f'{x}.md' for x in names]
call=f'\n> [!important] 2026-10-11 证据纠偏，以此为准\n> [[{TITLE}]]：轴与频率定义、Adam方向解释、自动判定/核验和成本记录已复盘；旧结论仅在修订范围内成立。下一项是固定已知支持、不同深度的小扰动材料响应，协议已冻结未运行。\n\n'
for p in targets:
 z=p.read_text(encoding='utf-8-sig').replace('`n','\n');z=re.sub(r'(^# [^\n]+\n)',lambda m:m[0]+call,z,count=1,flags=re.M);p.write_text(z,encoding='utf-8')
allnames={p.stem for p in V.rglob('*.md')};assert all(x in allnames for x in re.findall(r'\[\[([^\]]+)\]\]',t))
dump(O/'documentation_verification.json',{'status':'passed','note':str(n),'linked_notes_exist':True,'navigation_updated':8,'old_experiment_files':'unchanged;see independent audit hashes','native_Obsidian_render':'not performed'})
print(n)
