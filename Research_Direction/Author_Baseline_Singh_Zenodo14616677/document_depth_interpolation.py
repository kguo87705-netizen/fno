"""Record the verified interpolation control; freeze the paired next test."""
import json,re,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_depth_interpolation import OUT,OLD
from run_frozen_depth_shift import WIDE,BASE,BG,ROOT
from run_reduced_material_scan import sha
V=Path('PINO超声成像知识库');TITLE='深度插值基线与冻结FNO比较 2026-10-09'
r=json.loads((OUT/'summary.json').read_text())
assert json.loads((OUT/'verification.json').read_text())['status']=='passed'
plan={
 'status':'planned_not_run','question':'Does local depth training coverage, rather than merely more updates, repair the frozen FNO response within a fixed budget?',
 'output_directory':'.build/fno-paired-depth-coverage-control-20261009',
 'arms':{'nominal_repeat':[0,0,0],'depth_coverage':[-2,0,2]},
 'initialization':'Both arms load the identical original width8 final state_dict; verify identical parameter hashes and predictions at update0. Reset Adam in both arms; do not reuse old optimizer momentum.',
 'fixed':'Same model architecture, a=.5, source0, stored physical labels, IEEE policy, training wavefield/trace scales, field+receiver loss, Adam lr.001,300 total updates per arm, same index cycle0/1/2, no augmentation.',
 'only_main_change':'Training geometry distribution: three copies of nominal state versus -2/0/+2. Each arm300 exposures,100 per slot; nominal exposure counts differ as intended distribution intervention.',
 'training_data':'Reuse prior three physical states from depth audit raw.pt; no new physical training labels. Validation +/-1 from interpolation raw.pt is never included in gradient updates.',
 'validation_role':'Previously used diagnostic development states, not fresh final testing. Purpose is controlled learning mechanism, not independent generalization certification.',
 'evaluation_updates':[0,150,300],'primary_update':300,
 'stop_rule':'Run exactly300 per arm; final300 primary. No best checkpoint selection, validation early stopping, truth-informed clipping or threshold changes. Preserve both intermediate states and failures.',
 'metrics':['Separate train/dev complete/scattered field/receiver errors for every state','Dev +/-1 finite geometry receiver response error/norm ratio/cosine','Actual updates,exposures,losses,gradient finiteness,state hashes,reload parity,per-arm time and peak memory','Nominal contrasts .3/.7 at updates0/300 as descriptive forgetting checks only; no training or stopping based on these'],
 'baselines':['Unchanged original width8','Frozen train-only depth linear direct and snapshot versions','Nominal repeated-training arm (more-updates control)','Nearest depth and zero scatter retained'],
 'causal_screen':'At final300 coverage arm must have lower scatter field and receiver error on both dev states and lower dev response relative error than nominal-repeat arm; also meet inherited both-state20% errors and response20%/cos.90 screens. Improvement without absolute screen is partial evidence only.',
 'neural_utility_gate':'Separately compare coverage arm to frozen depth linear baselines on both errors and response plus accounted preparation/training/inference costs. Passing coverage screen alone does not establish neural superiority or total inversion cost benefit.',
 'decisions':{'coverage_repairs_and_repeat_does_not':'Supports local geometry coverage as a cause within identical architecture/start/budget; keep simple interpolation unless neural benefit is separately demonstrated.',
 'both_repair_similarly':'More updates/nominal fitting may suffice, coverage-specific mechanism not established.',
 'coverage_improves_but_fails_screens':'Partial coverage effect; no automatic more epochs. Inspect train fit and representation before a new budget.',
 'neither_repairs':'This300-step intervention does not establish a repair; distinguish training fit from dev response, not capacity conclusion.',
 'controls_fail':'Inconclusive; stop implementation interpretation.'},
 'budget':'600 total optimizer updates; estimated4-6min, cap8min for paired training/evaluation, no downloads or data regeneration; stop on OOM/nonfinite, no grid/mode/precision reduction. Separate fixed-state reload verification allowed.',
 'scope':'Local fixed-shape/source0 depth family at a=.5; supplied geometry forward task, no blind localization, material inversion, physical material Jacobian or independent final test.',
 'frozen_sha256':{str(p):sha(p) for p in [OLD/'raw.pt',OUT/'raw.pt',OUT/'summary.json',OUT/'verification.json',OUT/'references.pt',OUT/'reference_commit.json',WIDE/'final.pt',BASE/'data_manifest.json',BASE/'contrast_03.pt',BASE/'contrast_07.pt',BG,ROOT/'run_receiver_loss_control.py',ROOT/'time_feature_input.py',ROOT/'basic_fno.py']}
}
assert not (OUT/'next_paired_depth_coverage_protocol.json').exists()
(OUT/'next_paired_depth_coverage_protocol.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
methods=['width8','linear_direct','linear_snapshot','nearest','zero_scatter']
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for m in methods:
 rows=[x for x in r['rows'] if x['method']==m]
 for ax,space in zip(axes,['field','trace']):
  ax.plot([x['shift_rows'] for x in rows],[100*x['scatter_'+space+'_relative_l2'] for x in rows],'o-',label=m)
for ax,space in zip(axes,['field snapshots','receiver traces']):
 ax.axhline(20,color='red',linestyle=':',label='20% predeclared screen');ax.set(xlabel='New depth shift (grid rows)',ylabel='Scattered relative L2 error (%)',title=space,xticks=[-1,1]);ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.savefig(OUT/'depth_interpolation_errors.png',dpi=170)
shutil.copy2(OUT/'depth_interpolation_errors.png',V/'90附件'/f'{TITLE}.png')
rows=[]
for m in methods:
 a,b=[x for x in r['rows'] if x['method']==m];q=r['finite_geometry_receiver_responses'][m]
 rows.append(f"| {m} | {100*a['scatter_field_relative_l2']:.3f}% / {100*b['scatter_field_relative_l2']:.3f}% | {100*a['scatter_trace_relative_l2']:.3f}% / {100*b['scatter_trace_relative_l2']:.3f}% | {100*q['relative_l2']:.3f}% | "+('未知（零响应）' if q['cosine'] is None else f"{q['cosine']:.6f}")+f" | {'通过' if r['screen_passes'][m] else '未通过'} |")
note=f'''---
type: evidence-review
date: 2026-10-09
status: 深度插值比较完成并复核；配对覆盖协议已保存未运行
tags: [FNO, FWI, 几何泛化, 简单基线, 证据审查]
---
# 深度插值基线与冻结FNO比较

**一句话判断：三深度训练标签的线性参照在新的中间位置同时通过波形与有限几何响应筛查，冻结FNO仍失败；当前局部正演已有可行简单参照，而网络失效是否由训练覆盖不足造成仍需配对干预。** 本轮误差结论置信度高，覆盖因果解释置信度中低。

承接 [[冻结FNO深度平移审查与几何响应 2026-10-09]]。R为项目根相对路径`.build/fno-depth-interpolation-control-20261009`；协议、训练参照提交、缓存、结果与复核见R/protocol.json、reference_commit.json、references.pt、raw.pt、summary.json、verification.json。代码`Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_depth_interpolation.py`。前轮三深度来自`.build/fno-frozen-depth-shift-audit-20261009-r2/raw.pt`，以下简称D。

## 1. 事实表：完成、解释、未知

| 类型 | 内容 | 来源 |
| --- | --- | --- |
| 已完成 | 三深度−2/0/+2旧诊断标签明确转为训练参照；只在新−1/+1位置生成标签，a=.5及模型不变 | R/protocol.json、raw.pt |
| 已完成 | 线性−1取训练−2/0各.5，+1取0/+2各.5；最近邻平局取较小有符号深度；参照在生成验证标签前保存SHA | R/reference_commit.json、references.pt、执行顺序 |
| 已完成 | 两个移位均449格，无环绕且在域内，源/接收/时间/空间坐标逐位不变；实际移位约±.3937毫米 | R/raw.pt、summary.json checks、代码断言 |
| 已完成 | 线性直接接收及快照重建版本均通过预定20%波形/响应和.9余弦筛查；原网络、最近邻、零散射均未通过 | R/summary.json |
| 已完成 | 两侧物理响应范数约名义散射范数.548759，不是近零分母；零散射误差均100% | R/summary.json、verification.json |
| 已完成 | 主实验2物理＋2网络、无训练，约8.649秒；核验另行重放2＋2，全部场/接收与参照逐位一致，依赖SHA不变 | R/summary.json、verification.json |
| 合理解释 | 局部物理变化可以由简单深度插值表达；网络固定几何训练可能没有约束正确响应 | 前半句有本轮证据；后半句未做同初值配对训练，不能当事实 |
| 未知 | 同架构增加深度覆盖是否修复、能否超越插值、材料方向导数及反演收益 | 尚无干预/材料导数/反演与完整成本结果 |
| 未执行 | 相同原checkpoint、相同300步的名义重复 vs 三深度覆盖训练 | R/next_paired_depth_coverage_protocol.json，planned_not_run |

## 2. 结果与评价边界

下表左右分别为−1/+1行；完整波形的分母与散射不同，完整四类指标另见R/summary.json，不混为一类。

| 方法 | 散射场误差（−1/+1） | 散射接收误差（−1/+1） | 接收有限几何响应误差 | 响应余弦 | 预定筛查 |
| --- | --- | --- | ---: | ---: | --- |
{chr(10).join(rows)}

直接线性接收完整误差约2.152%/2.091%，与散射6.570%/6.498%不能互换。线性快照重建与直接插值接收之差相对真实散射范数约.0454%/.0498%；两版本均通过，简单参照优势不能主要归因于直接使用全时间标签的表示差异。其训练信息更多：它有−2/0/+2标签，原FNO仅训练名义几何。**这不是公平的架构因果胜负试验，也不能证明网络容量不足。**

![[{TITLE}.png]]

几何响应是P(+1)−P(−1)与F(+1)−F(−1)的有限变化比较，不是局部连续几何Jacobian；没有材料对比度变化、网络AD或材料恢复。参照权重使用已知输入深度，不使用验证真值选取；不是盲定位。

8.649秒仅是本轮生成/预测/参照准备的运行时间，不含历史标签生成或网络训练，不能视为完整反演成本收益。新两点在此轮审查后成为已用开发数据，不称独立最终测试。

## 3. 三个竞争解释及反证更新

| 解释 | 支持证据 | 冲突/限制 | 若成立应有的表现；如何削弱 |
| --- | --- | --- | --- |
| H1 缺少几何训练约束导致网络响应错误 | 网络在±1仍>32%接收、响应余弦负；有三深度信息的线性参照可靠 | 信息量/方法不同，不能把差异直接归因覆盖；尚无网络覆盖干预 | 同初权重/更新预算，覆盖臂应优于名义重复臂并改善响应；若两臂等同，或覆盖臂训练拟合仍失败，覆盖解释受到限制 |
| H2 两行间隔中的波形过于非线性，简单局部代理也不足 | 最近邻约28%接收，物理有限变化较大 | 线性在两侧约6.5%、响应14.84%通过，明显削弱“本局部范围无法简单表达” | 本范围的强版本被削弱；更远距离/更换材料可能再次失败，但没有证据外推。更密状态仍使线性失败会限制本结论 |
| H3 时间重建或标签对齐造成主要误差 | 原网络确实采用快照重建 | 上轮快照响应已错，物理重建误差.074%；本轮两种线性表示均通过，坐标检查/重放一致 | 主要原因解释继续被削弱；若后续发现输入/源/坐标错误，必须撤回当前解释并先修复 |

不增加“网络太小”“有限孔径”作为未经验证的新结论。上一轮固定几何单参数恢复及成本判断保留；本轮没有改变其评价范围。

## 4. 对继续与修改计划的反方审查

继续当前插值路线：已有结果足以把训练深度线性列为局部简单基线，无需再生成相同点或调权重。但只在a=.5、一个形状、一个源通过，不能直接用于未知几何材料反演，尤其材料敏感度尚未验证。其优势含额外几何训练信息；不能据此宣布神经算子不需要研究。

修改为配对覆盖干预：两臂从同一checkpoint开始，均新建Adam、300步、a=.5、同损失与尺度/精度。唯一处理是训练几何分布，名义重复臂排除“只是多训练”。总曝光相等而名义曝光不同正是干预内容，不能称完全逐样本匹配。验证±1已经看过，只能用于机制诊断；不得根据它选早停或改门槛。即使覆盖臂通过，也只是局部学习响应修复，仍须独立比较线性成本、材料方向导数与后续恢复。

已有数据没有“网络受到覆盖干预”的结果，不能从线性胜出替代该因果判断。额外变量如网络宽度/层数、多个材料/源、损失改动均不在下一轮加入。

## 5. 最多三个候选实验

| 候选 | 要区分；固定/唯一变化 | 简单基线 | 预计成本 | 成功、失败、无法判定及后续 |
| --- | --- | --- | --- | --- |
| A 配对300步深度覆盖（唯一优先） | H1与额外优化混淆；同初权重、a=.5、Adam、损失和预算，仅训练几何分布不同 | 名义重复训练、冻结网络、线性两版本、最近邻/零散射 | 总600步，预计4–6分钟，上限8分钟，无新增物理标签 | 覆盖臂两点场/接收和响应均优于重复臂且过固定筛查：支持局部覆盖作用；两臂同过：额外优化可解释。部分改善：只报部分效应。非有限/哈希/初值检查失败：无法判 |
| B 原网络再缩小深度幅度 | 检查尺度限制，但小于1格需改变离散支撑表示 | 静态/线性 | 数次正演，但需新几何参数化 | 会同时引入表示变化，当前不是最小单因素；不能替代覆盖因果审查 |
| C 线性基线材料敏感度审查 | 转向材料方向可靠性；需新增移位几何的对比度标签 | 物理差分及几何插值 | 多个正演，尚未预注册 | 能检查材料方向，但扩大当前轴，不能回答网络覆盖问题；留在局部覆盖审查之后 |

不同时运行B/C。A不改变网络复杂度，且有已验证简单参照。

## 6. 唯一优先A的协议（已保存未运行）

1. 新目录`.build/fno-paired-depth-coverage-control-20261009`，先冻结协议、checkpoint、数据与依赖SHA。两臂原width8初权重及update0预测逐位相同，新建Adam lr=.001且均不沿用旧动量。
2. 使用D已有−2/0/+2的a=.5物理标签。重复臂三槽均0；覆盖臂槽序−2/0/+2，均循环0/1/2，各300步/每槽100次。已有±1观测仅评估，禁止参加训练。
3. 沿用训练field+receiver损失、原训练field/trace尺度、IEEE、源0与同一网络结构。评估固定0/150/300，主结果final300；无最佳轮选择、验证早停或阈值回调。
4. 所有状态完整/散射、场/接收分别报；±1有限几何响应报误差/范数比/余弦。记录全部更新、损失、梯度有限性、曝光、哈希、时间与显存，终态重载复核。原名义a=.3/.7在0和300只作遗忘诊断，不参与训练或停止规则。
5. 覆盖效果需final300覆盖臂在两点的散射场/接收及响应误差均优于重复臂，并通过两点20%、响应20%/cos≥.9筛查。改善但不达标仅是部分效应，不自动加轮数。
6. 神经实用收益另设门槛：与冻结线性两版本比较误差、响应及已核算准备/训练/推理成本。覆盖效应通过不等于神经优越，也不等于反演成本收益。
7. 预算共600步，估计4–6分钟，上限8分钟；OOM/非有限即停止，不能静默减网格/模态/精度。无下载、标签重生。终态重载核验允许另行进行，旧结果保留。

### 改变建议的证据

覆盖臂明显优于重复臂并修复响应，会支持局部覆盖解释；两臂同样修复，会把重点转为优化/名义拟合；覆盖臂训练好但验证响应差，会转向表示与泛化；两臂训练都差，先审查固定预算、损失与优化，不直接扩网。若任何参照/初值/哈希核验失效，先撤回本轮比较。即使网络最终修复，线性仍是必须保留的参照，材料恢复与同质量完整成本仍未知。

全部评价边界保留：完整≠散射、全场≠接收、训练拟合≠泛化、有限变化≠局部Jacobian、网络AD正确≠物理导数正确、波形拟合≠材料恢复、推理时间≠反演总成本。已知输入几何任务不是盲缺陷识别；已用开发数据不是独立最终测试。
'''
dest=V/'05实践与复习'/f'{TITLE}.md';assert not dest.exists();dest.write_text(note,encoding='utf-8')
targets=[V/'00导航'/'FNO与超声全波形反演科研计划 2026-10-07.md',V/'00导航'/'相控阵FWI与神经算子后续执行路线 2026-10-08.md',V/'04专题'/'相控阵FWI与神经算子阶段总结及知识梳理 2026-10-08.md',V/'05实践与复习'/'冻结FNO深度平移审查与几何响应 2026-10-09.md']
for p in targets:
 lines=p.read_text(encoding='utf-8').splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('# '))
 lines[i+1:i+1]=['', '> [!important] 深度插值新结果（2026-10-09）',f'> [[{TITLE}]]：新±1行位置线性散射接收约6.57%/6.50%、响应14.84%，两表示均过预定筛查；冻结网络仍失败。唯一下一步为同初权重、同300步的配对覆盖控制，协议已保存未运行。','']
 for i,l in enumerate(lines):
  if l.startswith('status:'):lines[i]='status: 深度插值基线已复核，下一步配对几何覆盖控制';break
 p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
names={p.stem for p in V.rglob('*') if p.is_file()}|{p.name for p in V.rglob('*') if p.is_file()}
for p in [dest,*targets]:
 s=p.read_text(encoding='utf-8');assert s.startswith('---\n') and s.count('```')%2==0
 for l in re.findall(r'\[\[([^\]]+)\]\]',s):assert l.split('|')[0].split('#')[0] in names,(p,l)
assert sha(OUT/'depth_interpolation_errors.png')==sha(V/'90附件'/f'{TITLE}.png')
assert not Path(plan['output_directory']).exists()
for p,h in plan['frozen_sha256'].items():assert sha(Path(p))==h
(OUT/'delivery_verification.json').write_text(json.dumps({'status':'passed','numeric_tables':'generated from verified machine summary','wikilinks':'all5 edited notes resolved','image':'copy SHA identical','next_plan':'planned not run, hashes checked','native_Obsidian_rendering':'not performed','files_sha256':{str(p):sha(p) for p in [dest,*targets]}},ensure_ascii=False,indent=2),encoding='utf-8')
print(dest.resolve())
