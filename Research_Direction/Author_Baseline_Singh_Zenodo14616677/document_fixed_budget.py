"""Document verified fixed-budget results; preregister only the next audit."""
import json, shutil, hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_reduced_material_scan import sha

OUT=Path('.build/fno-physical-fixed-budget-control-20261009')
VAULT=Path('PINO超声成像知识库')
TITLE='固定八步物理更新与初始化收益审查 2026-10-09'
r=json.loads((OUT/'summary.json').read_text())
v=json.loads((OUT/'verification.json').read_text())
assert v['status']=='passed'
plan={
 'status':'planned_not_run',
 'question':'Does the frozen contrast-only width8 forward surrogate respond correctly to a small unseen depth translation?',
 'output_directory':'.build/fno-frozen-depth-shift-audit-20261009',
 'only_main_factor':'Translate the nominal discrete support along physical y by -2,0,+2 grid rows; contrast remains .5.',
 'fixed':'Frozen width8 checkpoint, IEEE policy, source0, 24 receivers, 2000 times, original float32 discrete physics, normalization, time reconstruction and homogeneous background; no optimization, training or calibration.',
 'support_checks':'449 cells for each mask; translate with zero fill without roll/wrap; every cell inside the physical domain; save masks and actual shift in meters from y_m spacing.',
 'data_role':'Prospective frozen diagnostic validation; immediately becomes development data after review. Never an independent final test.',
 'baselines':['Frozen nominal .5 training field/trace (geometry-insensitive lookup)','Zero scattered field/trace plus homogeneous background','Nominal zero-shift replay as implementation control'],
 'observations':'Generate three float32 physical field-snapshot/receiver labels after saving protocol and hashes. No shifted label may become a reference prediction.',
 'metrics':['full and scattered field relative L2 separately','full and scattered receiver relative L2 separately','finite symmetric geometry response P(+2)-P(-2) versus F(+2)-F(-2): relative L2 and cosine','network/lookup per-case error ratios, source and receiver coordinate checks'],
 'screening':'Predeclare shifted scatter field and receiver error <=.20 for both signs, finite geometry-response receiver relative error <=.20 and cosine >=.90. These are engineering screening rules, not a claim of physical derivative or inversion accuracy. No threshold tuning from new labels.',
 'success':'Both signs meet screens, finite response meets screen, and network beats nominal lookup scatter receiver error on both shifted states; supports only this discrete local depth transfer.',
 'failure':'A valid run fails any screen: record the failing metric and sign; next examine geometry-family coverage with the same architecture, not automatic network growth.',
 'inconclusive':'Mask/control/hash checks fail, nonfinite outputs, or finite physical response norm is <=1e-6 times nominal scattered trace norm; stop interpretation and audit implementation/response scale.',
 'next_if_success':'Preregister a depth-family derivative/data protocol before any unknown-geometry material inversion. A three-state finite response is not a local Jacobian.',
 'budget':'Exactly3 physical and3 network predictions; estimated <2min, cap5min, no training/download. If nominal replay requires repair or resource cap is exceeded, stop and preserve artifacts.',
 'boundary':'Geometry is supplied as forward input. No blind localization, unknown-shape recovery, noise robustness or full-inversion cost claim.',
 'frozen_evidence_sha256':{str(p):sha(p) for p in [OUT/'raw.pt',OUT/'summary.json',OUT/'verification.json',OUT/'protocol.json']}
}
assert not (OUT/'next_depth_shift_protocol.json').exists()
(OUT/'next_depth_shift_protocol.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')

fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
colors={'width8':'#2171b5','piecewise_linear':'#238b45','fixed05':'#756bb1'}
for ax,t in zip(axes.flat,[.35,.45,.55,.65]):
 for route in r['routes']:
  if abs(route['target']-t)<1e-6:
   ax.plot([x['update'] for x in route['records']],[max(x['contrast_error'],1e-10) for x in route['records']],marker='o',label=route['method'],color=colors[route['method']])
 ax.set(yscale='log',xlabel='Fixed physical update',ylabel='Absolute contrast error',title=f'Development contrast {t:.2f}',xticks=range(9))
 ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(OUT/'fixed_budget_trajectories.png',dpi=160)
shutil.copy2(OUT/'fixed_budget_trajectories.png',VAULT/'90附件'/f'{TITLE}.png')

rows=[]
for t in [.35,.45,.55,.65]:
 routes=[x for x in r['routes'] if abs(x['target']-t)<1e-6]
 rows.append('| '+f'{t:.2f}'+' | '+' | '.join(f"{x['records'][8]['contrast_error']:.3e}" for x in routes)+' |')
costrows=[]
for m,t in r['method_totals'].items():
 costrows.append(f"| {m} | {t['physical_calls']} | {t['physical_route_seconds']:.3f} | {t['historical_first_stage_grid_seconds']:.3f} | {t['descriptive_online_seconds_sum']:.3f} |")
increases=[]
for x in r['routes']:
 for i in x['cost_increase_updates']:
  before=x['records'][i-1]['cost'];after=x['records'][i]['cost']
  increases.append({'method':x['method'],'target':x['target'],'update':i,'absolute_increase':after-before,'previous_cost':before})
(OUT/'cost_increase_audit.json').write_text(json.dumps(increases,indent=2),encoding='utf-8')
maxinc=max(x['absolute_increase'] for x in increases)
note=f'''---
type: evidence-review
date: 2026-10-09
status: 八步实验完成并复核，深度平移审查已预注册未运行
tags: [FWI, FNO, 材料恢复, 成本, 证据审查]
---
# 固定八步物理更新与初始化收益审查

**当前瓶颈：已知几何的一维对比度任务尚未证明神经代理带来额外材料质量或同质量成本收益，最关键的未解问题转为冻结代理是否真的响应未见几何。** 对前半句置信度高（限本轮四个开发材料），对几何失效机制的判断置信度低，尚待最小控制实验。

承接 [[代理初值与一次物理材料更新 2026-10-09]]。本轮结果目录（项目根相对路径）R=`.build/fno-physical-fixed-budget-control-20261009`。机器来源为R/protocol.json、raw.pt、summary.json、verification.json、cost_increase_audit.json；执行代码`Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_physical_fixed_budget.py`。旧目录及原作者代码均未覆盖。

## 1. 当前事实、解释与未知

| 类别 | 可核查判断 | 来源 |
| --- | --- | --- |
| 已完成 | 四材料×三初值各执行恰好8步，共96次更新；无真值早停、损失停止或最佳轮选择 | R/protocol.json、raw.pt、verification.json |
| 已完成 | 网络与线性支路第8步均到达相同数值解，四点最大支路间材料差约浮点舍入量级；没有最终质量优势 | R/summary.json的全部12条轨迹 |
| 已完成 | 固定.5在.45／.55达到同样数值解；.35／.65仍有约1.13e−4／6.40e−5误差，前7步均裁剪 | R/summary.json |
| 已完成 | 网络/线性第2步材料绝对误差均小于3.7e−8；这是预定检查点的描述，不能倒选第2步为停止规则 | R/summary.json records[2] |
| 已完成 | 逻辑物理调用384，实际227，运行总时间113.154秒；缓存限各自支路，未跨方法共享 | R/raw.pt、summary.json |
| 已完成 | 96个更新公式与缓存指标精确重算，12个最终物理预测逐位重生一致，首步与上一轮一致，依赖SHA未变 | R/verification.json |
| 合理解释 | 直接.5远端尚未收敛主要与±.02步长及8步预算相容，不能称不可辨识 | 7次裁剪及第8步仍在改善支持；未单独干预步长，故非唯一因果证明 |
| 合理解释 | 收敛后的微小成本上升与数值平台相容 | R/cost_increase_audit.json；最大绝对增量{maxinc:.3e}，未追踪底层数值来源 |
| 未知 | 正式同质量完整成本、数据生成成本、训练摊销拐点、随机种子/硬件重复计时 | 缺少相应测量；不补猜 |
| 未知 | 当前width8未见深度、形状、源位置泛化及物理材料Jacobian在新几何的正确性 | 现有固定几何结果不能回答 |
| 未执行 | 冻结width8，固定a=.5，仅深度±2行的三状态正演审查 | R/next_depth_shift_protocol.json，planned_not_run |

第8步绝对对比度误差（目标使用保存的实际float32值提升，不是理想十进制数）：

| 目标a | 网络初值＋物理 | 训练线性初值＋物理 | 固定.5＋物理 |
| --- | ---: | ---: | ---: |
{chr(10).join(rows)}

两代理最终完整接收相对误差约1.66e−6至4.07e−6，散射接收相对误差约5.05e−6至1.12e−5；二者分母不同，不能混报。非零误差与float32观测/提升float64参考的差异相容，但“精度差异是唯一原因”未被隔离验证。

![[{TITLE}.png]]

图只为对数显示把0截到1e−10；机器记录保留真实0。全9个状态保留，平台处出现微小成本上升，未删除或只报下降迭代。

### 成本核算范围

| 方法 | 实际物理调用 | 四支路物理秒数 | 历史201点初值搜索秒数 | 描述性在线相加秒数 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(costrows)}

网络另外已有训练及四次评估532.994秒；训练数据生成、线性参考生成耗时均未知。初值搜索耗时来自`.build/fno-frozen-new-contrast-validation-20261009/summary.json`，与本轮分支时间相加只是描述性账目，不是同一执行批次的独立完整流程基准。两代理四材料最终质量相同，本轮线性实际调用更少、描述性在线时间更低，**当前没有神经成本优势证据**。固定.5远端质量较低，不能将其时间直接当作同质量获胜。

## 2. 三个竞争解释及反证

| 具体解释 | 支持证据 | 冲突/限制 | 若成立应观察到什么；何种结果削弱它 |
| --- | --- | --- | --- |
| H1：网络在这一维任务主要提供可由训练插值替代的初值，物理更新才消除残差偏差 | 两代理最终一致；线性更便宜；此前残差投影能解释初值偏差 | 网络第0/1步部分点更好；未排除更难任务上的价值 | 本轮获支持，神经必要性被削弱。冻结未见几何下网络对真实响应明显优于固定参考，会限定/削弱“仅为插值”的外推 |
| H2：固定.5远端残差反映步长/预算限制，而非这个标量的物理不可辨识 | 7次限步，逐步接近；中间两点已达相同解；物理网格亦命中真值 | 未单独改变步长或预算；不能断言更多步必然可靠 | 当前获支持。若独立预注册更充分预算后仍远离目标且响应退化，或出现多个等价低失配解，将削弱它；本轮不自动加步 |
| H3：此前跨几何失败源于代理对局部几何变化的响应不可靠，固定几何成绩掩盖了这一点 | `.build/basic-fno-twelve-materials-20261008/comparison.json`跨几何开发散射接收误差约131.9%；当前训练仅固定支撑 | 旧网络/训练集不同，不能把它归因给当前width8；尚无当前模型小幅深度结果 | 若冻结a=.5的±2行深度响应误差大/方向错，支持局部几何响应问题；若两方向均达预定筛查且优于静态参考，削弱这一局部解释，但不否定大范围形状失效 |

未知机制保持未知；不把“有限孔径”“样本少”“网络小”直接当结论。

## 3. 对继续/修改计划的反方审查

继续原一维校正路线：八步结果已回答代理校正与简单线性能否恢复相同材料，进一步校准偏置或凭真值加步容易围绕已用数据优化。纯成本问题先可从本轮账目得到“未显示神经收益”，无需新增训练。若目标是正式部署成本，后续必须统一独立计时和损失停止规则，不能把本轮第2步达标倒设为规则。

修改为小幅深度审查：只改变支撑沿y的位移，保留对比度、源、接收、模型及归一化；这比同时扩展形状/多源/材料更可归因。最简单参照必须包括训练a=.5的静态参考和零散射。几何由正演输入提供，属于标准已知材料输入，不能称盲定位；不使用移位标签训练或修正预测。只比较波形并不能证明材料恢复，本轮下一步目的仅是检查代理对几何的有限响应。旧跨几何结果来自另一训练配置，无法代替此次小幅、当前模型的控制；成功也只支持三状态、一个深度尺度、一个源和a=.5。

## 4. 最多三个候选实验

| 候选 | 区分的解释；固定/唯一主要变化 | 简单基线 | 成本 | 成功、失败、无法判定及后续 |
| --- | --- | --- | --- | --- |
| A 冻结深度三状态审查（优先） | H1/H3的局部外推；固定当前模型和a=.5，仅支撑y位移−2/0/+2行 | 名义.5训练参考、零散射、零位移复现 | 3物理＋3网络；预计<2分钟，上限5分钟 | 两方向散射场/接收误差≤20%、有限接收响应误差≤20%且cos≥.9，并均优于静态参考：再预注册几何族导数/数据。任一有效指标失败：检查几何覆盖。实现检查失败/响应过弱：暂不判方法 |
| B 统一从头重复计时三条八步流水线 | 检查描述性成本是否可重复；方法不变，只改变计时为独立统一流水线 | 固定.5与线性均保留 | 初值搜索＋物理约数分钟，需重复；不补训练数据生成未知成本 | 网络同质量全账目更低才支持有限成本收益；否则线性优先。噪声/缓存不统一则无法判；不会解几何问题 |
| C 独立预注册固定.5十步 | H2；其他不变，仅预算8→10；仍禁止真值停止 | 已有三方法8步结果作历史参照 | 四条直接支路≤160逻辑调用，约1分钟 | 更接近现有数值解支持预算解释；未改善则审查优化/响应；数值平台不判失效。信息主要局限于已知一维，不优先 |

**唯一优先为A。B、C均未运行，不能同时展开。** 本轮停止继续在固定几何上倒调偏置、网格或停止轮。

## 5. A的可执行协议（已保存，未运行）

1. 新目录`.build/fno-frozen-depth-shift-audit-20261009`；先冻结checkpoint、训练参考、背景、输入编码、时间重建、离散求解器与协议SHA，再生成标签。原记录不覆盖。
2. 原449格支撑沿物理y轴平移−2、0、+2行，零填充不周期滚动。对比度a=.5、源0、24接收、2000时刻不变；保存实际格距和米制位移，检查格数与边界。
3. 冻结width8与IEEE策略，保持训练归一化；三次原float32物理正演产生全场快照和接收标签，三次网络预测。统一同一背景构造完整与散射量，训练名义.5标签只能作为静态参照。
4. 报全场/接收、完整/散射四类误差，及P(+2)−P(−2)相对于F(+2)−F(−2)的接收有限响应误差与余弦。它是跨离散几何的有限响应，不是局部几何Jacobian。
5. 上表20%/.9是预注册工程筛查，不代表反演精度标准。物理响应范数若≤名义散射范数的1e−6则无法判定；几何、源坐标、零位移复现及哈希检查必须先通过。任何失败不按标签调阈值。
6. 恰好3＋3预测，5分钟预算上限，无训练、下载、材料优化；保存全部状态及核验。成功再设计几何族的导数与数据协议，失败优先同架构的数据覆盖控制，不自动增加网络复杂度。

## 6. 什么证据会改变建议

若发现本轮初值/物理波形或缓存错误，先撤回材料与成本判断并修复核验；若统一完整计时重复后网络在相同质量下优于线性，改为成本验证主线；若冻结深度两侧响应可靠且优于静态参考，局部几何覆盖不足解释会被削弱，再扩大一个预注册几何轴；若失败仅来自背景或时间重建，应先隔离该更简单原因，不直接加几何数据或网络。

评价边界始终保留：完整波形≠散射；全场≠接收；单几何材料插值≠未见材料/几何泛化；有限材料差分≠局部Jacobian；网络AD正确≠物理导数正确；失配下降≠材料恢复；单次推理≠完整反演总成本。已有四观测反复用于开发，不能称独立最终测试。本轮是已知支撑标量恢复，不是盲缺陷识别；下一轮已知输入几何正演也不是盲定位。
'''
dest=VAULT/'05实践与复习'/f'{TITLE}.md'
assert not dest.exists()
dest.write_text(note,encoding='utf-8')

targets=[VAULT/'00导航'/'FNO与超声全波形反演科研计划 2026-10-07.md',VAULT/'00导航'/'相控阵FWI与神经算子后续执行路线 2026-10-08.md',VAULT/'04专题'/'相控阵FWI与神经算子阶段总结及知识梳理 2026-10-08.md',VAULT/'05实践与复习'/'代理初值与一次物理材料更新 2026-10-09.md']
for p in targets:
 s=p.read_text(encoding='utf-8');lines=s.splitlines()
 idx=next(i for i,l in enumerate(lines) if l.startswith('# '))
 block=['', '> [!important] 八步新结果（2026-10-09）',f'> [[{TITLE}]]：三初值均固定8步。网络与线性最终材料一致，当前未显示神经质量/成本优势；固定.5远端仍受预算影响。唯一下一步为冻结a=.5的±2行深度正演审查，已预注册、未运行。旧阶段结果保留。','']
 lines[idx+1:idx+1]=block
 for i,l in enumerate(lines):
  if l.startswith('status:'):
   lines[i]='status: 八步物理对照已复核，下一步冻结深度平移审查';break
 p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(dest.resolve())
