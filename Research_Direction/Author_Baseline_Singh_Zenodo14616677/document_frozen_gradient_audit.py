"""Document frozen-gradient evidence and preregister matched-exposure replay."""
import json, hashlib, re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('D:/下载/FNO_Research_Workspace')
OUT=BASE/'.build/fno-frozen-loss-gradient-audit-20261009'
PREV=BASE/'.build/fno-paired-depth-coverage-recovery-r2-20261009'
VAULT=BASE/'PINO超声成像知识库'
TITLE='冻结梯度审查与材料锚点回放协议 2026-10-09'
NOTE=VAULT/'05实践与复习'/f'{TITLE}.md'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
assert not NOTE.exists()
s=read(OUT/'summary.json');assert read(OUT/'verification.json')['status']=='passed'
plan={
 'status':'planned_not_run','experiment':'matched_geometry_exposure_material_anchor_replay',
 'output_directory':str(BASE/'.build/fno-matched-material-replay-20261009'),
 'question':'Does replay of previously seen nominal material anchors preserve material finite response without losing geometry response?',
 'initial_checkpoint':str(PREV/'depth_coverage_update300.pt'),
 'initialization':'Both arms restore identical model and Adam state at step300;lr0.001,IEEE,no scheduling,identical architecture/loss/scales/source.',
 'arms':{'nominal_control':['depth-2','depth0','depth2','depth0','depth0'],
         'material_replay':['depth-2','depth0','depth2','contrast03','contrast07']},
 'schedule':'Repeat each five-slot order20 times;100 additional updates per arm;the first three geometry slots receive20 exposures each in both arms. Only last two slot labels differ. No geometric augmentation.',
 'single_factor':'Composition of40 replay slots: nominala0.5 vs nominala0.3/a0.7,20 each. Intentional extra nominal exposure in control;claim is replay-composition effect,not equal total nominal counts.',
 'loss':'Unchanged field MSE plus receiver MSE;zero encoding,8input channels,width8,source0,same500frames and24 receivers,old normalization.',
 'budget':{'training_updates':200,'new_physical_calls':0,'wall_seconds_cap':240,'checkpoint_evaluation_updates':[0,100],
           'verification':'Reload both finals and replay all7 stored states once;no extra training;log costs separately;total including verification within240sec or interrupted_not_comparable.'},
 'evaluation_states':['depth-2','depth0','depth2','depth-1','depth1','contrast03','contrast07'],
 'baselines':['frozen coverage300 no update','original checkpoint cached results','nominal_control matched steps/exposure','existing3depth piecewise-linear geometry reference'],
 'primary_material':'Nominal receiver finite response(a0.7-a0.3);report subintervals0.3to0.5 and0.5to0.7 as secondary;also report anchor scatter field/trace and full field/trace.',
 'success':'Replay primary finite-response relativeL2 <=0.8*control AND <=0.8*frozen coverage;neither anchor scatter trace/field >1.1*frozen coverage. On both development depths scatter field/trace <=1.1*frozen coverage,trace <=0.20;geometry response relativeL2 <=0.20 andcos>=0.9. Not proxy qualification unless inherited field<=0.20 also passes.',
 'failure':'Primary material response does not improve over control,or geometry guardrail violated. Report partial material benefit separately if geometry harms;do not redefine success.',
 'inconclusive':'Valid improvements smaller than20%,or contradictory subintervals,or nonfinite/failure/incomplete budget. One initialization limits robustness;no winner from intermediate checkpoint.',
 'threshold_origin':'Inherited20%/.9 proxy screens unchanged.20% relative improvement and10% degradation are new preregistered operational replay effect/guardrail thresholds,not statistical significance or physical guarantees.',
 'branches':{'success':'Next preregister small physical material-Jacobian check on fixed support;keep simple interpolation baseline. No blind recovery claim.',
             'material_gain_geometry_harm':'Replay trade-off;next isolate common loss with same data exposure,not add network.',
             'no_material_gain':'Audit actual frozen Adam update directions including moments before increasing steps/capacity.',
             'incomplete':'Preserve failures and timings;resume only via new frozen recovery protocol.'},
 'scope':'All states already used for development. Nominal material anchors previously trained;neither unseen material/geometry joint generalization nor independent final test. Finite response is not localJacobian. No final truth stopping.',
 'frozen_sha256':{str(p):sha(p) for p in [PREV/'depth_coverage_update300.pt',OUT/'summary.json',OUT/'raw.pt',OUT/'verification.json',PREV/'cached_nominal_material_response.json',BASE/'Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_paired_depth_coverage.py']}
}
assert not (OUT/'next_experiment_protocol.json').exists();dump(OUT/'next_experiment_protocol.json',plan)
fig,ax=plt.subplots(figsize=(7.5,3.5))
labels=['depth -2','depth 0','depth +2','material .3','material .7'];x=list(range(5))
for offset,name in [(-.18,'original'),(.18,'coverage')]:
 ax.bar([i+offset for i in x],[r['field_receiver']['cosine'] for r in s['models'][name]['rows']],width=.36,label=name)
ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,labels);ax.set_ylabel('Field / receiver gradient cosine');ax.set_ylim(-1,1);ax.legend();fig.tight_layout()
plot=VAULT/'90附件'/f'{TITLE}.png';fig.savefig(plot,dpi=160);plt.close(fig)
rows='\n'.join('| '+name+' | '+r['state']+f" | {r['field_receiver']['cosine']:.6f} | {r['field_receiver']['second_over_first']:.3f} | {r['total_field']['cosine']:.6f} | {100*r['regions']['interior']['fraction_total_error']:.2f}% |" for name,m in s['models'].items() for r in m['rows'])
text=f'''---
type: evidence-review
date: 2026-10-09
status: 冻结梯度已完成并核验；材料回放配对协议已冻结未运行
tags: [FNO, FWI, 参数梯度, 材料映射, 科研审查]
---
# 冻结梯度审查与材料锚点回放协议

**瓶颈判断：覆盖训练后的代理存在局部几何目标与材料锚点冲突，但全场不佳不能归结为接收梯度压倒全场；优先做曝光匹配的材料回放控制。** 数值事实置信度高；对遗忘机制置信度中等，对真实Adam训练因果置信度低，尚未测量。

承接 [[配对深度覆盖终态审查 2026-10-09]]。实际根目录为`D:\\下载\\FNO_Research_Workspace`。以下A=`.build/fno-frozen-loss-gradient-audit-20261009`，P=`.build/fno-paired-depth-coverage-recovery-r2-20261009`，I=`.build/fno-depth-interpolation-control-20261009`；所有路径均相对D根目录。执行代码`Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_frozen_gradient_audit.py`。

## 1. 事实、解释和未知

| 类别 | 判断 | 核查来源 |
| --- | --- | --- |
| 完成 | 两冻结模型、各5状态、10正演/20参数反传、0更新/0物理正演；实测{s['elapsed_seconds']:.3f}秒，模型前后SHA相同 | A/protocol.json、raw.pt、summary.json、verification.json |
| 完成 | 原模型几何平均总梯度与材料平均总梯度余弦+0.113343；覆盖模型−0.338976。覆盖模型负几何梯度的材料目标方向导数+0.000136554 | A/summary.json geometry_material；实坐标欧氏内积 |
| 完成 | 覆盖端点场/接收余弦−0.582719/−0.284950，但接收/场范数仅0.255/0.221；合计下降方向与场梯度余弦0.971698/0.975410 | A/summary.json rows |
| 完成 | 覆盖三几何场平均梯度范数/个体平均范数为0.422767，原模型0.981986；有跨状态抵消，不等于无下降方向 | A/summary.json geometry_field_mean_over_mean_individual_norm |
| 完成 | 覆盖深度端点内域占场误差能量87.71%/88.86%；远侧边界占8.71%/9.47%，剩余阵列侧3.58%/1.66% | A/summary.json regions；内域占网格96.875%，不能用能量占比推出单位面积误差集中 |
| 完成 | 覆盖名义材料宽区间有限接收响应误差30.297%，原模型4.621%；接收改善伴随材料映射退化 | P/cached_nominal_material_response.json；都是已用数据 |
| 合理待验 | 原材料监督缺失可能导致遗忘/目标取舍；原始欧氏几何下降方向在覆盖终态局部伤害材料平均目标 | 当前梯度与旧有限响应相符，但不是训练全程或Adam实际步的因果证明 |
| 未知 | Adam方向、有限步效果、梯度冲突沿轨迹是否持续、未见深度的材料局部Jacobian、材料恢复及总成本 | 未执行对应测量，不补猜 |
| 未执行 | 匹配几何曝光的材料锚点回放 | A/next_experiment_protocol.json status=planned_not_run |

全场/接收参数梯度的冻结测量：

| 模型 | 状态 | 场/接收余弦 | 接收/场范数 | 总/场余弦 | 内域占误差能量 |
| --- | --- | ---: | ---: | ---: | ---: |
{rows}

![[{TITLE}.png]]

核验包括保存Gram的精确重算、有限性、对称性、半正定容差、区域能量分解、依赖SHA及模型权重不变；**未做独立参数有限差分或物理材料导数核验**。复数权重实部/虚部完整计入16,777,737个实坐标。Gram是参数梯度统计，绝不能称物理Jacobian。

## 2. 三个竞争解释及反证

| 解释 | 支持 | 冲突/不足 | 若成立应观察 | 削弱/否定条件 |
| --- | --- | --- | --- | --- |
| H1 材料监督缺失造成可回放修复的映射遗忘 | 材料有限响应由4.62%退化到30.30%；终态几何/材料余弦−0.339 | 原终态余弦+0.113；只有两时间点；名义重复也退化，不能全归几何覆盖 | 相同几何曝光/步数下，材料回放比a=.5回放显著改善有限响应并保留几何表现 | 匹配回放无收益，或收益只能牺牲几何获得；此时可修复假设被削弱，不能一并否定所有遗忘解释 |
| H2 接收监督主导或冲突导致场失败 | 覆盖两端场/接收梯度为负 | 接收范数仅约场的四分之一；总梯度仍强烈指向场下降；原depth+2两者为正 | 去掉接收监督应在同曝光下改善场，并暴露接收代价 | 原始合计梯度并未反向场，削弱“接收压倒场”；实际Adam方向/单因素损失实验仍可能发现局部取舍 |
| H3 共享表示/优化难以同时拟合不同深度全场 | 平均场梯度抵消增强；端点场仍约47%；同标签线性场约6.3% | 场梯度非零、总方向仍可降场；没有独立优化或容量对照，不能称网络太小 | 材料回放无用后，实际优化方向或单一损失对照发现持续抵消/下降失效 | 不改结构即可同时改善场、几何和材料会削弱表示限制；仅多训练有效更支持未收敛解释 |

相对上一轮：H1获得局部支持；H2的“梯度主导”版本削弱、局部冲突版本保留；H3仍无法区分优化与表示。有限孔径不是本轮已证明原因。

## 3. 对建议的反方审查

继续只用a=.5训练可以降低原目标，但已有名义重复与覆盖结果显示材料退化；不应直接加轮数。全场梯度非零也说明原计划不是完全没有下降可能，只是收益未知。

修改为材料回放有依据，但不能保证修复物理敏感度。更简单的已存三深度线性插值已优于网络，是必须保留的参照；当前梯度审查已排除“无需新实验即可证明回放有效”的说法，因为它只有局部一阶信息。回放与对照使用同初权重、同Adam状态、相同几何槽和总步数，仅40回放槽标签不同。对照多40次名义a=.5曝光，结论只能归于回放组成，不能声称总名义曝光完全相同。

锚点a=.3/.7是原训练材料且几何已知，属于额外监督干预，不偷用新测试标签；这些锚点只能衡量记忆维护。成功最多支持这组已用几何与名义材料有限响应的取舍改善，不能证明跨材料泛化、物理局部Jacobian、盲缺陷识别、材料恢复或同质量总成本收益。

## 4. 候选最小实验比较

| 候选 | 区分假设 | 固定与唯一变化 | 简单基线 | 预计成本 | 成功/失败/无法判定与后续 |
| --- | --- | --- | --- | --- | --- |
| **材料锚点回放配对，优先** | H1是否可修复，与H3相比 | 同覆盖300起点/Adam/损失/结构/100步/60几何槽；40回放槽a=.5对a=.3/.7 | 不更新覆盖模型、原模型、匹配a=.5回放、已存线性几何插值 | 200更新，0新物理；约120–180秒训练评估，合计硬上限240秒 | 材料改善且几何保留→局部物理导数审查；材料改善但几何损失→目标取舍；无收益→实际Adam方向审查；不足20%或故障→不定 |
| 固定覆盖数据，场损失对场+接收 | H2局部冲突/主导与H3 | 同起点/Adam/数据曝光/步数，仅接收权重1对0 | 原损失与不更新模型、线性插值 | 约200更新/0物理，需另冻协议 | 场改善接收退化→取舍；两者均改善→原损失优化问题；无改善→削弱H2；不直接解材料退化 |
| 冻结实际Adam候选步方向审查 | H3优化及H1实际步 | 冻结模型/数据，比较已有moments预条件方向与原始梯度，不写回更新 | 零步和欧氏负梯度 | 约5–10正演/反传、0更新/0物理，约20–40秒 | Adam反向场/材料→优化解释；同向→局部阻碍弱；不能回答有限100步回放是否有效 |

第三项更便宜，但其一阶答案不能消除当前最关键的“材料回放能否兼顾几何”不确定性；因此此轮只推荐第一项。其余均未执行，不自动串行开展。

## 5. 唯一优先实验协议

完整机器协议：[next_experiment_protocol.json]({(OUT/'next_experiment_protocol.json').as_posix()})。本轮**仅冻结协议，未开始训练**；当前冻结审查预算0更新已经遵守。

1. 新目录`.build/fno-matched-material-replay-20261009`；不得覆盖旧结果。两臂逐位加载覆盖300的权重和Adam状态，检查初始预测、optimizer step300、输入依赖SHA；继续lr=.001，不改结构/损失/尺度/IEEE/source0。
2. 每臂重复五槽20次：对照[depth−2,depth0,depth+2,depth0,depth0]；回放[depth−2,depth0,depth+2,a=.3,a=.7]。每臂100步；前3几何槽各20曝光，最后40槽唯一干预。
3. 固定0/100评估，final100唯一主结果；不可按材料真值择优或追加步数。七已存状态全场/接收、完整/散射分别报告，名义材料两子区间及宽区间有限响应、几何±1有限响应分别报告；不新增物理标签。
4. 材料主指标为名义(.7−.3)接收有限响应相对误差，需比对照和不更新覆盖分别至少降低20%；两锚点散射场/接收不能比覆盖恶化>10%。开发±1散射场/接收不比覆盖恶化>10%，接收≤20%，几何响应≤20%、余弦≥.9。原代理合格条件场≤20%仍保留，若场不达标，只称回放有收益。
5. 新20%效应量与10%守门是前瞻操作阈值，非统计显著性；已有代理阈值不变。材料无改善或几何越界判失败，较小改善/子区间矛盾判不定，保留全部指标而不倒调规则。
6. 预算200新更新、0新物理正演，预计训练及评估120–180秒，含回读核验总上限240秒；故障保留中断权重/日志、计实际失败成本，不以部分臂选赢家。CPU临时安全存档，终态重载七状态核验；训练损失、曝光、优化步、墙时、峰显存均记录。

## 6. 会改变建议的新证据

若发现当前参数梯度计算错误或标签/状态不匹配，先修复审查而不运行回放；若严格曝光对照下回放无材料收益，转向实际Adam方向审查；若回放材料与几何兼得，才考虑固定已知支持的局部物理材料导数检查。若仅改善波形而材料有限响应未改善，不推进材料恢复主张。若同质量物理参照与全训练/数据/核验成本没有完整账本，仍不宣称神经方案总成本占优。

评价边界：完整波形≠散射波形；全场≠接收；单材料拟合≠未见材料泛化；有限材料差分≠局部Jacobian；自身导数正确≠物理导数正确；波形失配下降≠材料恢复；单次速度≠反演总成本。开发数据不称最终独立测试，已知支持单参数恢复不称盲识别。
'''
NOTE.write_text(text,encoding='utf-8')
targets=[VAULT/'00导航/FNO与超声全波形反演科研计划 2026-10-07.md',VAULT/'00导航/相控阵FWI与神经算子后续执行路线 2026-10-08.md',VAULT/'04专题/相控阵FWI与神经算子阶段总结及知识梳理 2026-10-08.md',VAULT/'05实践与复习/配对深度覆盖终态审查 2026-10-09.md']
callout=f'\n> [!important] 最新冻结梯度审查已核验（2026-10-09）\n> [[{TITLE}]]：覆盖终态几何/材料梯度余弦−0.339；接收梯度压倒全场的解释削弱。下一步只优先材料锚点回放配对，协议已冻结未运行；不加网络或自动追加训练。\n\n'
for p in targets:
 t=p.read_text(encoding='utf-8');t=re.sub(r'^status:.*$', 'status: 冻结梯度已核验，材料回放配对协议待执行',t,count=1,flags=re.M)
 t=re.sub(r'(^# [^\n]+\n)',lambda m:m[0]+callout,t,count=1,flags=re.M)
 p.write_text(t,encoding='utf-8')
dump(OUT/'documentation_manifest.json',{'note':str(NOTE),'plot':str(plot),'next_protocol_sha256':sha(OUT/'next_experiment_protocol.json'),'changed_navigation':[str(p) for p in targets],'training_executed':False})
print('Documented and preregistered:',NOTE)
