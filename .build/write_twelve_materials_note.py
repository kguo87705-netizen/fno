import json,math
from pathlib import Path
v=Path('PINO超声成像知识库');out=Path('.build/basic-fno-twelve-materials-20261008')
r=json.loads((out/'comparison.json').read_text());m=json.loads((out/'material_response.json').read_text());verification=json.loads((out/'artifact_verification.json').read_text());assert r['status']=='completed' and verification['status']=='passed'
train=r['final_training'];val=r['final_validation'];refs=r['references'];pct=lambda x:f'{100*x:.3f}%'
rows=[]
for label,key in [('平均监督损失','mean_total_objective'),('散射场相对L2','scatter_field_relative_l2'),('散射接收相对L2','scatter_trace_relative_l2'),('完整场相对L2','full_field_relative_l2'),('完整接收相对L2','full_trace_relative_l2')]:
    f=(lambda x:f'{x:.7f}') if key=='mean_total_objective' else pct
    rows.append('| '+label+' | '+' | '.join(f(z[key]) for z in [train,refs['training_mean']['train'],val,refs['training_mean']['validation'],refs['zero']['validation']])+' |')
workrows=[]
for a,b in zip(val['records'],refs['training_mean']['validation']['records']):
    assert a['group_id']==b['group_id'];workrows.append(f"| {a['group_id'][-3:]} | {pct(math.sqrt(a['trace_error']/a['scatter_trace_true']))} | {pct(math.sqrt(b['trace_error']/b['scatter_trace_true']))} |")
regionrows=[]
for label,key in [('远侧两层y=0..1','opposite_boundary'),('内部y=2..125','interior'),('阵列侧两层y=126..127','array_side')]:regionrows.append(f"| {label} | {pct(val['regions'][key]['scatter_field_relative_l2'])} | {pct(refs['training_mean']['validation']['regions'][key]['scatter_field_relative_l2'])} | {pct(m['summary'][key]['relative_l2'])} |")
response=m['summary'];old=json.loads(Path('.build/basic-fno-four-materials-20261008/comparison.json').read_text());oldm=json.loads(Path('.build/basic-fno-four-materials-20261008/material_response.json').read_text())
text=f"""---
type: experiment-record
date: 2026-10-08
status: 固定4800次完成，训练拟合与六材料开发验证分开评价
tags: [FNO, 散射, 材料泛化, 单源, 基础模型]
---
# 普通FNO十二材料单源训练与验证

承接 [[普通FNO四材料单源训练与验证 2026-10-08]]；补充调研见 [[波动算子材料泛化与数据划分补充调研 2026-10-08]]，主线见 [[FNO与超声全波形反演科研计划 2026-10-07]]。

## 预先固定的协议

训练000、001、002、003、008、009、010、011、016、017、018、019，仅源0；验证004、005、012、013、020、021仅源0。按完整工件隔离；训练与验证来自同一生成规则，不称严格分布外测试。旧测试未读取。

八通道普通FNO，附加两通道为零；width4、4层、modes(16,8,64)、padding0；从seed2重训，初值哈希与四材料完全相同。目标FD(gamma)−FD(1)，基底不依赖材料，接收监督权重1，Adam lr0.001。旧12训练工件全部源统计的RMS仍为场0.15921596709548344、接收0.25594272720229144，不用验证归一化或散射目标重新定尺度。

稳定顺序循环400轮，每件400次暴露，共4800次更新。0/1200/2400/4800评价，只使用预定最终4800次主结果；中间评价不用于早停或选模。工件数与总预算相对四材料同时增加，不是样本数孤立因果消融。训练计时{r['elapsed_seconds']:.1f}秒是本机运行记录，未作为相对物理求解器加速证据。

输入逐步生成，避免驻留12份完整输入；训练前首尾工件及完成后全部12件输入与原编码器逐值一致，不改变模型输入。冻结均值只由12训练工件源0的散射目标计算，并保持原样用于验证。

## 固定最终结果

相对L2先合并全部对应工件的能量再开方，不是简单平均各工件百分比。完整2000步接收由无缺陷基底加三次重建的散射得到；监督只在500帧接收点计算。

| 指标 | 训练FNO | 训练均值 | 验证FNO | 验证冻结训练均值 | 验证零散射 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

训练监督损失为共享均值最优参照的{train['mean_total_objective']/refs['training_mean']['train']['mean_total_objective']:.4f}倍。训练拟合与验证必须分别判断，不能凭训练损失下降认定材料算子泛化。完整波形百分比有共同基底贡献，不能替代散射误差。

| 验证工件，源0 | FNO散射接收相对L2 | 冻结训练均值 |
| --- | ---: | ---: |
{chr(10).join(workrows)}

![[普通FNO十二材料单源训练与验证.png]]

图中曲线仅连接四个实际评价点，不表示中间连续实测轨迹。图示验证004、接收12不是筛选出的最好工件，而是预定首个验证例。

## 内部与边界

| 区域 | FNO散射场误差 | 冻结训练均值误差 | 材料差分场误差 |
| --- | ---: | ---: | ---: |
{chr(10).join(regionrows)}

接收点位于y=127且只占部分x位置；接收误差不能代表整层或内部场。内部仍须单独验收。

## 材料响应方向

冻结最终模型，比较FNO(gamma)−FNO(1)与FD(gamma)−FD(1)。这是六开发验证材料的有限幅度诊断，全1输入是反事实；不是局部材料导数。

| 指标 | 场 | 接收 |
| --- | ---: | ---: |
| 差分相对L2，零响应=100% | {pct(response['field']['relative_l2'])} | {pct(response['trace']['relative_l2'])} |
| 预测范数／参考范数 | {pct(response['field']['norm_ratio'])} | {pct(response['trace']['norm_ratio'])} |
| 余弦 | {response['field']['cosine']:.6f} | {response['trace']['cosine']:.6f} |

## 与四材料阶段比较

| 固定最终指标 | 四训练材料，1600更新 | 十二训练材料，4800更新 |
| --- | ---: | ---: |
| 训练散射接收误差 | {pct(old['final_training']['scatter_trace_relative_l2'])} | {pct(train['scatter_trace_relative_l2'])} |
| 验证散射接收误差 | {pct(old['final_validation']['scatter_trace_relative_l2'])} | {pct(val['scatter_trace_relative_l2'])} |
| 验证散射场误差 | {pct(old['final_validation']['scatter_field_relative_l2'])} | {pct(val['scatter_field_relative_l2'])} |
| 验证材料接收响应余弦 | {oldm['summary']['trace']['cosine']:.6f} | {response['trace']['cosine']:.6f} |

单种子、固定源、开发验证集的阶段比较，不把所有差异归因于材料数量。材料覆盖最近邻诊断及医学领域划分启发已另行记录。

## 后续工作与边界

继续 [[基础DeepONet接口与首轮协议 2026-10-08]]：先做训练000／源0、固定400次的基础模型拟合，完整网格和带符号接收评价后再决定是否扩大。随机采样场损失不是旧FNO每步完整场损失的逐步等价实现，第一轮不作架构因果排名。

现在不接入代理FWI、不据已有开发验证反复调参后宣称独立测试成功；可靠散射和材料响应之后才推进局部物理梯度、新冻结留出及完整反演。二维标量gamma不是弹性金属相控阵实测。

## 已完成核查与复现

4800个Adam步数、12件各400次暴露、同初值和有限权重通过；最终模型重载，12训练／6验证全部合并指标一致。冻结均值仅用12训练目标逐值复算；二次型均值关系、空间分区及材料响应能量恒等式通过。七个作者原文件、manifest与基底哈希不变；旧四材料实验与代码另外保留13项哈希快照。图像、附件副本和笔记链接在交付前检查。

预先协议`.build/basic-fno-four-materials-20261008/next_twelve_materials_protocol.json`保留历史planned_not_run；本轮实际结果在`.build/basic-fno-twelve-materials-20261008`，运行协议和completed结果分别保存。重跑请选择新目录。

```powershell
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_twelve_materials.py --data .build/operator-pilot-v3-20261008 --plan .build/basic-fno-four-materials-20261008/next_twelve_materials_protocol.json --out .build/basic-fno-twelve-materials-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/audit_twelve_material_response.py --data .build/operator-pilot-v3-20261008 --run .build/basic-fno-twelve-materials-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/verify_twelve_materials.py --data .build/operator-pilot-v3-20261008 --run .build/basic-fno-twelve-materials-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/summarize_twelve_materials.py --data .build/operator-pilot-v3-20261008 --run .build/basic-fno-twelve-materials-20261008
```
"""
(v/'05实践与复习/普通FNO十二材料单源训练与验证 2026-10-08.md').write_text(text,encoding='utf-8')
print('Final twelve-material note generated from verified metrics.')
