"""Recalculate recorded scores/metrics, then export a diagnostic report and figure."""
from pathlib import Path
import json
import hashlib
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[2]
export_started=time.perf_counter()
OUT=ROOT/".build/tx-design-value-20261012"
summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
protocol=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
ledger=json.loads((OUT/"ledger.json").read_text(encoding="utf-8"))
table=pd.read_csv(OUT/"metrics.csv")
bg=torch.load(OUT/"background.pt",weights_only=True).numpy().astype(np.float64)
for selection in summary["action_selections"]:
    arrays=[]
    for value in protocol["candidate_materials"]:
        data=torch.load(OUT/f"{selection['geometry']}__gamma_{value:.2f}.pt",weights_only=True)
        arrays.append(data["traces"].numpy().astype(np.float64)-bg)
    bank=np.stack(arrays)
    scores={}
    for action in ["left","right"]:
        differences=np.diff(bank[:,protocol["actions"][action]],axis=0)
        scores[action]=float(min(float(np.square(differences[i]).sum())/(protocol["candidate_materials"][i+1]-protocol["candidate_materials"][i])**2 for i in range(3)))
        assert np.isclose(scores[action],selection["scores"][action],rtol=1e-12)
    selected="right" if scores["right"]>scores["left"]+1e-12*max(scores.values()) else "left"
    assert selected==selection["selected"]
for row in summary["pooled_metrics"]:
    selected=table[(table.noise_fraction==row["noise_fraction"])&(table.policy==row["policy"])]
    assert np.isclose(selected.mae.mean(),row["mae"],rtol=1e-10,atol=1e-14)
primary={x["policy"]:x["mae"] for x in summary["pooled_metrics"] if x["noise_fraction"]==.01}
best_fixed=min(primary["fixed_left"],primary["fixed_right"])
reduction=best_fixed-primary["physics_selected"]
g=protocol["gate"]
gate=(len({x["selected"] for x in summary["action_selections"]})==2 and
      reduction>=g["absolute_mae_reduction_at_least"] and
      reduction/best_fixed>=g["relative_mae_reduction_vs_best_fixed_at_least"])
assert gate==summary["gate"]["passed"]
assert np.isclose(reduction,summary["gate"]["absolute_mae_reduction"])
audit=dict(status="passed",checks=["all action scores recomputed from physical waveforms", "pooled metrics recomputed from CSV", "frozen scientific gate recomputed"],scientific_gate_passed=gate)
(OUT/"independent_analysis_check.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")

font=Path("C:/Windows/Fonts/msyh.ttc")
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams["axes.unicode_minus"]=False
names={"left_deep":"左侧／深","right_deep":"右侧／深","left_shallow":"左侧／浅","right_shallow":"右侧／浅","center_small":"中心／小","center_large":"中心／大"}
policies={"fixed_left":"固定左侧（2源）","fixed_right":"固定右侧（2源）","physics_selected":"灵敏度选择（2源）","full_four_source":"全4源参考"}
fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
indices=np.arange(6);width=.35
for i,(action,label,color) in enumerate([("left","左侧2源","#3274a1"),("right","右侧2源","#e59639")]):
    axes[0].bar(indices+(i-.5)*width,[x["scores"][action] for x in summary["action_selections"]],width,label=label,color=color)
axes[0].set_xticks(indices,[names[x["geometry"]] for x in summary["action_selections"]],rotation=25)
axes[0].set_ylabel("最小相邻材料响应差平方 / 材料步长平方")
axes[0].set_title("响应灵敏度：最优动作随几何改变")
axes[0].legend(frameon=False)
for policy,label in policies.items():
    values=[x["mae"]*1e4 for x in summary["pooled_metrics"] if x["policy"]==policy]
    axes[1].plot([0,1,3],values,"o-",label=label)
axes[1].set_xticks([0,1,3]);axes[1].set_xlabel("噪声标准差 / 背景RMS（%）")
axes[1].set_ylabel(r"材料参数平均绝对误差（$\times10^{-4}$）")
axes[1].set_title("恢复质量：灵敏度选择未超过最佳固定策略")
axes[1].legend(frameon=False,fontsize=9)
for ax in axes:
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
fig.suptitle("已知支持、同网格标量模型的开发诊断；未训练网络",fontsize=14)
fig.savefig(OUT/"diagnostic.png",dpi=160)
plt.close(fig)

score_rows="\n".join(f"|{names[x['geometry']]}|{x['scores']['left']:.2f}|{x['scores']['right']:.2f}|{'左' if x['selected']=='left' else '右'}|" for x in summary["action_selections"])
metric_rows="\n".join(f"|{policies[policy]}|{primary[policy]:.8f}|{'4' if policy=='full_four_source' else '2'}|" for policy in policies)
report=f"""# 发射动作价值诊断：首轮冻结实验（2026-10-12）

## 结论

**发射动作的响应灵敏度随目标几何改变，但本轮灵敏度选择没有改善材料恢复，预登记科学收益门槛未通过。** 该结论限定于已知支持、同网格标量模型和当前分段线性估计器。

在主评价条件（噪声标准差为背景RMS的1%）下，选择策略的材料参数MAE为 {primary['physics_selected']:.8f}，最佳固定策略为 {best_fixed:.8f}；相对恶化 {-reduction/best_fixed*100:.2f}%。预登记要求动作发生切换、相对MAE至少下降10%、绝对MAE至少下降0.0001，后两项未满足。

## 冻结设计

- 6个预先指定的几何：左右镜像的两个深度条件，以及中心的两种尺寸；不是6次随机独立工件试验。
- x为横向位置，阵列位于y=50 mm；y=35 mm比y=25 mm更靠近阵列。
- 候选材料节点：0.44、0.48、0.52、0.56；评价目标：0.46、0.50、0.54，目标不在候选节点上。
- 估计器：物理候选波形间的分段线性连续最小二乘估计，允许落在节点之间，避免两候选离散量化必然误差。
- 动作分数只由候选字典计算，取最小相邻节点响应差平方/材料步长平方。目标真值只用于生成观测及事后评价。
- 固定左、固定右、灵敏度选择均为2源、每源24接收通道；全4源参考有两倍发射预算。
- 噪声水平0%、1%、3%；每条件100个预先指定种子，同一噪声实现用于比较各策略。噪声重复不能当成独立工件数。
- 没有初始观测驱动的动作更新；这是已知几何下的静态物理实验设计，不是主动闭环。

## 动作响应

|几何|左侧分数|右侧分数|选择|
|---|---:|---:|---|
{score_rows}

左右几何的排序方向相反，支持“单个偏左目标上左源更敏感”不能外推为全局最佳动作。中心条件左右分数较接近，不应据小差异宣称可靠物理优势。

## 主评价结果

|策略|材料参数MAE|发射源数|
|---|---:|---:|
{metric_rows}

全4源在本估计器下也不保证更低MAE。增加观测不能消除近似模型的偏差；本结果不能解释为更多发射在真实仪器中有害。

无噪声下选择策略MAE为0.00025809，与1%噪声下接近。目标与候选使用相同求解器，因而无噪声非零误差主要反映分段线性近似/估计偏差及数值精度，而非神经网络误差。现有结果不足以证明该灵敏度分数改善恢复质量，也不足以否定其他估计器下的动作选择价值。

## 成本与失败记录

- 背景4次、候选字典96次、评价目标72次、独立重复核验2次，合计**174次实际源求解**，全部完成；没有自动重试或失败重算。
- 网络前向0次，训练更新0次。
- 准备 {ledger['preparation_seconds']:.2f} 秒；生成与重复核验 {ledger['generation_and_replay_seconds']:.2f} 秒；分析与导出 {ledger['analysis_and_export_seconds']:.2f} 秒。输入/输出哈希检查和估计器行为核验时间另见verification.json；报告重算、绘图和导出时间另记report_export.json。绘图曾出现字体缺字警告并已修复，首次导出耗时3.02秒，另计入该文件。
- 候选字典的96次离线源求解不能视作免费。为比较策略，本轮实际生成了全部4源目标数据；2源预算是模拟部署预算，并非已测得实际采集节省。
- 没有测量真实仪器采集时间、历史训练成本或独立求解器误差。

## 核验与下一步

输入与生成文件哈希、42个四源波形文件、174条源求解记录、2次逐元素重复核验、连续估计器行为及216条指标的重算通过。报告工具另外从波形重算所有动作分数，并从CSV重算汇总指标和科学门槛。技术核验通过与科学收益未通过分别记录。

下一步只做估计偏差诊断：先检查候选字典加密或物理一维优化是否降低无噪声偏差，再重新冻结公平的动作评分协议，并增加未参与开发的新几何评价。不得调阈值使本轮通过，也不把这6个开发几何再次称为最终独立测试。在恢复质量收益出现之前，不训练更大的FNO、不引入不确定性网络或展开式反演。

## 复现与文件

在仓库根目录运行（需要torch、numpy、pandas及PyTables/tables；Mac尚未实机验证）：

```sh
python Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_tx_design_value_20261012.py generate --out .build/tx-design-value-reproduction
python Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_tx_design_value_20261012.py analyze --out .build/tx-design-value-reproduction
python Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_tx_design_value_20261012.py verify --out .build/tx-design-value-reproduction
```

必须使用新输出目录。`generate`会冻结输入哈希并重新生成波形；`analyze`从已生成波形计算指标；`verify`核对协议、数据及结果。代码不依赖旧实验的PT文件；只使用仓库中的源HDF5、设置CSV和物理代码。CPU回退可用，但运行时间尚未验证，超过冻结时间上限会保存失败记录并停止。

- `protocol.json`：求解前冻结的协议与哈希。
- `data_manifest.json`、`ledger.json`、`environment.json`：数据、真实调用成本和执行环境。
- `summary.json`、`metrics.csv`、`parameter_estimates.npz`：结论与可重算的参数估计。
- `verification.json`、`independent_analysis_check.json`：技术核验与独立分析重算。
- `diagnostic.png`：响应排序与恢复误差。
- 原始波形PT文件留在Windows，由Git忽略；其余轻量结果可同步到Mac。
"""
(OUT/"report.md").write_text(report,encoding="utf-8")
exports={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ["report.md","diagnostic.png","independent_analysis_check.json"]}
(OUT/"report_export.json").write_text(json.dumps(dict(status="completed", script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), elapsed_seconds=time.perf_counter()-export_started, previous_font_warning_export_seconds=3.02, artifact_sha256=exports),indent=2),encoding="utf-8")
print(json.dumps(dict(analysis_check="passed",report=str(OUT/"report.md"),figure=str(OUT/"diagnostic.png")),ensure_ascii=False))
