---
id: "P01"
title: "Neural operators for ultrasonic full waveform inversion: A next-generation ultrasonic NDT with improved fidelity, efficiency, and robustness"
type: "paper"
status: "未读"
evidence: "摘要级导读"
fulltext_read: false
source_row: 2
team_from_sheet: "华东理工 刘立帅"
date_from_sheet: "2026-04"
task: "反演"
doi: "10.1016/j.ymssp.2026.114179"
tags: ["文献", "超声", "核心"]
updated: "2026-09-10"
pdf_status: "已入库"
pdf_file: "90附件/论文原文/P01_U-FNO_FWI_2026.pdf"
---
# P01 U-FNO超声全波形直接反演

## 本地原文

**[[90附件/论文原文/P01_U-FNO_FWI_2026.pdf|打开原文 PDF]]** · [[90附件/论文原文/P01_U-FNO_FWI_2026.pdf#page=1|从第一页开始]]

- 来源：你下载到 Downloads 的原文件（已保留），2026-09-10归档。
- 版本：用户下载的期刊论文排版版；17页。[DOI入口](https://doi.org/10.1016/j.ymssp.2026.114179)。
- 已核对标题、DOI、全部页面可解析及首尾页；尚未完成全文精读。
- 原文件名：`1-s2.0-S0888327026003365-main.pdf`。


## 题录与来源

- 英文题目：Neural operators for ultrasonic full waveform inversion: A next-generation ultrasonic NDT with improved fidelity, efficiency, and robustness
- 作者（原表，可能不完整）：Jiang Lu, Lishuai Liu, Xuan Li, …, Yanxun Xiang
- 期刊／来源（原表）：Mechanical Systems and Signal Processing
- 原表时间：2026-04；类型：期刊论文
- 原表定位：工作表「PINO超声成像文献」第 2 行，A:J。见 [[原始表格说明]]。
- [原表链接](https://www.researchgate.net/publication/403739104)；[本次核验来源](https://www.sciencedirect.com/science/article/abs/pii/S0888327026003365)
- 核验日期：2026-09-10。原表时间 2026-04；出版页面卷期日期 2026-05-01。两者可能分别对应上线与卷期，尚未逐项确认上线日期。

## 已有证据支持什么

作者用 U-FNO 学习从 FMC 数据到纵波声速场的直接逆映射，报告了数值与实测验证。这里的推理流程不需要逐次迭代正演；不能据此认定训练损失包含 PDE 残差。 [来源](https://www.sciencedirect.com/science/article/abs/pii/S0888327026003365)

**任务方向：** 反演。  
**输入 → 输出：** FMC超声全波形数据 → 纵波声速场。

## 先补这些概念

[[正演与反演]] · [[FMC与TFM]] · [[FNO与神经算子]] · [[PINN与PINO的边界]] · [[泛化与误差指标]]

## 三遍阅读

1. **15分钟定位**：摘要、方法图、结论。用自己的话说明为什么需要这篇方法。
2. **45–60分钟拆解**：找到输入输出、控制方程、网络与训练目标。暂缺全文时先做概念学习。
3. **45–60分钟审证据**：检查消融、数据划分、实验边界与失败案例；补上页码和图号。

## 带着这些问题找原文

- [ ] 从网络图和方法段确认 FMC 三个轴如何编码，声速输出如何归一化。
- [ ] 逐项抄出训练损失；哪些项来自标签，哪些项真正涉及控制方程？
- [ ] 找出噪声测试与实测测试的样本来源，是否存在试件或缺陷形状泄漏？
- [ ] 核对计时是否只算推理，训练和数据生成成本是否纳入比较？

## 一项主动练习

画出传统迭代 FWI 与直接逆映射的两条流程，并在图上标出成本发生的位置。

## 精读记录（你来填写）

| 项目 | 我的记录 | 原文页码／公式／图表 |
| --- | --- | --- |
| 输入输出的维度和单位 | 待填写 | 待定位 |
| 控制方程及近似条件 | 待填写 | 待定位 |
| 完整损失与权重 | 待填写 | 待定位 |
| 数据数量与划分单位 | 待填写 | 待定位 |
| 最关键实验及比较条件 | 待填写 | 待定位 |
| 失败情形与适用边界 | 待填写 | 待定位 |

## 完成标准

- [x] 获得并核对全文版本，已归档至 `90附件/论文原文`，见本页链接。
- [ ] 不看原文，写出三句话：问题、方法、证据。
- [ ] 画出完整输入输出链路，并解释每项损失。
- [ ] 找到至少一个支持贡献的消融及一个限制；若原文未提供，明确记为缺失。
- [ ] 完成一次隔天回忆，把仍不懂的问题加入 [[疑问与研究想法]]。

## 与其他文献连接

[[P04 FMC-Net相控阵超分辨成像]] · [[P05 双阶段FNO导波散射]]。比较入口：[[文献对照与选题]]。

返回 [[开始学习]] · [[文献地图]] · [[学习进度]]
