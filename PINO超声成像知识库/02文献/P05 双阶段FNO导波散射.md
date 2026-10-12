---
id: "P05"
title: "A Dual-Stage Fourier Neural Operator for Lamb Wave Scattering with Physics Embedded and Residual Refinement"
type: "paper"
status: "未读"
evidence: "摘要级导读"
fulltext_read: false
source_row: 6
team_from_sheet: "天津大学 刘洋"
date_from_sheet: "2026"
task: "正演"
doi: "10.1016/j.ndteint.2026.103759"
tags: ["文献", "超声", "核心"]
updated: "2026-09-10"
pdf_status: "已入库"
pdf_file: "90附件/论文原文/P05_DualStage_FNO_2026.pdf"
---
# P05 双阶段FNO导波散射

## 本地原文

**[[90附件/论文原文/P05_DualStage_FNO_2026.pdf|打开原文 PDF]]** · [[90附件/论文原文/P05_DualStage_FNO_2026.pdf#page=1|从第一页开始]]

- 来源：你下载到 Downloads 的原文件（已保留），2026-09-10归档。
- 版本：用户下载的期刊论文排版版；16页。[DOI入口](https://doi.org/10.1016/j.ndteint.2026.103759)。
- 已核对标题、DOI、全部页面可解析及首尾页；尚未完成全文精读。
- 原文件名：`1-s2.0-S0963869526001301-main.pdf`。


## 题录与来源

- 英文题目：A Dual-Stage Fourier Neural Operator for Lamb Wave Scattering with Physics Embedded and Residual Refinement
- 作者（原表，可能不完整）：Linfeng Wang, Hongyan Zhang, Zhao Wang, …, Yang Liu
- 期刊／来源（原表）：NDT & E International, Vol. 162, Article 103759
- 原表时间：2026；类型：期刊论文
- 原表定位：工作表「PINO超声成像文献」第 6 行，A:J。见 [[原始表格说明]]。
- [原表链接](https://www.researchgate.net/publication/404232531)；[本次核验来源](https://www.sciencedirect.com/science/article/pii/S0963869526001301)
- 核验日期：2026-09-10。出版页核对 NDT & E International 162，2026-07，103759。

## 已有证据支持什么

双阶段 FNO 先结合物理约束学习整体波场，再学习预测与参考波场的差异以改善局部散射。出版页描述有限差分薄板模型与激光扫描验证；它提供正演能力，并不等于已经完成厚度反演。 [来源](https://www.sciencedirect.com/science/article/pii/S0963869526001301)

**任务方向：** 正演。  
**输入 → 输出：** 厚度分布 → Lamb波散射波场。

## 先补这些概念

[[FNO与神经算子]] · [[残差精修与谱截断]] · [[导波与Lamb模态]] · [[薄板方程与适用条件]] · [[有限差分与自动微分]]

## 三遍阅读

1. **15分钟定位**：摘要、方法图、结论。用自己的话说明为什么需要这篇方法。
2. **45–60分钟拆解**：找到输入输出、控制方程、网络与训练目标。暂缺全文时先做概念学习。
3. **45–60分钟审证据**：检查消融、数据划分、实验边界与失败案例；补上页码和图号。

## 带着这些问题找原文

- [ ] 第二阶段输入包含什么？第一阶段训练时是否冻结或联合更新？
- [ ] 第二阶段学的是场值误差还是 PDE 残差？用公式区分。
- [ ] 厚度变化如何进入薄板方程，变系数项是否完整？
- [ ] 四组消融能否分离物理约束与残差精修的收益？

## 一项主动练习

写出 u预测=G1(h)+G2(…) 的示意流程，待阅读全文后补齐括号，不把示意式当成原文网络。

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

[[P08 PTUNet的S0散射预测]] · [[P06 PI-DeepUFNet不规则缺陷波场]] · [[P07 双阶段FNO腐蚀成像会议]] · [[P01 U-FNO超声全波形直接反演]]。比较入口：[[文献对照与选题]]。

返回 [[开始学习]] · [[文献地图]] · [[学习进度]]
