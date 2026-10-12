---
id: "P02"
title: "Physics-informed Fourier neural network with self-adaptive loss weight for modeling transient wave propagation"
type: "paper"
status: "未读"
evidence: "摘要级导读"
fulltext_read: false
source_row: 3
team_from_sheet: "华东理工 刘立帅"
date_from_sheet: "2026-02"
task: "正演与参数反演"
doi: "10.1016/j.engappai.2026.113994"
tags: ["文献", "超声", "拓展"]
updated: "2026-09-10"
pdf_status: "已入库"
pdf_file: "90附件/论文原文/P02_PIFNN_2026.pdf"
---
# P02 PIFNN瞬态波与自适应权重

## 本地原文

**[[90附件/论文原文/P02_PIFNN_2026.pdf|打开原文 PDF]]** · [[90附件/论文原文/P02_PIFNN_2026.pdf#page=1|从第一页开始]]

- 来源：你下载到 Downloads 的原文件（已保留），2026-09-10归档。
- 版本：用户下载的期刊论文排版版；14页。[DOI入口](https://doi.org/10.1016/j.engappai.2026.113994)。
- 已核对标题、DOI、全部页面可解析及首尾页；尚未完成全文精读。
- 原文件名：`1-s2.0-S0952197626002757-main.pdf`。


## 题录与来源

- 英文题目：Physics-informed Fourier neural network with self-adaptive loss weight for modeling transient wave propagation
- 作者（原表，可能不完整）：Jiang Lu, Lishuai Liu, Haiming Xu, …, Fu-Zhen Xuan
- 期刊／来源（原表）：Engineering Applications of Artificial Intelligence
- 原表时间：2026-02；类型：期刊论文
- 原表定位：工作表「PINO超声成像文献」第 3 行，A:J。见 [[原始表格说明]]。
- [原表链接](https://www.researchgate.net/publication/400741378)；[本次核验来源](https://www.sciencedirect.com/science/article/pii/S0952197626002757)
- 核验日期：2026-09-10。原表时间 2026-02；出版页卷期为 168，2026-03-15，文章号 113994。不要自动把原表月份判作错误。

## 已有证据支持什么

PIFNN 将 Fourier 特征映射和基于神经切线核（NTK）的自适应损失权重引入 PINN，针对高频表示和损失收敛失衡问题开展数值验证。Fourier 特征映射本身不能证明是 FNO。 [来源](https://www.sciencedirect.com/science/article/pii/S0952197626002757)

**任务方向：** 正演与参数反演。  
**输入 → 输出：** 坐标及物理条件 → 瞬态波场；参数反演细节待全文。

## 先补这些概念

[[PINN与PINO的边界]] · [[谱偏置与Fourier特征]] · [[物理损失与自适应权重]] · [[波动方程与Helmholtz方程]]

## 三遍阅读

1. **15分钟定位**：摘要、方法图、结论。用自己的话说明为什么需要这篇方法。
2. **45–60分钟拆解**：找到输入输出、控制方程、网络与训练目标。暂缺全文时先做概念学习。
3. **45–60分钟审证据**：检查消融、数据划分、实验边界与失败案例；补上页码和图号。

## 带着这些问题找原文

- [ ] Fourier 特征作用于哪些坐标？频率矩阵如何选取？
- [ ] NTK 权重公式在哪一节？更新频率与计算成本是多少？
- [ ] 有无只加 Fourier 特征、只加自适应权重的独立消融？
- [ ] 参数反演中哪些参数未知，观测如何约束它们？

## 一项主动练习

列一个 2×2 消融表：有无 Fourier 特征 × 有无自适应权重。先写预期，再查原文。

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

[[P10 PINN经颅波传播]] · [[P05 双阶段FNO导波散射]]。比较入口：[[文献对照与选题]]。

返回 [[开始学习]] · [[文献地图]] · [[学习进度]]
