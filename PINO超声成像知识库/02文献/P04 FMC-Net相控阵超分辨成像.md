---
id: "P04"
title: "A multiscale residual U-net architecture for super-resolution ultrasonic phased array imaging from full matrix capture data (FMC-Net)"
type: "paper"
status: "未读"
evidence: "摘要级导读"
fulltext_read: false
source_row: 5
team_from_sheet: "华东理工 刘立帅"
date_from_sheet: "2023-10"
task: "成像"
doi: "10.1121/10.0021171"
tags: ["文献", "超声", "拓展"]
updated: "2026-09-10"
pdf_status: "待下载"
pdf_file: ""
---
# P04 FMC-Net相控阵超分辨成像

## 原文获取

**尚未下载原文。** 作者已在ResearchGate公开上传全文，但下载请求及AIP下载均返回403。建议先手动打开作者页面点击 Download full-text PDF。

[出版社／DOI入口](https://doi.org/10.1121/10.0021171) · [ResearchGate页面](https://www.researchgate.net/publication/374384943)

下载后可放入 `90附件/待导入原文`，由助手核对版本后归档。见 [[原文下载清单]]。


> [!info] 阅读起点
> 先理解数据驱动成像基线，再判断神经算子增益来自哪里。 本页为摘要级导读，没有将摘要核验标成全文精读。练习与问题是学习设计，不是作者结论。

## 题录与来源

- 英文题目：A multiscale residual U-net architecture for super-resolution ultrasonic phased array imaging from full matrix capture data (FMC-Net)
- 作者（原表，可能不完整）：Lishuai Liu, Wen Liu, Da Teng, …, Fu-Zhen Xuan
- 期刊／来源（原表）：The Journal of the Acoustical Society of America
- 原表时间：2023-10；类型：期刊论文
- 原表定位：工作表「PINO超声成像文献」第 5 行，A:J。见 [[原始表格说明]]。
- [原表链接](https://www.researchgate.net/publication/374384943)；[本次核验来源](https://pubmed.ncbi.nlm.nih.gov/37782121/)
- 核验日期：2026-09-10。作者摘要由 PubMed 收录；题录补充 JASA 154(4):2044–2054，2023-10，DOI 已核对。

## 已有证据支持什么

FMC-Net 使用多尺度残差编码器与解码器直接形成图像，摘要明确输出对应散射系数分布。论文以数值和实验与 TFM、波数算法比较；超分辨表述应放回具体缺陷与测试条件理解。 [来源](https://pubmed.ncbi.nlm.nih.gov/37782121/)

**任务方向：** 成像。  
**输入 → 输出：** FMC数据 → 声学散射系数空间分布。

## 先补这些概念

[[FMC与TFM]] · [[U-Net与多尺度残差]] · [[泛化与误差指标]]

## 三遍阅读

1. **15分钟定位**：摘要、方法图、结论。用自己的话说明为什么需要这篇方法。
2. **45–60分钟拆解**：找到输入输出、控制方程、网络与训练目标。暂缺全文时先做概念学习。
3. **45–60分钟审证据**：检查消融、数据划分、实验边界与失败案例；补上页码和图号。

## 带着这些问题找原文

- [ ] 标签是散射系数还是二值缺陷掩膜？原文如何定义？
- [ ] FMC 的发射、接收与时间维度如何变成网络输入？
- [ ] 多尺度残差模块与普通 U-Net 的参数量是否接近？
- [ ] 分辨率结论用了什么双目标间距或其他判据？

## 一项主动练习

写出散射系数图与声速图的区别，说明为什么 P04 和 P01 的结果不能直接视作同一物理量。

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

- [ ] 获得并核对全文版本，将 PDF 放入 `90附件`，在此添加内部链接。
- [ ] 不看原文，写出三句话：问题、方法、证据。
- [ ] 画出完整输入输出链路，并解释每项损失。
- [ ] 找到至少一个支持贡献的消融及一个限制；若原文未提供，明确记为缺失。
- [ ] 完成一次隔天回忆，把仍不懂的问题加入 [[疑问与研究想法]]。

## 与其他文献连接

[[P01 U-FNO超声全波形直接反演]] · [[P03 波场极性感知U-Net深度成像]]。比较入口：[[文献对照与选题]]。

返回 [[开始学习]] · [[文献地图]] · [[学习进度]]
