# FNO × 超声检测：文献分析与科研方向

> 2026-10-07 更新：用户已选择相控阵 FWI 加速主线。当前安排见 [科研计划](../PINO超声成像知识库/00导航/FNO与超声全波形反演科研计划%202026-10-07.md)，依据见 [跨领域进展](../PINO超声成像知识库/04专题/相控阵FWI与神经算子跨领域进展%202026-10-07.md)。以下保留 2026-09-15 的历史研判，不再作为当前唯一推荐。

分析日期：2026-09-15。基于本地11篇编号原文的关键章节核查、1篇摘要证据，以及少量外部一手来源排查。原始文献未改动。

## 先看这里

**推荐起点：有限标签下物理约束FNO的超声散射正演可靠性。** 先做二维声学小实验，验证物理约束在独立数值参考下是否仍有收益，再考虑薄板导波和实测。它目前是研究假设，尚未运行实验，也未认证全球新颖性。

1. [最终三个推荐与Top 1完整实验方案](05_Final_Recommendations.md)
2. [Innovation Matrix与证据链](03_Gap_Analysis.md)
3. [10个候选方向、难度与评分](04_Candidate_Directions.md)
4. [本科研究阶梯](06_Research_Ladder.md)

## 全部分析文件

|文件|用途|
|---|---|
|[00 Library Index](00_Library_Index.md)|全目录清单、重复件、解析失败、分类与阅读优先级|
|[01 Literature Matrix](01_Literature_Matrix.md)|12篇逐项矩阵；原文事实与本轮判断分开|
|[02 Research Map](02_Research_Map.md)|正演、成像、逆问题的接口及FNO位置|
|[03 Gap Analysis](03_Gap_Analysis.md)|7类gap与Innovation Matrix|
|[04 Candidate Directions](04_Candidate_Directions.md)|10个独立候选、MVP、风险、加权排名|
|[05 Final Recommendations](05_Final_Recommendations.md)|3个推荐，Top 1方程/数据/损失/基线/消融/停止条件|
|[06 Research Ladder](06_Research_Ladder.md)|Level 0–7：知识、代码、阅读、实验、晋级标准|
|[Future Work Mining](Future_Work_Mining.md)|作者未来工作与分析推断；关键词命中页|
|[External Check](External_Check.md)|补充一手检索，防止将已有思路当创新|
|[Verification Notes](Verification_Notes.md)|公式、样本数、实验措辞疑点及核验范围|

## 阅读卡

[B01 FNO](papers/B01.md) · [B02 PINO](papers/B02.md) · [P01 U-FNO逆问题](papers/P01.md) · [P02 PIFNN](papers/P02.md) · [P03 极性成像](papers/P03.md) · [P04 FMC-Net摘要](papers/P04.md) · [P05 双阶段正演](papers/P05.md) · [P06 PI-DeepUFNet预印本](papers/P06.md) · [P07 导波逆问题](papers/P07.md) · [P08 PTUNet](papers/P08.md) · [P09 FEUNet](papers/P09.md) · [P10 PINN](papers/P10.md)

## 证据怎么理解

- 本地11篇原文共187页已抽取，按方法/实验/讨论重点核对；不等同于逐字审稿或复现实验。
- P04全文缺失，仅核实作者摘要；两个译本和未命名PDF无法可靠解析，需要人工检查，不影响可读英文原文的使用。
- P06为未同行评审预印本，测试数量口径有疑问。P05/P08公式页已视觉核查，疑点单列，未据此断言作者结果无效。
- `evidence/`保留原文页码文本、清单、生成脚本和核验记录，供追溯；它不是需要优先阅读的摘要合集。
- Markdown链接和评分已做结构检查；未做Obsidian原生界面排版核验。保持此目录与原知识库的相对位置，原PDF链接即可继续使用。

## 一句话总结

文献已经证明了若干FNO超声正演/反演方案的可行性。你现在最值得做的是选一个小而明确的问题，严格控制数据、物理和比较条件，检验它到底在哪些情况下有效。
