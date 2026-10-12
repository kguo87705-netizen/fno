# 补充检索：用于防止误判创新，不替代本地精读

检索日期2026-09-15。先分析本地11篇原文，再用少量外部一手来源排查已有相邻思路。不是系统综述，也不是“截至今日无人研究”的证明。

|ID|工作与一手链接|本轮核实范围|对选题的影响|
|---|---|---|---|
|E01|Li等，2023，[Geo-FNO](https://jmlr.org/papers/v24/23-0064.html)|出版页摘要：学习物理域到规则潜在域的变形|复杂几何FNO已有，不能仅以曲面/网格适配宣称首创|
|E02|Nair等，[Physics and geometry informed neural operator network with application to acoustic scattering](https://arxiv.org/abs/2406.03407)|作者摘要：NURBS几何编码、物理约束DeepONet、刚性散射体|无标签几何声学算子已有；“物理+不规则散射体”不是空白|
|E03|Huang、Alkhalifah，2025，[Learned frequency-domain scattered wavefield solutions using neural operators](https://academic.oup.com/gji/article/241/3/1467/8096642)|出版社正文方法概述：背景场隐式表示源与频率，介质加背景场到散射场|源/频率条件化、背景场输入不能声称新架构；地震验证不自动覆盖金属超声|
|E04|Zhu等，2023，[Fourier-DeepONet](https://arxiv.org/abs/2305.17289)|作者摘要：FNO解码DeepONet，源位置/频率变化下FWI|条件化逆算子已有；需要UT特定、可测的贡献|
|E05|Li等，NeurIPS 2023，[Geometry-Informed Neural Operator for Large-Scale 3D PDEs](https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html)|会议作者摘要：SDF几何表示，图算子在非规则采样与规则潜在网格间映射，FNO处理潜在网格|为graph/mesh-based路线提供具体依据；其汽车空气动力学验证不代表超声弹性边界已解决|
|P04补证|Liu等，2023，[FMC-Net作者摘要](https://pubmed.ncbi.nlm.nih.gov/37782121/)|FMC到散射系数图、数值/实验对比TFM与波数算法|只能摘要级，不补写数据量/网络全部细节|

执行过的检索词：`Fourier neural operator ultrasonic source frequency generalization wavefield`、`physics geometry neural operator acoustic scattering Nair`、`Fourier Neural Operator Lamb sample`、`ultrasonic neural operator limited data`、`Lamb neural operator resolution`、`Learning Deformations Fourier Neural Operators geometry`。

排除规则：没有用聚合站或生成式解读作为技术证据；搜索结果日期不代替正式出版页日期。上述来源只支撑所列有限命题。Geo-FNO摘要的速度数值不迁移为本课题速度预期。

补充查询：`site.proceedings.neurips.cc Geometry-Informed Neural Operator 3D`，用于核对图算子与非规则网格路线。

下一轮针对确定的课题再检索：`ultrasonic scattering physics informed neural operator discretization error sample efficiency`、`Lamb waves neural operator solver mismatch`，做前后向引文追踪与代码核对。本轮推荐是有依据的研究问题，不是已完成新颖性认证。
