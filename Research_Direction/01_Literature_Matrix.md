# 01 文献矩阵与核心精读记录

本轮从本地原文核对方法、实验和讨论中的关键证据，非逐句翻译。11篇有原文，P04仅有在线作者摘要。页码统一为PDF物理页码；“未确认”不等于作者没有做。所有“局限性/相关性”中明示本轮判断的内容均为分析，不冒充作者结论。

## 快速比较

|ID|论文|FNO|物理损失|映射|证据|
|---|---|---|---|---|---|
|[B01](papers/B01.md)|Fourier Neural Operator for Parametric Partial Differential Equations|是|否：主训练为监督数据损失|Burgers初值；Darcy渗透率；NS早期涡量 → 终态解或后续涡量场|本地原文关键章节|

## B01 Fourier Neural Operator for Parametric Partial Differential Equations

定位：p4–6 架构与训练；p9 超分辨、逆问题、数据局限。详见 [B01精读卡](papers/B01.md)。

|字段|内容|
|---|---|
|年份|2020预印本；本地v3标注ICLR 2021|
|研究问题|跨实例学习PDE解算子|
|方法|Fourier核积分层与局部分支叠加|
|输入|Burgers初值；Darcy渗透率；NS早期涡量|
|输出|终态解或后续涡量场|
|PDE / 物理模型|Burgers、Darcy、二维Navier–Stokes|
|数据来源|数值求解器生成；不同网格由高分辨率下采样|
|数据规模|默认1000训练/200测试；NS低黏度超分辨实验10000训练（p6、9）|
|网络|四层FNO|
|是否 FNO|是|
|是否 Physics-informed|否：主训练为监督数据损失|
|Forward / Inverse|正演；正演代理用于贝叶斯反演|
|2D / 3D|1D、2D空间；FNO-3D也包含x-y-t情形|
|计算优势|p9代理单次0.005s，对照GPU求解器2.2s；场景特定，不是超声测速|
|局限性|需要大量训练对；FFT实现要求均匀采样；未验证超声|
|作者提出 Future Work|p9结合数值求解器降低数据需求；循环共享层结构|
|与超声检测关系|方法基础，非超声实证|
|与我的课题相关性|高|

**算子核查：** 跨参数：在训练任务族中；跨网格：方法定义支持离散化变化，本文FFT实验为规则网格；跨分辨率：有NS实验；PDE未入损失；导数不用于主训练。

**本轮判断：** B01的非线性层可再生成高频，不能说FNO输出严格只含保留的模式。跨分辨率不等于跨缺陷、跨频率或跨边界条件。
|[B02](papers/B02.md)|Physics-Informed Neural Operator for Learning Partial Differential Equations|是|是|初值/系数函数 → PDE解函数|本地原文关键章节|

## B02 Physics-Informed Neural Operator for Learning Partial Differential Equations

定位：p8–10 训练与导数；p15–17、22–27 实验和边界。详见 [B02精读卡](papers/B02.md)。

|字段|内容|
|---|---|
|年份|2021首版；本地arXiv v4，不能把版本年与正式出版年混用|
|研究问题|结合低分辨率标签与物理约束学习解算子|
|方法|算子学习，可选逐实例微调与anchor loss|
|输入|初值/系数函数|
|输出|PDE解函数|
|PDE / 物理模型|Burgers、Darcy、Navier–Stokes等|
|数据来源|数值数据与额外无标签PDE实例|
|数据规模|分任务：Burgers/Darcy各1000训练；NS多档样本及额外PDE实例（p22–24）|
|网络|FNO骨干PINO|
|是否 FNO|是|
|是否 Physics-informed|是|
|Forward / Inverse|正演与Darcy反问题|
|2D / 3D|1D/2D空间及时间维|
|计算优势|p17 Darcy离线约1小时；精度与微调耗时需分开|
|局限性|FFT高维扩展、数值导数误差、微调成本|
|作者提出 Future Work|p17优化方法、精度与复杂度折中、跨几何迁移、预训练库|
|与超声检测关系|超声方法迁移依据，非超声验证|
|与我的课题相关性|高|

**算子核查：** 跨参数：有PDE族训练及Re迁移/微调；跨网格/分辨率：有；不等于任意非结构网格。PDE入损失；导数讨论FD、Fourier数值微分、query autograd及函数级链式法。

**本轮判断：** PINO不是把输入坐标加Fourier特征；对网络权重反传不等于对物理坐标求导。FFT导数要求考虑光滑性、边界延拓与混叠。
|[P01](papers/P01.md)|Neural operators for ultrasonic full waveform inversion: A next-generation ultrasonic NDT with improved fidelity, efficiency, and robustness|是|主损失不含PDE残差；物理用于造数据|FMC：64发射×64接收×1000时间点 → 200×200纵波声速图|本地原文关键章节|

## P01 Neural operators for ultrasonic full waveform inversion: A next-generation ultrasonic NDT with improved fidelity, efficiency, and robustness

定位：p4–8 方法与数据；p10–15 性能、实验与局限。详见 [P01精读卡](papers/P01.md)。

|字段|内容|
|---|---|
|年份|2026，MSSP 251:114179|
|研究问题|从单侧阵列全波形直接重建声速图|
|方法|U-FNO监督逆映射|
|输入|FMC：64发射×64接收×1000时间点|
|输出|200×200纵波声速图|
|PDE / 物理模型|二维声学模型；忽略纵横波转换|
|数据来源|EMNIST字母形状→声速→交错网格FD；3件哈氏合金水浸试件|
|数据规模|6000：4500训练/500验证/1000测试（p7）；实验3件（p11）|
|网络|1个Fourier层+2个U-Fourier层（p6表1）|
|是否 FNO|是|
|是否 Physics-informed|主损失不含PDE残差；物理用于造数据|
|Forward / Inverse|直接反演；不是把FNO嵌入迭代FWI|
|2D / 3D|2D空间|
|计算优势|p11表4：U-FNO 0.049s/重建；InversionNet 0.026s；FWI 7275s；U-FNO训练约30h|
|局限性|1500–5800m/s，主要水填缺陷；采集/材料/结构偏移；二维点源简化|
|作者提出 Future Work|p14–15数据多样化、迁移学习、物理约束、更准确弹性模型、3D|
|与超声检测关系|直接UT/FMC；支持逆问题，但不是所有固体裂纹|
|与我的课题相关性|高|

**算子核查：** 跨参数：新形状及加噪测试；跨分辨率/阵列：本次未找到明确实验证明。固定尺寸投影和U-Net分支需另核验，不能由FNO名字推断任意尺寸可用。

**本轮判断：** 它比所比FWI快，但并不比InversionNet快。表中训练、推理与FWI每迭代时间不能混用。TFM强度图与声速图不是同一物理量。
|[P02](papers/P02.md)|Physics-informed Fourier neural network with self-adaptive loss weight for modeling transient wave propagation|否：Fourier feature不是FNO|是|空间/时间坐标 → 该实例的波场|本地原文关键章节|

## P02 Physics-informed Fourier neural network with self-adaptive loss weight for modeling transient wave propagation

定位：p3–5 输入、AD与NTK；p12 结论。详见 [P02精读卡](papers/P02.md)。

|字段|内容|
|---|---|
|年份|2026，EAAI 168:113994|
|研究问题|PINN高频谱偏置与损失收敛失衡|
|方法|Fourier特征+基于NTK的自适应权重|
|输入|空间/时间坐标|
|输出|该实例的波场|
|PDE / 物理模型|弦振动、二维声学波动方程|
|数据来源|数值参考与早期快照、PDE/IC/BC配点|
|数据规模|分案例配点；波传播用两张早期快照；不等同于两个介质训练样本|
|网络|坐标MLP/PIFNN|
|是否 FNO|否：Fourier feature不是FNO|
|是否 Physics-informed|是|
|Forward / Inverse|本轮核实实证主要正演，不能将动机里的参数反演当完成实验|
|2D / 3D|1D/2D空间+时间|
|计算优势|p12作者报告更好反射特征，不能据此承诺跨实例加速|
|局限性|特征初始化敏感；高维/非线性推广待研究；未证明一次训练覆盖任意介质|
|作者提出 Future Work|p12特征初始化、非线性/高维波动方程|
|与超声检测关系|可借鉴损失训练机制，不是PINO直接先例|
|与我的课题相关性|高|

**算子核查：** 跨参数算子：未证明；坐标任意查询不等于跨介质。PDE加入损失；物理导数使用自动微分。

**本轮判断：** 将NTK权重迁到FNO仍需验证成本和效果；完整NTK矩阵可能很大，先用固定权重及梯度范数基线。
|[P03](papers/P03.md)|A steady-state wavefield polarity-aware U-Net for ultrasonic guided waves depth-resolved imaging|否|无PDE损失；特征预处理有物理依据|稳态波场实部/虚部分别取符号后的SSWP → 深度/厚度分布图|本地原文关键章节|

## P03 A steady-state wavefield polarity-aware U-Net for ultrasonic guided waves depth-resolved imaging

定位：p3–4 极性定义；p5–6 数据；p8 结果讨论。详见 [P03精读卡](papers/P03.md)。

|字段|内容|
|---|---|
|年份|正式卷期2026，NDT&E 158:103597；原题录2025可能为上线时间|
|研究问题|减少波场幅值域偏移，改善深度成像|
|方法|参考场过滤+稳态极性双通道U-Net|
|输入|稳态波场实部/虚部分别取符号后的SSWP|
|输出|深度/厚度分布图|
|PDE / 物理模型|导波弹性传播背景；频域FE数据|
|数据来源|6000频域仿真与实验激光扫描|
|数据规模|6000，9:1训练/测试；未单列验证集（p6）|
|网络|U-Net|
|是否 FNO|否|
|是否 Physics-informed|无PDE损失；特征预处理有物理依据|
|Forward / Inverse|成像/反演|
|2D / 3D|2D面内图|
|计算优势|p5–6造数约270h；不应仅以推理快忽略造数|
|局限性|幅值被舍弃；实验为机加工减薄；泛化仍受形状/采样条件限制（后两点为本轮判断）|
|作者提出 Future Work|p8未见独立明确future-work清单；不得代作者添加|
|与超声检测关系|工业导波、激光扫描；直接sim-to-real证据|
|与我的课题相关性|高|

**算子核查：** 跨实例有；实验域验证有。跨分辨率系统验证未确认；没有算子网格不变性证据。

**本轮判断：** 不能把SSWP叫完整相位：分别保留实虚部符号会丢失更多信息。论文是深度图恢复，不能据其结果认定原波场高频全部被恢复。
|[P04](papers/P04.md)|A multiscale residual U-net architecture for super-resolution ultrasonic phased array imaging from full matrix capture data|否|摘要不能确认PDE损失|FMC → 声学散射系数空间分布|摘要|

## P04 A multiscale residual U-net architecture for super-resolution ultrasonic phased array imaging from full matrix capture data

定位：2026-09-15核对作者摘要：https://pubmed.ncbi.nlm.nih.gov/37782121/。详见 [P04精读卡](papers/P04.md)。

|字段|内容|
|---|---|
|年份|2023，JASA 154:2044–2054|
|研究问题|FMC直接成像|
|方法|多尺度残差编码器-解码器|
|输入|FMC|
|输出|声学散射系数空间分布|
|PDE / 物理模型|全文缺失，未确认|
|数据来源|作者摘要称数值与实验|
|数据规模|未确认|
|网络|FMC-Net / U-Net|
|是否 FNO|否|
|是否 Physics-informed|摘要不能确认PDE损失|
|Forward / Inverse|成像|
|2D / 3D|全文缺失，未确认|
|计算优势|摘要报告优于TFM/波数法的成像质量；没有可核对耗时|
|局限性|仅摘要证据；训练划分、分辨率判据未核查|
|作者提出 Future Work|未确认|
|与超声检测关系|工业相控阵成像的重要比较项|
|与我的课题相关性|中（相关度高但证据不足）|

**算子核查：** 跨参数/网格/分辨率/导数均未完成全文核查。

**本轮判断：** 全文缺失，不能给出数据量、超分辨倍率或完整基线实现。
|[P05](papers/P05.md)|A dual-stage Fourier neural operator for Lamb wave scattering with physics embedded and residual refinement|是|是：有限差分PDE损失|厚度h(x,y)；第二阶段另加第一阶段预测 → 面外波场w(x,y,t)|本地原文关键章节|

## P05 A dual-stage Fourier neural operator for Lamb wave scattering with physics embedded and residual refinement

定位：p3–6 方程/数据/阶段；p8–9 误差；p13–15 局限与附录。详见 [P05精读卡](papers/P05.md)。

|字段|内容|
|---|---|
|年份|2026，NDT&E 162:103759|
|研究问题|改善导波正演局部散射与高频细节|
|方法|先PDE约束FNO，再残差精修|
|输入|厚度h(x,y)；第二阶段另加第一阶段预测|
|输出|面外波场w(x,y,t)|
|PDE / 物理模型|KL薄板近似，主要A0；离散弯矩形式|
|数据来源|KL有限差分；附录少量3D FEM对照；激光测振|
|数据规模|1280；1024训练，其余256；251×251×201输出（p5–6）|
|网络|双阶段FNO|
|是否 FNO|是|
|是否 Physics-informed|是：有限差分PDE损失|
|Forward / Inverse|正演；反演是后续设想|
|2D / 3D|2D空间+时间，非完整3D弹性场|
|计算优势|p5–6造数7.27h；两阶段训练37.11h；1280推理287s；误差相对Stage0降约30–45%|
|局限性|单模式简化；大样本依赖；机加工缺陷；第二阶段同时增加modes需公平消融|
|作者提出 Future Work|p13–14多模态/Navier/3D、少标签、真实腐蚀|
|与超声检测关系|最直接的导波正演参考|
|与我的课题相关性|高|

**算子核查：** 跨缺陷实例有；频率固定200kHz、源位置固定；跨分辨率实证未确认。物理导数为FD；不是AD坐标网络。

**本轮判断：** p3式(6)(7)符号与弯矩推导存在疑点，见核验记录。固定源/频率下新缺陷表现，不是任意频率或探头泛化。
|[P06](papers/P06.md)|Physics-Informed Deep U-Fourier Neural Network for Efficient Prediction of Guided Wavefield in Plates with Irregular Defects|是|是：前10轮纯数据，之后双损失权重0.5/0.5（p24）|250×250×2：厚度与初始波场 → 201×250×250时空波场|本地原文关键章节|

## P06 Physics-Informed Deep U-Fourier Neural Network for Efficient Prediction of Guided Wavefield in Plates with Irregular Defects

定位：p22–24 数据与训练；p27、35统计；p36–40 实验和结论。详见 [P06精读卡](papers/P06.md)。

|字段|内容|
|---|---|
|年份|本地2026 SSRN预印本，43页；首页明确未同行评审|
|研究问题|不规则厚度场的时空导波正演|
|方法|U-Net局部编码+3D FNO+KL物理约束|
|输入|250×250×2：厚度与初始波场|
|输出|201×250×250时空波场|
|PDE / 物理模型|变厚度KL弯曲波，FD约束|
|数据来源|KL数值数据；机加工缺陷SLDV角向幅值比较|
|数据规模|p23–24：2000按8:1:1；p35又称500独立测试；口径冲突未消除|
|网络|PI-DeepUFNet|
|是否 FNO|是|
|是否 Physics-informed|是：前10轮纯数据，之后双损失权重0.5/0.5（p24）|
|Forward / Inverse|正演|
|2D / 3D|2D空间+时间；3D FNO指x-y-t|
|计算优势|p24 RTX4090训练150 epochs；p38定性称秒级，未独立确认可比端到端计时|
|局限性|样本口径不一致；初始场输入获取成本；实验以归一化角向幅值为主；非全面sim-to-real|
|作者提出 Future Work|p39–40推广其他控制方程、为后续反演和数字孪生提供正演|
|与超声检测关系|直接导波；是先例但证据置信受预印本与口径影响|
|与我的课题相关性|高|

**算子核查：** 新缺陷测试有；源固定200kHz；未确认跨分辨率实证；导数FD。

**本轮判断：** 不能把该模型简写成只输入厚度。PCC高不等于幅值定量准确，归一化方向图也不等于绝对波形/成像验证。
|[P07](papers/P07.md)|Dual-Stage Fourier Neural Operator: Mitigating Spectral Bias for Enhanced Ultrasonic Guided Wave Corrosion Imaging|是|本轮核实损失为归一化MSE，不应称PDE残差训练|频域复声学信号：2×32×32 → 声速图，再由色散关系转厚度|本地原文关键章节|

## P07 Dual-Stage Fourier Neural Operator: Mitigating Spectral Bias for Enhanced Ultrasonic Guided Wave Corrosion Imaging

定位：p2–4 全部方法与实验。详见 [P07精读卡](papers/P07.md)。

|字段|内容|
|---|---|
|年份|2025 IEEE IUS；DOI 10.1109/IUS62464.2025.11201843|
|研究问题|导波腐蚀反演边界过平滑|
|方法|FNO初估+残差修正|
|输入|频域复声学信号：2×32×32|
|输出|声速图，再由色散关系转厚度|
|PDE / 物理模型|频域波动离散方程；A0色散关系|
|数据来源|数值随机腐蚀|
|数据规模|2000：1400/300/300；250×250网格；32元圆阵（p3–4）|
|网络|DS-FNO|
|是否 FNO|是|
|是否 Physics-informed|本轮核实损失为归一化MSE，不应称PDE残差训练|
|Forward / Inverse|反演|
|2D / 3D|2D面内|
|计算优势|p4主要展示代表样本改善；未见充分计时统计|
|局限性|4页会议文；缺少大规模汇总误差；未找到实物实验配置|
|作者提出 Future Work|结论未给独立明确future-work项目|
|与超声检测关系|直接腐蚀成像；与P05正演任务不同|
|与我的课题相关性|高|

**算子核查：** 跨样本有；跨阵列、频率及分辨率未验证；监督反演无需PDE空间求导。

**本轮判断：** 结论写experimental validation，但正文是numerical validation；不能翻译成已完成实物实验。
|[P08](papers/P08.md)|S0 Lamb wave scattering in plate structures using physics-enhanced TransUNet|否|是：FD物理残差|101×101厚度图 → 101×101×101面外场|本地原文关键章节|

## P08 S0 Lamb wave scattering in plate structures using physics-enhanced TransUNet

定位：p5 方法；p6–7 数据；p13–14 实验、限制。详见 [P08精读卡](papers/P08.md)。

|字段|内容|
|---|---|
|年份|正式卷期2026，Ultrasonics159:107852；DOI年份2025|
|研究问题|预测S0入射后散射的面外响应|
|方法|TransUNet+后期KL物理约束|
|输入|101×101厚度图|
|输出|101×101×101面外场|
|PDE / 物理模型|3D弹性FEM造数；KL约束主要针对面外A0分量|
|数据来源|FEM和SLDV|
|数据规模|400：350训练/50测试；100kHz；1.5mm板（p6–7）|
|网络|PTUNet|
|是否 FNO|否|
|是否 Physics-informed|是：FD物理残差|
|Forward / Inverse|正演|
|2D / 3D|输出2D空间+时间；标签由3D FEM提取|
|计算优势|少量数据下优于纯TransUNet；不能据此引用未经核实速度倍率|
|局限性|缺陷内部厚度均匀；纯S0激励困难；弱面外分量易被噪声淹没|
|作者提出 Future Work|p14神经算子、更精确方程、3D、积分约束/Chebyshev导数|
|与超声检测关系|支持跨物理模型数据与PDE损失不匹配问题|
|与我的课题相关性|高|

**算子核查：** 跨缺陷实例有；时间为输出通道，不是自动跨时间分辨率。FD计算高阶导数。

**本轮判断：** KL不描述完整S0-A0耦合，论文将面外散射A0作为约束对象。p5系数变换有量纲疑点，不能直接复制。
|[P09](papers/P09.md)|Finite difference-embedded UNet for solving transcranial ultrasound frequency-domain wavefield|否|是|不同声速模型 → 复频域声压，实/虚通道|本地原文关键章节|

## P09 Finite difference-embedded UNet for solving transcranial ultrasound frequency-domain wavefield

定位：p5–7 数据量实验；p10–11 局限/后续。详见 [P09精读卡](papers/P09.md)。

|字段|内容|
|---|---|
|年份|2024，JASA155:2257–2269|
|研究问题|减少频域波场预测的标签依赖和噪声敏感|
|方法|U-Net+最优九点FD物理损失|
|输入|不同声速模型|
|输出|复频域声压，实/虚通道|
|PDE / 物理模型|频域声学/Helmholtz型方程|
|数据来源|理想颅骨、MIDA单切片加随机血块；FD标签|
|数据规模|颅骨512训练/100测试；脑模型训练64–512、100测试（p5–7）|
|网络|FEUNet|
|是否 FNO|否|
|是否 Physics-informed|是|
|Forward / Inverse|正演|
|2D / 3D|2D空间|
|计算优势|同数据预算对比UNet；附录计时；本报告不拼接成FNO加速倍率|
|局限性|软约束不逐点保证物理；高频尺寸成本；数据/PDE同FD存在离散一致性偏好风险（后者本轮判断）|
|作者提出 Future Work|p10–11硬约束、3D、固液耦合、孔隙/衰减吸收、自适应权重|
|与超声检测关系|医学超声；方法可迁移，不能视为工业NDT验证|
|与我的课题相关性|高|

**算子核查：** 介质参数变化有；网格迁移未验证；FD残差明确；不是FNO。

**本轮判断：** 同一底图上的血块变化不等于跨不同患者颅骨泛化；医学频率/材料/边界不可直接照搬至金属导波。
|[P10](papers/P10.md)|Physics-informed neural networks for transcranial ultrasound wave propagation|否|是|空间和时间坐标 → 该介质实例的时域波场|本地原文关键章节|

## P10 Physics-informed neural networks for transcranial ultrasound wave propagation

定位：p1快照/PDE；p9–11 计算与future work。详见 [P10精读卡](papers/P10.md)。

|字段|内容|
|---|---|
|年份|2023，Ultrasonics132:107026|
|研究问题|少量早期快照预测经颅传播|
|方法|坐标PINN+PDE/快照/BC|
|输入|空间和时间坐标|
|输出|该介质实例的时域波场|
|PDE / 物理模型|二维声学波动方程|
|数据来源|谱元SEM参考解、两张早期快照|
|数据规模|三个逐渐复杂的模型情景及血块对比；不是跨介质的大样本算子集|
|网络|全连接PINN|
|是否 FNO|否|
|是否 Physics-informed|是|
|Forward / Inverse|正演|
|2D / 3D|2D空间+时间|
|计算优势|p11表1列10张场查询时间；不能忽略此前PINN优化时间|
|局限性|p9速度场或源位置变化需重训；高频训练昂贵|
|作者提出 Future Work|p9–11源条件输入、迁移、频域、3D、粘弹各向异性/孔隙介质|
|与超声检测关系|医学超声；用于理解单实例优化瓶颈|
|与我的课题相关性|高|

**算子核查：** 当前主要单实例；导数AD；改变坐标查询密度不等于神经算子跨网格训练。

**本轮判断：** 作者时间外推与跨介质算子泛化是两个问题。不能引用表1说PINN总耗时必优于SEM。
