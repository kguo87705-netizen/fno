# Future Work Mining

人工核查下表与机器检索命中分开。`however`等只是定位词，不等于作者提出新方向。所有内容均转述，保留页码；不复制长段原文。

|论文|定位|作者明确提出 / 明确讨论|本轮推断|
|---|---|---|---|
|[B01](papers/B01.md)|p9 Discussion|结合数值求解器减少数据需求；探索共享参数循环结构。|超声小样本研究应对照物理约束及数值基线。|
|[B02](papers/B02.md)|p17 Conclusion；p8–9 derivatives|高维FFT扩展困难；精度/复杂度与优化方法；跨几何迁移、预训练库。|要把残差离散误差和实例微调成本纳入评价。|
|[P01](papers/P01.md)|p13–15 Discussion/limitations/conclusion|实验信号更精确校准；多样化数据、迁移、物理约束；更准确弹性物理及3D。|以探头/耦合/时间零点受控偏移设计域适应，而非只加白噪声。|
|[P02](papers/P02.md)|p12 conclusion|Fourier特征初始化、非线性和高维波动方程。|NTK机制可借鉴，但FNO中要比较额外梯度计算开销。|
|[P03](papers/P03.md)|p8 Results and discussion / conclusion|本轮未发现明确future-work清单；作者明确讨论仿真/实验幅值偏差。|比较完整复场与极性输入的信息损失、鲁棒性及定量精度。|
|[P04](papers/P04.md)|在线摘要|无法从摘要可靠提取future work或实验局限。|取得全文后核实分辨率判据，不能按标题认定超分辨上限。|
|[P05](papers/P05.md)|p13–14 §5.2|多模态与Navier方程、3D；少/弱标签；真实电化学腐蚀与高分辨厚度测量。|先做标签预算曲线及独立求解器核验，后做复杂腐蚀。|
|[P06](papers/P06.md)|p38–40 实验与结论|作为后续定量反演/数字孪生的正演基础；更换离散物理算子推广其他物理场。|需要确认测试样本口径与初始场获取方式；归一化方向图之外补绝对相位/幅值验证。|
|[P07](papers/P07.md)|p3–4 numerical validation / conclusion|未见独立future-work段；正文实验覆盖为数值。|补试件外测试、误差分布及与单阶段大模型的公平比较。|
|[P08](papers/P08.md)|p14 §5.2|更复杂厚度变化、神经算子、更精确方程、3D、积分形式约束/Chebyshev高阶导数。|后续P05/P06已推进部分路线，不能把2025/2026前文的future work原封不动当空白。|
|[P09](papers/P09.md)|p10–11 §IV.B|硬约束；3D、固液耦合、孔隙介质、吸收衰减；自适应超参数。|软残差下降未必带来独立参考误差下降，可做小样本对照。|
|[P10](papers/P10.md)|p9–11 §4.2/4.3/5|源位置条件输入、迁移、频域；3D和更复杂介质；更广速度模型。|P09已推进频域路线；不把后续已完成的方向重复当新gap。|

## 关键词检索记录

范围：11篇编号英文原文全文（包含参考文献，因此计数仅用于导航）。词表为用户列出的全部词。页码命中保存在下面，不把参考文献标题误作本论文研究。

### B01

|关键词|命中PDF页（全文匹配）|
|---|---|
|future work|无字面匹配；不代表无相近论述|
|limitation|1, 9|
|however|1, 6, 9|
|remains challenging|无字面匹配；不代表无相近论述|
|difficult|无字面匹配；不代表无相近论述|
|computational cost|6|
|expensive|4, 9|
|generalization|5|
|resolution|1, 2, 3, 5, 6, 7, 8, 9, 11, 13, 14|
|irregular geometry|无字面匹配；不代表无相近论述|
|inverse problem|1, 3, 4, 6, 9, 10, 11, 15, 16|

### B02

|关键词|命中PDF页（全文匹配）|
|---|---|
|future work|17|
|limitation|1, 4, 14, 16, 17, 18|
|however|3, 4, 7, 8, 9, 12, 13, 14, 15, 25, 26|
|remains challenging|无字面匹配；不代表无相近论述|
|difficult|4, 5|
|computational cost|8, 14|
|expensive|1, 9|
|generalization|1, 2, 3, 6, 7, 10, 13, 14, 22, 23|
|resolution|1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 20, 22, 23, 25, 26, 27|
|irregular geometry|无字面匹配；不代表无相近论述|
|inverse problem|2, 4, 10, 16, 17, 20, 26|

### P01

|关键词|命中PDF页（全文匹配）|
|---|---|
|future work|15|
|limitation|3, 14|
|however|1, 2, 3, 4, 6, 11, 13, 15|
|remains challenging|14|
|difficult|2, 4, 14|
|computational cost|1, 3, 9, 13, 14|
|expensive|2|
|generalization|1, 3, 7, 8, 13, 14, 15, 16|
|resolution|1, 2, 3, 4, 6, 7, 15, 16|
|irregular geometry|无字面匹配；不代表无相近论述|
|inverse problem|1, 3, 14, 15|

### P02

|关键词|命中PDF页（全文匹配）|
|---|---|
|future work|无字面匹配；不代表无相近论述|
|limitation|1, 2, 7, 11|
|however|2, 3|
|remains challenging|无字面匹配；不代表无相近论述|
|difficult|2, 3, 13|
|computational cost|无字面匹配；不代表无相近论述|
|expensive|2|
|generalization|2, 6, 13|
|resolution|无字面匹配；不代表无相近论述|
|irregular geometry|无字面匹配；不代表无相近论述|
|inverse problem|2, 12, 14|

### P03

|关键词|命中PDF页（全文匹配）|
|---|---|
|future work|无字面匹配；不代表无相近论述|
|limitation|2, 8|
|however|1, 2, 5, 8|
|remains challenging|无字面匹配；不代表无相近论述|
|difficult|1, 2|
|computational cost|无字面匹配；不代表无相近论述|
|expensive|无字面匹配；不代表无相近论述|
|generalization|1, 5, 8|
|resolution|1, 2, 5, 8, 9|
|irregular geometry|无字面匹配；不代表无相近论述|
|inverse problem|无字面匹配；不代表无相近论述|

### P05

|关键词|命中PDF页（全文匹配）|
|---|---|
|future work|14|
|limitation|1, 2, 3, 13, 14|
|however|1, 2, 7|
|remains challenging|1|
|difficult|7|
|computational cost|无字面匹配；不代表无相近论述|
|expensive|1, 2|
|generalization|2, 7, 8, 10|
|resolution|4, 5, 15|
|irregular geometry|无字面匹配；不代表无相近论述|
|inverse problem|15|

### P06

|关键词|命中PDF页（全文匹配）|
|---|---|
|future work|无字面匹配；不代表无相近论述|
|limitation|6|
|however|4, 29|
|remains challenging|无字面匹配；不代表无相近论述|
|difficult|5|
|computational cost|无字面匹配；不代表无相近论述|
|expensive|无字面匹配；不代表无相近论述|
|generalization|21, 24, 26, 32, 39|
|resolution|4, 6, 7, 20, 23, 28, 32, 41|
|irregular geometry|无字面匹配；不代表无相近论述|
|inverse problem|7, 42|

### P07

|关键词|命中PDF页（全文匹配）|
|---|---|
|future work|无字面匹配；不代表无相近论述|
|limitation|1|
|however|1|
|remains challenging|无字面匹配；不代表无相近论述|
|difficult|无字面匹配；不代表无相近论述|
|computational cost|无字面匹配；不代表无相近论述|
|expensive|无字面匹配；不代表无相近论述|
|generalization|1, 4|
|resolution|无字面匹配；不代表无相近论述|
|irregular geometry|无字面匹配；不代表无相近论述|
|inverse problem|2|

### P08

|关键词|命中PDF页（全文匹配）|
|---|---|
|future work|14|
|limitation|1, 2, 9, 14|
|however|2, 3|
|remains challenging|无字面匹配；不代表无相近论述|
|difficult|1, 14|
|computational cost|1, 2, 5, 6, 14|
|expensive|无字面匹配；不代表无相近论述|
|generalization|2, 3, 7, 8, 9, 10, 14|
|resolution|2, 3, 5, 6|
|irregular geometry|无字面匹配；不代表无相近论述|
|inverse problem|16|

### P09

|关键词|命中PDF页（全文匹配）|
|---|---|
|future work|10, 11, 12|
|limitation|1, 10, 11|
|however|1, 2, 6, 8, 10, 11|
|remains challenging|无字面匹配；不代表无相近论述|
|difficult|无字面匹配；不代表无相近论述|
|computational cost|1, 2, 10|
|expensive|11|
|generalization|8|
|resolution|1|
|irregular geometry|无字面匹配；不代表无相近论述|
|inverse problem|13|

### P10

|关键词|命中PDF页（全文匹配）|
|---|---|
|future work|8, 11|
|limitation|1, 2, 8|
|however|1, 2, 4, 5, 6, 7, 8, 9|
|remains challenging|无字面匹配；不代表无相近论述|
|difficult|1, 2, 3, 4, 5, 8|
|computational cost|1|
|expensive|无字面匹配；不代表无相近论述|
|generalization|1, 2, 4, 9, 11|
|resolution|1|
|irregular geometry|无字面匹配；不代表无相近论述|
|inverse problem|2, 12|
