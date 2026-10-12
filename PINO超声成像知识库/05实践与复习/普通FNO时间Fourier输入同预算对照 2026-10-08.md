---
type: experiment-record
date: 2026-10-08
status: 两组固定20轮完成，材料响应审计已保存
tags: [FNO, Fourier特征, 散射, 材料响应, 受控对照]
---
# 普通FNO时间Fourier输入同预算对照

承接 [[普通FNO散射目标多工件验证 2026-10-08]] 与 [[FNO与超声全波形反演科研计划 2026-10-07]]。本轮检查：在当前窄通道普通 FNO 与固定预算下，显式时间振荡输入是否改善材料相关散射。频率偏置只是候选解释，尚未确定为失败原因。

## 预先固定的协议

两个普通 FNO 都有八输入通道，前六通道为材料 gamma、归一化 x/y/t 和源 x/y。对照组额外通道恒为零；实验组为 sin(2π×500000×t)、cos(2π×500000×t)，t 使用物理秒。频率来自作者固定激励，没有按验证成绩搜索。

两组 width=4、4 层、modes=(16,8,64)、无 padding；seed=2 的初始权重哈希相同。Adam lr=0.001；12 个训练工件、6 个验证工件，每件四次发射；复用先前保存的全部 20 轮样本顺序，各 960 次更新，固定最终轮作主比较，不按验证挑选检查点。

目标仍为 FD(gamma)−FD(1)，基底为固定无缺陷离散正演，损失为全波场 MSE 加权重 1 的接收监督 MSE。两项使用旧训练全场 RMS=0.15921596709548344 与接收 RMS=0.25594272720229144，未改为散射 RMS。完整接收评价采用原 2000 步无缺陷波形加三次插值重建的散射，接收位置为实际 y 索引 127。

八通道零特征组是因果对照。旧六通道结果仅作历史背景，因为提升层形状与随机数消耗不同。BasicFNO 默认六通道初始化的哈希已检查兼容；两组新增特征的前六通道逐值一致，sin/cos 满足单位圆且时间频谱峰位于固定载频。

只加载训练/验证数据，不加载旧测试工件。当前仍为二维合成标量密度缩放模型，声速固定；不是弹性相控阵、实测 FMC 或材料速度反演验证。

## 评价与解释边界

报告训练/验证散射波场、散射接收及完整场相对 L2。指标按全部发射能量合并再开方，同时保留零散射预测作为参照。材料响应比较 FNO(gamma)−FNO(1) 与 FD(gamma)−FD(1)，固定基底相减抵消；均匀材料输入为诊断反事实。这不是局部梯度检查。

冻结 sin/cos 模型后，额外将两条特征置零，量化输出变化。这个移除输入不属于训练时表示，不能替代两组独立训练的受控比较，也不能仅凭变化范数宣称学到正确散射。

## 固定最终轮结果

两组各完成 20 轮、960 次更新，主要结果为预定最终轮。下表为全部 24 次验证发射合并能量的指标。

| 验证指标 | 八通道零特征 | 八通道 sin/cos |
| --- | ---: | ---: |
| 散射波场相对 L2 | 100.105% | 100.130% |
| 散射接收相对 L2 | 99.596% | 99.562% |
| 完整波场相对 L2 | 28.383% | 28.390% |
| 完整接收相对 L2 | 24.753% | 24.745% |

验证零散射参照：散射场/接收均为 100%；完整场/接收为 28.353%/24.854%。训练散射场/接收：零特征 100.044%/99.368%，sin/cos 100.034%/99.118%。

| 验证材料差分指标 | 零特征 | sin/cos |
| --- | ---: | ---: |
| 差分响应相对 L2（零响应=1） | 1.000000 | 1.000001 |
| 预测/参考差分范数 | 0.025% | 0.211% |
| 差分余弦 | -0.000939 | 0.000405 |

冻结 sin/cos 模型再移除两条特征，输出变化范数仅为参考散射波场的 3.704%、散射接收的 5.402%；移除后散射接收误差为 99.671%。这项诊断不替代独立训练的零特征组。

![[普通FNO时间Fourier输入同预算对照.png]]

图中两组场误差使用共同色阶；误差曲线接近 1 的变化不可误读为大幅收益。单个波形仅作示例，结论取全部验证发射。

当前输入特征在单种子固定预算下没有带来实质性的散射学习改善，材料响应仍很弱。训练散射也拟合不足，不能简单归结为验证泛化失败。两组的小数值差异没有重复种子或置信区间支撑，不能宣称可靠优劣；本结果也不证明时间 Fourier 特征普遍无效，更不证明或排除频率偏置。暂不开展代理 FWI。

## 补充：散射目标频谱覆盖

此前完整场的覆盖率不能直接套用到散射残差，本轮重新计算了全部训练/验证目标的全 FFT 矩形掩膜能量。

| 散射目标能量覆盖 | 训练48发射 | 验证24发射 |
| --- | ---: | ---: |
| 时间频谱范围 | 99.782% | 99.785% |
| 联合x/y/t矩形范围 | 96.155% | 95.865% |

掩膜为 x[-16,15]、y[-8,7]、t[-63,63]；它不是谱层精确 Hermitian 输出，局部支路和非线性还可产生额外频率，不能用来给出严格 FNO 误差下界。主要时间能量仍在范围内，未见仅靠缺失载频即可解释约100%散射误差的证据。特征移除确实改变了输出，材料差分仍很弱；不能把特征利用等同于有效材料条件映射。

## 下一步：同源双工件材料条件诊断

把问题缩到训练工件 000、001 的同一发射 0，已核查除 gamma 外的输入完全相同，完整接收差异为平均波形范数的 33.885%。两组仍用八通道零特征普通 FNO：一组输入真实 gamma，另一组将 gamma 换成全 1，但保留各自不同的真值。两组同初值、交替样本顺序、旧尺度、接收权重 1、Adam lr=0.001，各固定 800 次更新，每个工件各暴露 400 次。不读取验证或测试。

增加两工件散射真值的算术平均参照：在其他输入相同、等权二次损失下，它是材料无关预测的最优解。该参照只用已拟合训练真值计算，是材料无关预测的诊断下界，可作为固定训练均值基线；不能用待评价工件的真值重算均值，也不具备材料条件预测能力。真实材料组能否降到平均参照以下，以及两材料预测差分是否匹配参考差分，是主要判据。已算得平均参照监督损失0.0636916、零散射损失0.138478，前者为后者45.994%；这说明共同波形也能降低损失，下降本身不能证明学到材料区分。

这一步是条件映射可学性诊断，不能将与旧 12 工件实验的差异单独归因于工件数，也不是泛化验收。协议已保存为 `next_material_conditioning_protocol.json`，标为 planned_not_run；本轮仅完成了输入一致性、真值差异和平均参照核查，没有执行新训练。优先把基础映射做清楚，再决定扩大预算、调整容量或引入复杂结构。

## 验证与限制

最终两组检查点重新加载后，全部 24 次验证发射指标复算一致；960 次更新、20 轮排列、同初值与旧归一化通过检查。已知零散射和线性时间信号的还原/能量检查、输入频率/单位圆与材料自动微分连接、差分能量恒等式、七个作者原文件及数据/基底哈希通过。旧结果文件哈希另行核查；图像已实际查看，附件副本及新增笔记链接已检查。没有新的测试成绩、物理梯度、实测或反演验证。

## 可复现文件

预先计划：`.build/basic-fno-scatter-multi-20261008/next_time_features_protocol.json`，保留历史 planned_not_run 状态；实际运行协议与结果保存到 `.build/basic-fno-time-feature-control-20261008`。重跑须使用新输出目录。

```powershell
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/train_time_feature_control.py --data .build/operator-pilot-v3-20261008 --control .build/basic-fno-multi-control-20261008 --plan .build/basic-fno-scatter-multi-20261008/next_time_features_protocol.json --out .build/basic-fno-time-feature-control-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/audit_validation_material_response.py --data .build/operator-pilot-v3-20261008 --run .build/basic-fno-time-feature-control-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/audit_time_feature_use.py --data .build/operator-pilot-v3-20261008 --run .build/basic-fno-time-feature-control-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/audit_scatter_target_spectrum.py --data .build/operator-pilot-v3-20261008 --run .build/basic-fno-time-feature-control-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/summarize_time_feature_control.py --run .build/basic-fno-time-feature-control-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/verify_time_feature_control.py --data .build/operator-pilot-v3-20261008 --run .build/basic-fno-time-feature-control-20261008 --control .build/basic-fno-multi-control-20261008
```

方法灵感来自 [Tancik 等 Fourier Features](https://arxiv.org/abs/2006.10739)，其对象是 MLP 高频低维回归，不能直接保证本 FNO 的效果。[Kong 等频域 FNO 频率偏置工作](https://arxiv.org/abs/2503.02023v2) 的频域方程与残差网络也与本实验不同；本轮仅检验输入特征。

## 后续执行

双工件材料区分诊断已完成，实际执行及最终结果见 [[普通FNO同源双工件材料区分诊断 2026-10-08]]。本页保留此前时间特征实验和当时预先计划。
