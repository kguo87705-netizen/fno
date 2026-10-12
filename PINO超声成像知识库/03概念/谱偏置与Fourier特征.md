---
type: concept
tags: [概念, 超声]
---
# 谱偏置与Fourier特征

普通网络可能更易拟合平缓低频成分，高频波动更难训练；Fourier 特征通过坐标编码改善表示。

## 最小知识

常见示意编码 $\gamma(x)=[\sin(2\pi Bx),\cos(2\pi Bx)]$。频率尺度选取会影响可表达频段、优化与噪声敏感性。

学习中的谱偏置、FNO 频域截断、测量欠采样是三种不同问题。增加保留模态不自动解决损失失衡；增加采样点也不自动解决优化困难。

## 检验理解

一个波场高频误差大，是否足以证明由谱偏置造成？

> [!tip]- 回答后展开
> 不足；应排查采样、网格、模态截断、标签误差及优化因素，并用消融区分。

## 去论文里验证

[[P02 PIFNN瞬态波与自适应权重]] · [[P05 双阶段FNO导波散射]] · [[P07 双阶段FNO腐蚀成像会议]]

以上公式和例子为教学示意，具体论文符号、近似条件和实现以原文为准。关联阅读来源：[P02来源](https://www.sciencedirect.com/science/article/pii/S0952197626002757) · [P05来源](https://www.sciencedirect.com/science/article/pii/S0963869526001301) · [P07来源](https://confcats-event-sessions.s3.us-east-1.amazonaws.com/ius25/uploads/IUS_2025_Program_v16.pdf)。算子概念另见 [[B01 FNO基础论文]] 与 [[B02 PINO基础论文]]。

返回 [[概念索引]] · [[开始学习]]
