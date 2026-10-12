---
type: concept
tags: [概念, 超声]
---
# 波动方程与Helmholtz方程

先看论文求的是时间变化的场还是单一频率的复振幅，并确认声学近似是否适用。

## 最小知识

教学简化：常密度、无耗散标量介质中，$p_{tt}-c^2\nabla^2p=s$。取 $p=\operatorname{Re}(P e^{-i\omega t})$，则可写 $\nabla^2P+(\omega/c)^2P=-S/c^2$。

时域需要初始与边界信息，频域求解需源与边界信息。一般非均匀密度、弹性波、耗散或变厚度板不能直接套此简式。此处是教学推导，不宣称为任一论文的原始方程。

## 检验理解

频域结果只有幅值时，能否完整恢复单频时域信号？

> [!tip]- 回答后展开
> 通常不能，还需要相位。实部和虚部共同表示复场。

## 去论文里验证

[[P02 PIFNN瞬态波与自适应权重]] · [[P09 FEUNet经颅频域波场]] · [[P10 PINN经颅波传播]]

以上公式和例子为教学示意，具体论文符号、近似条件和实现以原文为准。关联阅读来源：[P02来源](https://www.sciencedirect.com/science/article/pii/S0952197626002757) · [P09来源](https://pubmed.ncbi.nlm.nih.gov/38536062/) · [P10来源](https://www.sciencedirect.com/science/article/pii/S0041624X23001026)。算子概念另见 [[B01 FNO基础论文]] 与 [[B02 PINO基础论文]]。

返回 [[概念索引]] · [[开始学习]]
