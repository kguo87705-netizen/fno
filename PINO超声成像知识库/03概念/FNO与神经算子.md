---
type: concept
tags: [概念, 超声]
---
# FNO与神经算子

神经算子学习函数到函数的映射，如整张厚度分布到整片波场；FNO 用频域参数化处理全局信息。

## 最小知识

教学层表达：$v_{l+1}=\sigma(W_l v_l+\mathcal F^{-1}[R_l\mathcal F(v_l)])$。$R_l$ 表示频域可学习变换；典型实现只在选定模态上计算。

要检查输入函数族、采样网格和跨分辨率测试。能接收不同网格大小不代表对任意网格、频率和缺陷都准确。

基础入口：[[B01 FNO基础论文]]。

## 检验理解

Fourier 特征映射与 FNO 的 Fourier 层一样吗？

> [!tip]- 回答后展开
> 不同；前者常把坐标编码为正余弦特征，后者对隐藏场做频域算子变换。

## 去论文里验证

[[P01 U-FNO超声全波形直接反演]] · [[P05 双阶段FNO导波散射]] · [[P06 PI-DeepUFNet不规则缺陷波场]]

以上公式和例子为教学示意，具体论文符号、近似条件和实现以原文为准。关联阅读来源：[P01来源](https://www.sciencedirect.com/science/article/abs/pii/S0888327026003365) · [P05来源](https://www.sciencedirect.com/science/article/pii/S0963869526001301) · [P06来源](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6967141)。算子概念另见 [[B01 FNO基础论文]] 与 [[B02 PINO基础论文]]。

返回 [[概念索引]] · [[开始学习]]
