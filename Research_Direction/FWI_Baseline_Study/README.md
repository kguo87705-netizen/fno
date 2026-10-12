# 二维传统 FWI 数值冒烟实验

`baseline.py` 独立实现了 Singh 等人 2025 年工作使用的密度缩放标量波方程，用于检查从正演、接收到材料梯度和优化的完整数据流。[论文](https://link.springer.com/article/10.1007/s00466-025-02600-w) · [作者代码](https://zenodo.org/records/14616677)

运行：

```powershell
python baseline.py --out ..\..\.build\fwi-baseline-smoke
python baseline.py --iterations 80 --out ..\..\.build\fwi-baseline-smoke-80
python baseline.py --reference-refinement 2 --iterations 80 --out ..\..\.build\fwi-baseline-refined-80
python baseline.py --reference-refinement 2 --calibrate-homogeneous-gain --iterations 80 --out ..\..\.build\fwi-baseline-refined-gain-80
python grid_convergence.py --out ..\..\.build\fwi-grid-convergence
python aperture_sampling.py --out ..\..\.build\fwi-aperture-sampling
python aperture_inversion.py --sigma-mm 1.5 --out ..\..\.build\fwi-aperture-inversion-1p5
python aperture_inversion.py --sigma-mm 3 --out ..\..\.build\fwi-aperture-inversion-3
python aperture_inversion.py --case small --acquisition indexed --out ..\..\.build\fwi-holdout-small-indexed
python aperture_inversion.py --case small --sigma-mm 1.5 --out ..\..\.build\fwi-holdout-small-1p5
python aperture_inversion.py --case small --sigma-mm 3 --out ..\..\.build\fwi-holdout-small-3
python aperture_inversion.py --case double --acquisition indexed --out ..\..\.build\fwi-holdout-double-indexed
python aperture_inversion.py --case double --sigma-mm 1.5 --out ..\..\.build\fwi-holdout-double-1p5
python aperture_inversion.py --case double --sigma-mm 3 --out ..\..\.build\fwi-holdout-double-3
python compare_holdouts.py --build ..\..\.build --out ..\..\.build\fwi-holdout-comparison
python aperture_inversion.py --case shallow --sigma-mm 3 --out ..\..\.build\fwi-depth-shallow-3
python aperture_inversion.py --case deep --sigma-mm 3 --out ..\..\.build\fwi-depth-deep-3
python compare_depths.py --build ..\..\.build --out ..\..\.build\fwi-depth-comparison
```

使用已有的 PyTorch 和 Matplotlib，不需要下载作者的案例包。输出 `report.json` 包含两种步长的方向导数检验和损失历史，`fields.pt` 含真实材料、重建材料与接收波形，`smoke_check.png` 显示材料图、波形和损失。运行时按可用性选择 CUDA 或 CPU；`--cpu` 强制使用 CPU。

反演使用 $64\times32$ 网格、两个声源、17 个接收点和 420 时间步。默认用同一网格生成观测，存在 inverse crime。`--reference-refinement 2` 改用 $128\times64$ 网格生成观测，检测离散误差，但仍用同一差分格式和理想物理。`--calibrate-homogeneous-gain` 用无缺陷样本的接收数据估计一个共享幅度比例；这是单独的消融，不使用缺陷真值进行校准。以上都不是论文复现或真实金属相控阵验证。

方程与接收量详见 Obsidian 的 [[相控阵FWI基线选择与复现卡 2026-10-07]]。此脚本假定反射边界和理想点激励，暂未包含弹性模式转换、阵元响应或噪声。

`grid_convergence.py` 只做正演诊断：比较 1、2、4 倍空间网格的无缺陷／有缺陷波形，并在 1、2 倍网格上把时间步减半。幅度和整数采样时移只用无缺陷波形拟合。输出 `report.json` 和 `grid_convergence.png`；4 倍网格仍是有限精度数值结果，不能视作连续真值。

`aperture_sampling.py` 分别比较固定物理坐标双线性采样、仅平滑发射、仅平滑接收以及两端平滑的跨网格波形；`aperture_inversion.py` 对两种高斯宽度做 80 次跨网格反演。高斯分布只是合成阵元模型，宽度未按真实硬件标定，3 mm 标准差属于强平滑对照。数值和图分别保存在所列输出目录，结果详见 Obsidian 的 [[相控阵源接收有限孔径消融 2026-10-07]]。

`aperture_inversion.py --case small|double` 在先前未试的缺陷几何上复用固定阈值 0.65 与 80 次迭代；`compare_holdouts.py` 汇总并绘制对比。结果详见 [[传统FWI小缺陷与双缺陷留出检查 2026-10-07]]。这些几何留出不等同于独立求解器或实验泛化。

`aperture_inversion.py --case shallow|deep` 保持原椭圆缺陷尺寸、只改变中心深度为 15 或 35 mm。对每个深度分别使用 `--acquisition indexed`、`--sigma-mm 1.5`、`--sigma-mm 3` 运行；中层 25 mm 复用最初的三组结果。`compare_depths.py` 读取九组输出生成同色标对照，详见 [[传统FWI缺陷深度对照 2026-10-07]]。此模型无材料吸收且使用反射边界，深度趋势不能解释为真实衰减规律。

2026-10-08 的边界与时间窗诊断：`boundary_window_diagnostic.py --out ..\..\.build\fwi-boundary-window-20261008` 比较相同步长下的原域与扩大侧面／底面域，并分析各时段的缺陷信号。`aperture_inversion.py --case shallow|original|deep --sigma-mm 3 --boundary-domain expanded` 保持原材料未知量而将新增背景固定为 1；相应输出保存到 `.build/fwi-boundary-expanded-{shallow,original,deep}-20261008`。`validate_expanded_gradient.py` 核对权重嵌入与方向导数；`compare_boundaries.py` 汇总三档深度。扩大域改变有限工件边界问题，不能自动认定它是更真实的工业模型。详见 [[传统FWI边界与时间窗诊断 2026-10-08]]。
