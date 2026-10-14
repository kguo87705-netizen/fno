# 独立评价与质量护栏：首轮

负责人批准的范围：T1独立几何与细网格观测、35步计时、T4尺度计时、EXP-RT100检查点复现、原生epoch0可行性、验证阈值冻结。总上限8小时；本轮禁止测试集反演、T3和FNO训练。

运行基础提交为`137095a`，本目录为其后新增实现。当前执行状态见`.build/fwi-quality-guards-20261013/round_status.json`；最终摘要见`first_round_summary.json`，只有生成后才代表首轮汇总完成。

## 数据入口

- `accepted_geometries.json`是最终连续几何；`geometries.json`保留最初候选。
- 首次连续距离检查发现3个候选违反约定，原始文件保留，重生成文件在`corrected_data/`。
- **必须使用`common.data_path(case)`或`geometry_data_overrides.json`读取接受的数据，不能直接遍历`data/`作为最终数据集。**
- 粗格点真值和标签是256×128，观测为四源×24接收器×2000采样，来自511×255、4000×15 ns细网格。
- 初始梯度由粗网格与细网格观测计算。新网格非嵌套，源用线性分配、接收用线性插值维持相同物理坐标。

## 执行与结果

`generate_independent.py`生成原始候选数据；`correct_geometry_margins.py`保留失败记录并生成接受版本；`round_driver.py`按负责人顺序运行其余首轮内容。

`invert.py`统一作者迁移U-Net设置，仅改变检查点。原生epoch0遵循Pretraining.py的完整初始化，不另行改输出头。`retrain_checkpoints.py`保留epoch0/10/30/100，并严格比对旧EXP-RT100最终权重。

`evaluate.py`只在seed30验证集选阈值并写`frozen_thresholds.json`。其他种子和测试集不得重新选阈值。分组同时报告缺陷检出率和缺陷内像素召回；Wilson区间按缺陷计数，未处理同板相关性。

`benchmark_scale.py`保留原求解器的完整波场存储，不引入检查点。物理侧四源批次，R2单输入批次；物理输出Nt帧，FNO固定500帧。两侧计时不能直接相除宣称同任务加速。两次OOM后T4停止，未测项保留为未测。

`finalize_reports.py`仅汇总保存结果、核对数据结构并追加T1/T2/T4笔记，不执行测试反演。它只应在本轮结束后运行一次，避免重复追加。

所有H2判断须等待冻结阈值后的独立测试和五种子配对比较；验证集校准结果不作为H2确认。禁止为了过关修改阈值或挑最佳迭代。
