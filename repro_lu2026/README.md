# Lu et al. 2026 超声FWI缩小版复现

## 当前进度

- **M0完成**：MATLAB R2024b、RTX5070、k-Wave1.4.1；所有本次输出在E盘。
- **M1完成**：同一EMNIST模型的完整64发射参考FMC已保存。计算及检查点251.64秒，含启动/最终保存260.93秒；全局互易相对L2为1.76e-6，最坏道对8.46e-6。
- **并行可行性完成**：相同6炮在1/2/3进程下分别24.30/20.71/20.47秒；与参考波形误差均为0。暂推荐2进程，3进程额外收益未证明稳定。
- **Deepwave对比完成**：Python3.12.15、torch2.11.0+cu128、Deepwave0.0.27均安装在E盘，单炮GPU验证通过；一次64炮传播及回传2.145秒，生成保存2.627秒，整卡峰值2036MiB，无OOM。原始相对L2约1.00，全局幅度+时移标定后0.541836，相关系数均值0.828967，互易性2.72e-6。当前建议训练集仍用k-Wave双进程，先解决物理源等价问题再考虑替换。
- **原26文件已成功推送**：`713fdc4d6d7ceec9f7c6ba4a0a98e440cda0d1c7`。新增Deepwave报告、代码、配置、图像及状态更新留在本地待清单确认。
- M2批量数据、M3加载、M4原版FNO、M5评估尚未执行。

## 报告

- [Deepwave完整对比、预算和建议](reports/deepwave_comparison.md)
- [Deepwave执行前协议](reports/deepwave_protocol.md)
- [Deepwave实测数值与环境版本](configs/deepwave_measurements.json)
- [推送与安装网络阻塞记录](reports/installation_network_block.md)
- [M0与单次发射](reports/m0_and_single_tx.md)
- [本轮预注册协议](reports/protocol.md)
- [完整FMC和互易性](reports/kwave_reference.md)
- [并行测试及300/500样本耗时估算](reports/concurrency.md)
- [Deepwave安装前方案](reports/deepwave_install_plan.md)
- [仓库交付和验证边界](reports/repository_handoff.md)
- [待提交文件清单](reports/submission_manifest.md)

![已确认的声速模型](figures/confirmed_model.png)

## 文件与运行方法

所有本次新增版本化内容均在本子目录，未改动仓库原文件。configs/*.json包含参数及当前机器的E盘路径；迁移机器时修改路径配置。数据、MATLAB和k-Wave依赖不打包进Git。

1. `code/prepare_emnist_model.py`：从官方EMNIST Letters IDX gzip转置正立、28→56插值、补零200并映射声速。需要numpy/Pillow；示例：E盘环境的python执行 `code/prepare_emnist_model.py configs/m1_single_tx.json`。输入gzip需先置于配置指定的本地路径。已有相同标签只校验，不覆盖。
2. `code/run_m1_single_tx.ps1`与`code/m1_single_tx.m`：一次发射、场诊断和计时。PowerShell示例：`& .\code\run_m1_single_tx.ps1 -ConfigPath .\configs\m1_single_tx.json`。请传绝对配置路径以免MATLAB工作目录切换造成歧义。
3. `code/run_kwave_group.ps1`与`code/kwave_fmc_worker.m`：同一模型的指定发射集合，可用独立MATLAB进程并行。默认配置指向当前本机路径；在MATLAB启动前设置E盘缓存，并创建唯一结果目录。只在用户授权后执行。例如完整参考：`& .\code\run_kwave_group.ps1 -Processes 1 -Transmitters (1..64) -RunName kwave_reference`。少量测试可把Transmitters设为`@(1,13,26,39,52,64)`，Processes设为1/2/3。
4. `code/analyze_reference.m`：读取参考运行目录，计算2016对互易误差、检查旧第32炮一致性并生成图；MATLAB调用`analyze_reference(reference_run_dir,figure_dir)`。
5. `code/analyze_concurrency.m`：读取三个运行目录并与完整参考逐元素比较，输出吞吐量分析；MATLAB调用`analyze_concurrency(reference_run_dir,{group1,group2,group3},out_json,figure_dir)`。

并行工作器使用单炮MAT中的固定c/label/source_wavelet/time作为输入，以保证本轮参考和吞吐量比较完全相同；它不是通用批量模型生成器。上述MAT需先通过单炮脚本生成，或使用本机现有结果。图像分析脚本可能覆盖指定输出目录中的同名PNG，重分析时应提供新的图目录。

6. `code/run_deepwave.ps1`设置项目内缓存，创建唯一结果目录并调用`code/deepwave_compare.py`。先运行`-Stage smoke`，人工核验通过后才运行`-Stage full`，两者均必须传`-ConfigPath E:\FNO_research\repos\fno_light\repro_lu2026\configs\deepwave_comparison.json`。环境解释器固定在配置root下的`envs/deepwave/Scripts/python.exe`，不自动安装或编译。full执行64炮对照和分析，不生成多模型训练集；仅OOM时允许16炮分批。

## 时间与物理约定

常密度1000kg/m³，声速1500–5800m/s，只做纵波声学模型。56像素支撑宽11.2mm是依据Lu图5的实现假设。四边外置20点PML、无底面反射是对原文未明确上下边界的处理。

dt=5ns，内部4000点；输出每4点抽取，1000点覆盖0–19.98μs。t=0为采集启动；Ricker解析峰值延迟1.333333μs，离散峰1.335μs。后续实测数据须标定触发和换能器延迟，不自动视为对齐。
