# 作者传统 FWI 公开代码运行目录

来源：[Singh 等论文](https://link.springer.com/article/10.1007/s00466-025-02600-w)，[Zenodo v3，记录 14616677](https://zenodo.org/records/14616677)。2026-10-08 下载必要脚本、设置和约 2.9 MB 的案例 RAR；`source_manifest.json` 保存 MD5、SHA256 和大小。七个下载文件的 MD5 已与 Zenodo 清单逐一核对。原始 `.py`、CSV、README 和 RAR 的字节保留。

数据包用 Windows 自带 `tar` 解包。HDF5 后端 `tables` 安装在 `.build/author-baseline-env`，该虚拟环境通过 `--system-site-packages` 使用已有 PyTorch/NumPy/pandas/Matplotlib；新增依赖清单见 `local_environment_requirements.txt`，各次报告记录实际核心版本。

从工作区根目录运行：

```powershell
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_selected_case.py --case 1 --epochs 35 --out .build/author-singh-case1-35epochs-20261008
```

另外三例依次使用 `--case 2`、`3`、`4` 和各自输出目录。包装器只在内存替换案例选择、迭代次数、读写路径与输出目录存在性检查；求解器、伴随公式、Adam、裁剪和配置保持原始代码。使用 35 次时迭代预算与公开脚本相同。`run_report.json` 保存全部替换和校验值；原脚本生成的 HDF 历史、文本曲线和 SVG 仍保存，另外补算更新后的波形代价、保存 `fields.pt`。

```powershell
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/summarize_author_runs.py --build .build --out .build/author-singh-summary-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/check_author_adjoint.py --out .build/author-singh-summary-20261008/adjoint_check.json
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/audit_time_costs.py --out .build/author-singh-summary-20261008/time_axis_costs.json
```

运行后发现原始代价函数的积分维度问题：残差为 `[1,4,24,2000]`，原代码 `axis=2` 沿接收维积分，时间在最后一维。`audit_time_costs.py` 保留其他归一化、仅改评价的积分轴，重新评价四例的初始和最终材料场；不修改原始优化轨迹。原始代价和修正评价分别保留。按时间积分，最终失配为初始的 21.5%–34.1%，但作者口径材料误差增长到 1.65–2.13 倍。

案例 1 初始场的三种方向核查中，正确时间积分与作者梯度投影的相对差异约为 28.8%、0.7%、3.7%，未满足本项目 2% 的离散梯度验收要求。检查局限于这三种方向和 float32 设置，不能据此概括全部迭代状态；下一步需建立同一离散正演与正确损失的自动微分参考，再定位连续伴随与离散梯度的差异。

公开设置 `Nx=255, Ny=127` 对应内部材料 $256\times128$、含 ghost cells 的缓冲区 $258\times130$；源数 4、接收数 24、2000 步、30 ns，总时长 60 μs。**论文第 2.2 节另描述反演为 1000 步、60 ns。** 因此当前运行应称为公开 v3 代码案例执行；与论文图表精确一致性需要继续核对。

这套作者实现与 `Research_Direction/FWI_Baseline_Study` 的独立冒烟程序分别保存：前者用原案例、近空洞参数和连续伴随公式；后者是较粗的理想化合成诊断。不能混用二者的数值结果或阈值，也不能视为金属弹性波实测验证。

## 可微离散参考与受控反演

`discrete_reference.py` 保留原空间模板、面系数与系数值，将时间状态及反射索引函数式表达，用完整时间反传和 50 步 checkpoint 构建离散导数。接收波形在案例 1 初始场和作者第 5 次更新场逐点匹配；完整 2000 步的四方向、两幅度检查在 float32 与 float64 都通过 2% 阈值。该参考仅证明同一离散实现的导数，不能证明真实物理模型正确。

```powershell
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/audit_discrete_gradient.py --steps 2000 --dtype float32 --out .build/author-discrete-full32-initial-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/audit_discrete_gradient.py --steps 2000 --dtype float64 --updates 5 --adjoint-batch 1 --out .build/author-discrete-full64-update5-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/summarize_discrete_audit.py --out .build/author-discrete-gradient-summary-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_discrete_fwi.py --lr .08 --out .build/author-discrete-fwi-case1-35epochs-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_discrete_fwi.py --lr .02 --out .build/author-discrete-fwi-case1-lr002-35epochs-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/compare_discrete_controls.py --out .build/author-discrete-gradient-summary-20261008
```

`--adjoint-batch 1` 对每次发射独立计算原伴随再平均，用于完整双精度复核；自动微分仍涵盖全部四次发射。`--updates 5` 从保存的原案例历史读入第 5 次更新场。新反演使用标准每轮清空梯度、一次反传、相同梯度掩膜和边界约束；没有额外冻结物理边界层，也未改写作者代码。

作者连续伴随与离散参考的夹角约 39.47°/42.14°，阵列侧两层贡献约 99.4%/99.6% 的差异平方和。在固定 35 次更新下，离散梯度 lr=0.08 的失配比为 0.3791、材料误差比为 2.6099；lr=0.02 的失配比为 0.4136、材料误差比为 2.0796，后者每步失配下降。精确梯度不自动解决材料恢复；单案例对照不用于选择通用超参数。

## 四案例扩展与算子数据接口

`audit_discrete_gradient.py --case 2/3/4` 支持其余目标；`run_extended_audits.py` 顺序执行六组完整双精度检查。四案例初始与第 5 次更新场全部通过，最大方向差异 0.0151%。原作者文件继续保持字节不变。

```powershell
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/run_extended_audits.py
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/generate_operator_pilot.py --stride 4 --out .build/operator-pilot-v2-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/audit_time_sampling.py --stride 4 --out .build/operator-pilot-v1-20261008/time_sampling_check.json
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/check_operator_pilot.py --folder .build/operator-pilot-v2-20261008 --time-audit .build/operator-pilot-v1-20261008/time_sampling_check.json
```

数据生成器拒绝覆盖现有清单，重跑应指定新输出目录。8 个新合成工件按稳定编号分为 4/2/2 个训练/验证/测试工件，四次发射继承工件划分，原作者案例保留为外部诊断。`operator_data.py` 提供输入六通道 `[gamma,x/Lx,y/Ly,t/T,sx/Lx,sy/Ly]` 与单源波场输出；v2 形状为 `[6,256,128,500]` / `[1,256,128,500]`。材料与时间解码的导数连接已核查，归一化只取训练工件。

`time_reconstruction.py` 实现固定局部三次时间插值、已知 t=0 零初值；四步抽样重建在八个作者目标/状态配对中，波形误差最高约 0.0637%、损失变化 0.0469%、材料梯度 L2 差异 0.2395%。数据保留全部 2000 步接收波形。v1 的八步抽样只作诊断保存，v2 是当前接口版本。这些是数据处理链结果，尚未训练或验证神经网络。

## 普通 FNO 首次训练

24 工件数据位于 `.build/operator-pilot-v3-20261008`，按工件拆分为 12/6/6，每件四次发射。前八工件的划分与材料保持一致。`basic_fno.py` 为四层普通 x/y/t FNO，width=4、modes=(16,8,64)、padding=0，不使用物理损失或无缺陷波场捷径。

```powershell
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/generate_operator_pilot.py --count 24 --stride 4 --out .build/operator-pilot-v3-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/export_homogeneous.py --out .build/operator-pilot-v3-20261008/homogeneous.pt
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/train_basic_fno.py --data .build/operator-pilot-v3-20261008 --epochs 6 --out .build/basic-fno-v1-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/evaluate_basic_fno.py --data .build/operator-pilot-v3-20261008 --run .build/basic-fno-v1-20261008
.build/author-baseline-env/Scripts/python.exe Research_Direction/Author_Baseline_Singh_Zenodo14616677/refine_fno_gradient.py --data .build/operator-pilot-v3-20261008 --run .build/basic-fno-v1-20261008
```

固定六 epoch、288 次更新；仅用验证波场 MSE 选第六轮检查点。冻结测试 24 次发射的波场/接收误差为 95.18%/97.63%，无缺陷参照为 36.04%/31.20%。单工件、单发射、均匀初值的材料梯度夹角 92.68°、范数比 3.10%，暂不接入 FWI。

弱网络梯度的 float32 中心差分曾出现 16.95% 的数值差异；冻结同一权重、保留复参数虚部、用 float64/complex128 中心差分对保存的 float32 AD 方向复核，三档差异约 0.0162%，通过 2% 数值检查。这是内部求导链核查，不证明物理梯度准确。下一轮仅用训练/验证数据开展单样本过拟合与频带诊断；不利用当前测试集反复调参。完整记录见知识库“普通FNO首次训练与失败诊断 2026-10-08”。

## 单训练样本与频谱/边界诊断

`diagnose_basic_fno.py --data .build/operator-pilot-v3-20261008 --out .build/basic-fno-single-sample-20261008` 在首个训练工件发射 0 上，从相同种子随机初始化固定更新 400 次；协议在运行前保存。不读取测试集，最终检查点不按结果选择。波场误差降至 18.05%，接收仍为 64.14%，尚未完成过拟合。

`audit_training_trace_spectrum.py --run .build/basic-fno-single-sample-20261008` 与 `audit_training_boundary.py --data .build/operator-pilot-v3-20261008 --run .build/basic-fno-single-sample-20261008` 分别核查接收频谱与物理网格分区。训练/验证真值约 99.89% 时间谱能量在当前时间模态范围内，联合矩形覆盖约 95.82%；此诊断掩膜不等同于学习谱层的精确 Hermitian 输出，也不构成 FNO 误差下界。两侧 y 边界共 3.125% 单元承载 44.19% 误差能量；接收位于 y 索引 127，稀疏接收波形误差已有 64.13%，时间重建后 64.14%。

下一轮固定同一单样本、模型、种子、优化器与 400 次预算，仅增加预先固定权重的接收监督 MSE，检查误差权衡。这不是物理损失或梯度约束。单样本学习不证明多工件泛化、材料导数或反演可用。

## 接收监督固定预算对照

`run_receiver_loss_control.py --data .build/operator-pilot-v3-20261008 --control .build/basic-fno-single-sample-20261008 --out .build/basic-fno-receiver-loss-20261008` 保持相同单训练样本、种子、随机初始化、模型、优化器和 400 次更新预算，唯一增加接收权重 1。接收项在 500 帧波场的 24 个实际接收位置计算，以训练工件 trace RMS 归一化；每个残差平方后再取均值。完整 2000 步接收记录只用于评价。初始预测与旧对照一致，选择固定最终轮。

最终完整接收误差从 64.14% 降至 6.92%，但全波场误差从 18.05% 升至 38.62%；同一个训练样本无缺陷参照为 24.47%。接收相位与幅度改善，不证明材料导数或未见工件泛化。`audit_training_trace_spectrum.py`、`audit_training_boundary.py` 和 `summarize_receiver_loss_control.py --data ... --control ... --run ...` 保存相同评价链与对照图。

后续已执行原 12/6 个训练/验证工件，接收权重 0/1 双组各固定 20 epoch、960 次更新，报告最终轮的波场/接收/散射权衡。不读取旧测试集。本轮保留原对照和首轮结果，不开展代理反演。

## 多工件双组固定预算结果与材料响应

`train_multiworkpiece_control.py --data .build/operator-pilot-v3-20261008 --out .build/basic-fno-multi-control-20261008`：两组同初值哈希、同样本顺序、相同四层普通 FNO、seed=2、Adam lr=0.001，各固定 20 epoch、960 次更新，接收权重 0/1。最终轮作主要比较，无验证选模、无测试数据读取。权重 0 的前六轮训练 MSE 精确复现首轮结果。

最终训练接收误差 85.60%/55.32%，验证 85.77%/55.85%，验证无缺陷参照 24.85%。验证波场误差 75.27%/85.54%，无缺陷参照 28.35%。两组整体拟合不足，尚未收敛；不能只按接收相对改善宣布代理可用。

`audit_validation_material_response.py --data .build/operator-pilot-v3-20261008 --run .build/basic-fno-multi-control-20261008` 比较最终模型的 FNO(gamma)-FNO(1) 与 FD(gamma)-FD(1)，验证全部 24 次发射的差分相对误差 1.0007/1.0039、预测差分范数为参考的 2.96%/7.92%，余弦近零。这是有限幅度材料响应，不是局部梯度。追加审计协议在训练期间、最终结果产生前保存，不影响训练/选模。

`summarize_multiworkpiece_control.py --run .build/basic-fno-multi-control-20261008` 保存图像和最终汇总。下一步先用同一单训练样本固定 400 次、同模型/种子/损失权重/旧尺度，改学 FD(gamma)-FD(1) 的散射残差，再检查多工件验证；这个残差训练后续已执行，见下节。无缺陷基底保持与 gamma 无关，区别于 PDE 残差或第二网络学习第一网络误差。

## 散射目标：单样本与多工件

`run_scatter_target_pilot.py --data .build/operator-pilot-v3-20261008 --plan .build/basic-fno-multi-control-20261008/next_scatter_protocol.json --out .build/basic-fno-scatter-target-20261008` 保持旧单训练样本、模型/seed/Adam/接收权重1/旧尺度和400次预算。目标改为 FD(gamma)-FD(1)，完整接收用完整无缺陷波形加重建残差。最终散射波场/接收误差为58.11%/8.08%，完整场/接收14.67%/1.98%，优于旧单样本直接目标的38.62%/6.92%；这仍是已见样本。

`train_scatter_multiworkpiece.py --data .build/operator-pilot-v3-20261008 --control .build/basic-fno-multi-control-20261008 --out .build/basic-fno-scatter-multi-20261008` 在12/6工件上匹配旧直接目标权重1组的初始权重哈希、20轮实际样本顺序、尺度与优化器，固定960次更新。最终验证完整接收24.90%，无缺陷参照24.85%；散射波场/接收100.13%/100.17%，未有效学到散射。训练散射接收98.18%，也未充分拟合。

`audit_validation_material_response.py --data .build/operator-pilot-v3-20261008 --run .build/basic-fno-scatter-multi-20261008 --arms 1` 审计残差模型的材料差分（固定基底相减抵消），差分范数仅为参考0.86%、相对误差1.00009。新增 --arms 的默认仍为0/1，不改旧结果文件。`summarize_scatter_target_pilot.py --run ...` 和 `summarize_scatter_multiworkpiece.py --run ...` 保存对照图与汇总。

当时预先计划（后续已执行）：八通道普通FNO的零附加通道与固定0.5MHz sin/cos时间通道两组，同架构/初值/顺序/散射目标/尺度/20轮预算，只改输入特征。不可直接用旧六通道模型做因果比较；当前暂不接入FWI。


## 时间 Fourier 输入同预算对照

`train_time_feature_control.py` 在八通道普通 FNO 上配对零特征与固定 0.5MHz sin/cos 特征，初始权重哈希相同，原 20 轮顺序相同；两组各 960 次更新，散射目标、旧尺度与接收权重1保持不变。输出 `.build/basic-fno-time-feature-control-20261008`。最终验证散射接收 99.596%/99.562%，材料差分范数 0.025%/0.211%。

`audit_validation_material_response.py` 按检查点配置和特征模式加载；原六通道默认兼容。`audit_time_feature_use.py` 仅冻结实验模型后移除特征，不能代替两组训练对照。`summarize_time_feature_control.py` 保存图和汇总，`verify_time_feature_control.py` 重新加载两组并复算全部验证发射。复现命令及详细结论见知识库“普通FNO时间Fourier输入同预算对照 2026-10-08”。

下一步计划 `next_material_conditioning_protocol.json`：两个训练工件同一源，真实 gamma/全1 gamma 输入对照，各800更新，与两个训练真值的平均预测比较；不读取验证/测试。这只是可学性诊断，尚未运行，不作代理FWI或泛化结论。

`audit_scatter_target_spectrum.py` 对全部训练/验证散射目标重新检查描述性矩形FFT覆盖；时间能量约99.78%，联合范围约96.16%/95.87%。这不是严格FNO误差下界。补充结果保存在 `scatter_target_spectrum.json` 和知识库实验记录。


## 同源双工件材料区分

`run_material_conditioning_pair.py` 执行训练000/001、源0的真实gamma/全1gamma配对，各固定800更新、每件400暴露，八通道零特征、相同初值和交替顺序。真实组平均监督损失0.03832，低于材料无关共享均值最优0.06369；完整接收误差5.12%、散射接收20.40%，两材料接收差分误差28.25%、范数比89.12%、余弦0.96186。内部散射场66.79%、内部材料差分88.44%，暂不接入FWI。

`verify_material_conditioning_pair.py` 重载两组、检查800个Adam步数及均值二次型恒等式；`audit_pair_alternating_gradients.py` 仅审计监督参数梯度，不是物理梯度；`audit_material_pair_regions.py` 分开实际y网格区域；`summarize_material_conditioning_pair.py` 保存图和汇总。输出 `.build/basic-fno-material-pair-20261008`，完整复现命令见知识库新实验记录。

下一步 `next_four_materials_protocol.json`：四训练材料源0、每件400次暴露、共1600更新，冻结训练均值作为验证参照，评价六个验证材料源0；仍不读取旧测试，尚未运行。这不单独隔离工件数与总预算的因果影响。


## 四材料单源训练与验证

`run_four_materials.py` 在训练000..003的发射0上循环400轮、共1600次更新，从同一八通道零特征seed2初值重训；使用旧RMS与接收权重1。四训练目标的均值单独冻结为 `training_mean.pt`，只在原六验证材料的发射0上评价，未加载旧测试。

最终训练散射场/接收40.46%/7.63%，验证128.42%/125.29%；六件验证接收均差于零散射，冻结训练均值验证接收108.03%。`audit_four_material_response.py` 给出验证材料差分范数63.79%、余弦0.03344、相对误差116.80%，响应幅度已有但方向仍错。暂不开展代理FWI。

`verify_four_materials.py` 重载模型、逐值复算四训练/六验证及仅训练生成的均值、检查1600个Adam步数和能量恒等式；`summarize_four_materials.py` 保存图和全工件结果。输出 `.build/basic-fno-four-materials-20261008`，复现命令见知识库实验记录。下一步 `next_twelve_materials_protocol.json` 为12训练材料源0、每件400次暴露、4800次固定更新，开发验证不选模，尚未运行；不孤立归因于工件数。


## 十二材料单源覆盖扩展与基础DeepONet首轮

`run_twelve_materials.py --data .build/operator-pilot-v3-20261008 --plan .build/basic-fno-four-materials-20261008/next_twelve_materials_protocol.json --out .build/basic-fno-twelve-materials-20261008`：12训练源0循环400轮，总4800更新。普通FNO、旧归一化、接收权重1与seed2八通道初值均保持，逐步输入全部12件与原编码器逐值一致。训练散射场/接收26.61%/8.98%；六开发验证135.00%/131.91%，材料接收响应范数79.55%、余弦−0.04240。最终固定轮，未按验证改选检查点。

`audit_twelve_material_response.py`、`verify_twelve_materials.py`、`summarize_twelve_materials.py`均使用`--data ... --run .build/basic-fno-twelve-materials-20261008`。4800步、每件400暴露、冻结12训练均值、同初值、完整重载指标0差异及能量恒等式已通过。`fixed_source_inputs.py`仅减少驻留输入，数值不变。旧测试未读取，不接入代理FWI。

`audit_material_coverage.py --data .build/operator-pilot-v3-20261008 --out .build/material-coverage-20261008`固定gamma欧氏最近邻，复制训练源0波形，无验证标签参与邻居选择。验证散射接收120.11%；稀疏覆盖的描述性诊断，不证明数据不足是唯一原因。`plot_material_coverage.py`保存材料图。

`basic_deeponet.py`：固定256×128材料CNN branch与五坐标GELU MLP trunk，64维内积，带符号散射输出。实参数4,251,521，与FNO按复参数两个实数计4,194,469相近；不代表表达能力相同。`verify_basic_deeponet_interface.py`在训练000及小网格上完成编码、计算图AD与分块查询核查，不是物理梯度验证。

`run_deeponet_single.py --data .build/operator-pilot-v3-20261008 --out .build/basic-deeponet-single-20261008`固定训练000/source0、400更新、seed2、场坐标32768点/步（有放回均匀抽样，CPUseed19）、全部12000接收点/步、旧尺度/基底及Adam0.001。最终散射场100.00289%、接收100.00023%，近零预测，未拟合。抽样场损失与旧FNO全网格逐步损失不同，不作架构因果排名。

`verify_deeponet_single.py --data ... --run .build/basic-deeponet-single-20261008`：完整网格预测重载逐值一致、抽样序列SHA复算、400个Adam步数和有限状态、作者文件与数据哈希通过。`audit_deeponet_training_state.py`固定指向该目录，冻结初始/最终诊断：仍有branch/trunk梯度，不能单据此认定频谱偏置。`summarize_deeponet_single.py --data ... --run ...`保存图像，已视觉核查。

当时下一步`next_time_feature_protocol.json`保存在DeepONet结果目录，后续已执行，见下节：同七通道trunk、两组匹配初值/抽样/400更新，仅激活固定0.5MHz sin/cos时间特征。先解决单材料时域拟合，不扩大数据或接入FWI。科研笔记与三篇补充一手来源已同步Obsidian。


## DeepONet时间特征与固定预算延长

`deeponet_time_features.py`新增匹配七输入trunk，原五坐标逐值相同，附加零或实际秒数双精度计算的0.5MHz sin/cos；不改变旧五通道模型。`run_deeponet_time_control.py --data .build/operator-pilot-v3-20261008 --out .build/basic-deeponet-time-control-20261008/zero --mode zero --plan .build/basic-deeponet-single-20261008/next_time_feature_protocol.json`，sincos组改mode/out。两组400次，相同seed2初值、CPUseed19抽样流、32768场点与12000接收点、原尺度及Adam。400次接收99.9994%/99.8175%，场100.0005%/100.0452%，未拟合。

预先写入`next_budget4000_protocol.json`后，`extend_deeponet_time_control.py --data ... --out .build/basic-deeponet-budget-control-20261008/zero --mode zero --plan .build/basic-deeponet-time-control-20261008/next_budget4000_protocol.json`恢复相应400模型/Adam，跳过抽样流前400批再训练3600次，sincos同理。最终接收99.9985%/98.0712%，场100.0008%/100.4077%；微弱接收改善不构成有效散射拟合，尚不能唯一归因于频谱偏置。

`verify_deeponet_time_control.py`检查400结果，`verify_deeponet_budget_control.py`检查4000结果，均使用`--data ... --run 相应组目录`。两预算每组完整网格重载预测逐值一致、指标0差异；同初值、全部点序列SHA、Adam步数、有限状态、原作者七文件通过。恢复400检查点与第400次指标前缀通过。`audit_deeponet_time_state.py --run ...`仅冻结优化审计，梯度不是物理梯度；`summarize_deeponet_time_control.py`固定读取两预算目录并保存汇总图。图像已视觉核查并入库。

当时下一步（后续已执行，见下节）`.build/basic-deeponet-budget-control-20261008/next_branch_freeze_protocol.json`，历史planned_not_run保留：同sincos七通道模型、seed2/19、4000预算，一组联合训练，一组仅冻结整个初始branch、训练trunk与bias。检查单材料联合优化条件，不读取验证/test，不把冻结branch当作已学到材料算子，也不接入FWI。


## DeepONet分支冻结与训练时间基诊断

`run_deeponet_branch_control.py --data .build/operator-pilot-v3-20261008 --out .build/basic-deeponet-branch-control-20261008/joint --arm joint --plan .build/basic-deeponet-budget-control-20261008/next_branch_freeze_protocol.json`执行七通道sincos从同初值重新训练4000次；冻结组将arm及out子目录改为`frozen_initial_branch`。两组seed2/19、抽样点、旧尺度、损失与预算匹配；冻结组仅trunk和bias进入Adam，branch全部保持初值。最终散射场100.40794%／100.00832%，接收98.07092%／96.32091%。小幅接收改善仍不构成有效拟合。

`verify_deeponet_branch_control.py --data ... --run 相应组目录`检查完整网格重载、全部4000步抽样SHA、Adam步数、作者文件及冻结branch逐值不变与优化器参数选择，两组通过。`audit_deeponet_branch_state.py --run ...`审计监督优化状态，不能替代物理梯度；冻结branch的权重梯度为零是设定结果，材料输入梯度仍可穿过冻结函数。

`audit_receiver_temporal_basis.py`仅用训练000/source0，对每个接收器使用整条训练真值独立求最小二乘系数：单载波4列／16局部载波窗口34列／时间Fourier常数加63对sin-cos共127列，500点拟合后重建2000点误差98.2906%／63.5899%／3.09159%。最小二乘正规方程及Fourier与截断FFT数值一致性通过。它是训练内线性时间基解析参照，不是预测、材料算子或非线性DeepONet能力下界；稀疏真值重建误差仅0.05210%。结果`.build/receiver-temporal-basis-20261008`。

`summarize_deeponet_branch_control.py`保存对照汇总与图像，已实际视觉核查并同步Obsidian。下一协议`.build/basic-deeponet-branch-control-20261008/next_multifrequency_protocol.json`仍planned_not_run：131输入匹配trunk，两组均冻结相同初始branch，126零特征／k=1..63时间sin-cos特征，4000次固定预算与同点序列。新初值配对，旧七通道不是因果对照。本轮不读取验证或旧测试，不接入代理FWI。详见知识库“基础DeepONet分支冻结与时间基诊断 2026-10-08”。
## 131输入多频DeepONet对照

`run_deeponet_multifrequency.py`按`.build/basic-deeponet-branch-control-20261008/next_multifrequency_protocol.json`完成两组训练000/source0、4000更新的冻结branch对照。两组使用新131输入trunk；`zero126`为126个零特征，`fourier126`为k=1..63的时间sin/cos。初值、采样流、旧尺度、损失和优化器匹配，未读取验证或测试。

最终散射场／接收相对L2：zero126为100.0021%／100.0004%，fourier126为96.1527%／52.5875%。Fourier只改善接收时间曲线，全场仍未有效拟合；暂不扩大DeepONet结构或进入多材料训练。结果在`.build/basic-deeponet-multifrequency-20261008`。

`verify_deeponet_multifrequency.py`检查两组4000步、采样序列、有限Adam状态、冻结branch逐值不变、完整网格重载逐值一致；两组通过，重载指标差异0。下一步转传统FWI近真值／均匀初值诊断和FNO固定形状单轴材料族插值。

## 传统 FWI 近真值与材料方向诊断

`run_discrete_fwi.py`新增`--init uniform|truth|near_truth`，只影响新输出目录，不修改作者文件。案例1真值初始化和接近真值初始化各5次、lr0.02的结果见`.build/author-discrete-fwi-case1-truthinit-20261008`与`neartruth`；失配分别降至初始82.90%和80.29%，材料误差却从真值附近快速增大。

`audit_fwi_material_directions.py`在真值场对均匀、x/y平滑和中心局部四类方向做±0.005/±0.02试探。真值附近失配变化约10^-5至10^-4，报告保存在`.build/author-fwi-material-directions-20261008/report.json`。这是局部方向响应证据，不是完整可辨识性或定量FWI结论；下一步扩展浅层、深层和缺陷邻域方向及Gram矩阵。

后续解释修正：上项投影正负步不对称，不能作为中心方向导数；绝对代价小也不说明残差接近零。`audit_fwi_response_gram.py`双精度正密度相对扰动六方向复核，在`.build/author-fwi-response-gram-v2-20261008`保存响应Gram及观测一致性。真值正演与发布观测相对L2为51.55%，±8点时移与整体幅值校正后仍至少48.79%。六个单位响应方向的奇异值0.7282..1.2103，不支持此前“普遍平坦”解释。

`verify_fwi_response_gram.py`重算Gram和输入／作者哈希，并确认作者原真值正演与离散参考逐值一致；自洽观测下真值梯度为零、一步材料变化仅2.53e−13，发布观测下一步失配下降但材料偏离。`plot_fwi_response_gram.py`保存图像，已查看并入库。最早绝对方向试运行出现负密度和非有限结果，仅协议保留，不能作为成功报告。下一步追查生成设置和自洽反演。

## 作者生成设置与自洽反演四组对照

新取作者v3的`GenerateMeasurementDataTrain.py`／`GenerateMeasurementDataTest.py`，MD5与发布清单一致，单独保存于`.build/author-observation-control-20261008/downloaded_generation_sources`。两个生成脚本用空间细化2倍、dt减半、时间步数翻倍生成观测，再抽样至反演网格；源接收物理坐标匹配。它们不是四案例专用生成脚本，案例包没有细网格几何，因此尚不能精确复现measurement1或解释全部51.55%残差。

`run_observation_consistency_control.py`预先写入协议，执行案例1的发布／自洽观测×均匀／.95truth+.05初始，各固定35次更新，Adam lr.02与旧掩膜／尺度／裁剪／上下界相同。发布均匀／近真值最终波形47.7750%／42.1485%，材料误差倍率2.07964／253.568；自洽均匀／近真值18.4787%／3.0754%，材料倍率2.02959／7.04670。改善观测一致性不足以解决当前自由像素恢复。

`verify_observation_consistency_control.py`通过四组端点预测逐值重载、指标差异0、35步有限Adam状态、自洽观测重算及作者／输入／下载哈希；`plot_observation_consistency_control.py`绘制四条轨迹与共享色标材料图，已实际查看并入库。下一协议`next_reduced_material_protocol.json`仍planned_not_run：利用已知缺陷几何，只评价9个预定密度参数；这是降维机制诊断，不能声称盲反演。
