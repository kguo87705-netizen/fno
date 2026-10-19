# Deepwave 批量正演备选：安装前方案

核查日期：2026-10-19。**状态：未安装、未运行，等待用户确认。** 本报告只使用官方元数据、文档和发布源码核查；没有下载 Python/PyTorch/Deepwave 安装包。

## 拟锁定的环境

| 项目 | 版本/来源 | E 盘位置 | 下载与空间估计 |
|---|---|---|---|
| CPython | 3.12.15，Astral python-build-standalone 20261013，Windows x86_64 MSVC 完整 install_only 包 | `E:/FNO_research/envs/cpython-3.12.15/` | 发布资产46,390,009字节，约44.24MiB；解压暂估0.15–0.3GiB |
| 独立 venv | 由上述 E 盘解释器创建 | `E:/FNO_research/envs/deepwave/` | 所有 site-packages 均在此处 |
| PyTorch | **2.11.0+cu128**，官方 Windows cp312 CUDA 12.8 wheel | venv 内 | HEAD确认wheel为2,753,189,216字节，约2.56GiB；解压及依赖暂估5–7GiB |
| Deepwave | **0.0.27**，PyPI Windows x86_64 wheel | venv 内 | PyPI记录22,980,929字节，约21.92MiB；解压占用待安装后统计 |
| 数据与绘图库 | numpy、h5py、matplotlib，及所需依赖；解析后记录实际锁定版本 | venv 内 | 暂预留0.3–0.8GiB |

整体按环境约6–8GiB、下载缓存约3GiB及解压临时空间估计，**预留15GiB**。这是安装前预算，不是实测占用。不安装 torchvision/torchaudio，不安装系统 CUDA Toolkit、编译器或驱动。

Python 使用可解压的 standalone 构建是为了避免 Windows 安装器向系统盘写安装缓存/修改全局设置。该构建来自 Astral，不冒称 Python.org 官方 Windows 安装程序。两个 Python/venv 目录都位于用户指定的 envs/ 下。

## 所有写入重定向

安装/运行启动器将只为本项目进程设置：

- `PIP_CACHE_DIR=E:/FNO_research/cache/pip`
- `CONDA_PKGS_DIRS=E:/FNO_research/cache/conda`（本方案不使用 conda）
- `TORCH_HOME=E:/FNO_research/cache/torch`
- `TORCH_EXTENSIONS_DIR=E:/FNO_research/cache/torch_extensions`
- `TORCHINDUCTOR_CACHE_DIR=E:/FNO_research/cache/torchinductor`
- `MPLCONFIGDIR=E:/FNO_research/cache/matplotlib`
- `CUDA_CACHE_PATH=E:/FNO_research/cache/cuda`
- `TEMP`、`TMP` 指向 `E:/FNO_research/tmp`
- `APPDATA`、`LOCALAPPDATA`、`XDG_CACHE_HOME` 指向 E 盘项目缓存；禁用 `.pyc` 写入。

pip 只通过 E 盘虚拟环境的 `python -m pip` 调用，使用预编译 wheel（`--only-binary=:all:`），避免未获确认就隐式构建/下载编译环境。环境变量不写入系统全局。

## RTX 5070 兼容性证据与限制

- PyTorch 从2.7/CUDA12.8开始引入Blackwell支持；官方旧版本页提供2.11.0 CUDA12.8 Windows包，已核对cp312 win_amd64 wheel存在。当前驱动610.88已能运行MATLAB CUDA前向兼容测试。
- Deepwave 0.0.27 有Windows预编译wheel且要求Python>=3.8。发布标签v0.0.27的CI使用CUDA12.4，CMake设置`CUDA_ARCHITECTURES all`；不能据此声称该wheel含原生sm_120机器码。它在Blackwell上是否能通过PTX前向编译运行，必须实际小规模验证。
- **PyTorch支持RTX5070，不等于Deepwave扩展已经验证支持。** 安装后先检查torch版本、CUDA版本、设备能力、arch list及GPU运算，再执行极小的`deepwave.scalar` GPU试算；失败就记录并停止，不自行安装CUDA或源码编译。

## 获确认后的比较协议

1. 读取已经保存的k-Wave参考模型、阵元位置、Ricker样本和时间轴，不重新随机生成模型。保持0.1mm、400×400、dt=5ns、4000点、PML每边20点、64个单点源和64接收点。
2. `deepwave.scalar(..., accuracy=8)`，float32、`torch.no_grad()`，模型和波源不要求梯度；一次调用全64炮，输入源形状为`[64,1,4000]`。不保存用于反传的全时刻波场。若OOM，停止报告，不悄悄改成小批次后宣称完成64炮批处理。
3. 在GPU同步前后计时；分开报告首次调用/JIT成本和稳定调用耗时。检查实际内部CFL子步长，不能只看传入dt。输出同样每4点抽取一次。
4. 同时测PyTorch峰值allocated/reserved显存和整卡nvidia-smi采样峰值，明确口径。
5. 每道相对L2使用k-Wave道的L2范数作分母，并报告低能量道；零时移归一化互相关和有限延迟范围内的峰值互相关分开报告。预定画Tx/Rx=(1,1),(1,64),(32,1),(32,32),(32,64),(64,1)波形叠加。
6. 首先报告**未经幅度拟合/时间对齐**的差异。若补充幅度或时移标定，必须保存标定参数、来源和原始指标；不能逐道自由拟合后把误差称为两个求解器的原始一致性。

## 不能忽略的波源和数值差异

k-Wave当前`source.p_mode='additive'`把给定压力序列经过dt、dx、局部声速缩放后作为一阶方程的质量源，并应用k空间源修正。Deepwave scalar则把源作为二阶波动方程右端的强迫项；官方文档还明确源的第t个样本影响下一时间步，而接收记录当前步波场。因此，相同Ricker数组不自动代表相同的物理源归一化和离散时间零点，可能不仅是一个常数幅度差，也可能涉及相位/波形差异。

先遵守用户要求进行“同Ricker数组”的原始对比，完整报告差异来源；若需要源等效变换或均匀介质标定，把它作为独立、明确记录的补充实验，不覆盖原始结果。

此外，k-Wave伪谱与8阶有限差分的数值色散、PML实现均不同，尤其低声速区高频分量只有约2.5点/波长。不能用未收敛网格上的波形差异直接认定某一求解器错误。

## 尚不能给出的数字

Deepwave当前没有实测64炮时间、显存或跨求解器波形指标。令未来实测完整样本时间为T秒，则300/500样本分别为T/12和5T/36小时（不含启动、I/O余量）；在安装获批并实测前不填入猜测值。

## 来源

- [PyTorch 2.7 Blackwell发布说明](https://pytorch.org/blog/pytorch-2-7/)
- [PyTorch官方历史版本安装命令](https://pytorch.org/get-started/previous-versions/)
- [官方CUDA12.8 wheel索引](https://download.pytorch.org/whl/cu128/torch/)
- [Deepwave 0.0.27 PyPI](https://pypi.org/project/deepwave/0.0.27/)
- [Deepwave scalar方程与记录时序](https://ausargeo.com/deepwave/scalar)
- [Deepwave v0.0.27构建工作流](https://github.com/ar4/deepwave/blob/v0.0.27/.github/workflows/build.yml)
- [Deepwave v0.0.27 CMake配置](https://github.com/ar4/deepwave/blob/v0.0.27/CMakeLists.txt)
- [Astral standalone Python 20261013发行](https://github.com/astral-sh/python-build-standalone/releases/tag/20261013)
- k-Wave 1.4.1 本地源码`private/kspaceFirstOrder_scaleSourceTerms.m`及`kspaceFirstOrder2D.m`，与参考样本使用的工具箱一致。
