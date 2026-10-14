"""Append independent supplemental timings; classify errors from saved logs."""
from common import *
ROUND=ROOT/'.build/fwi-quality-guards-round2-20261013'
OUT=ROUND/'t4'
status=json.loads((OUT/'status.json').read_text())
backlog=(OUT/'scale1_fno_backward.log').read_text(encoding='utf-8')
assert 'out of memory' in backlog and 'y=model(x)' in backlog
dump(OUT/'failure_classification.json',dict(scale1_fno_forward='torch.OutOfMemoryError during no-grad forward',scale1_fno_backward='torch.AcceleratorError: CUDA out of memory in grad-enabled forward before backward; traceback line y=model(x), GELU',scale2_fno='not attempted after two failures',modification='none; unchanged R2 structure'))
path=ROOT/'PINO超声成像知识库/05实践与复习/T4 物理与R2尺度计时 2026-10-13.md'
lines=['\n## 第二轮获准补测：无历史纯正演与四源口径\n',
'上一轮512档两次OOM分别在物理完整历史正演、物理正演+伴随；不是FNO的OOM，当时512档FNO未运行。本次补测不存波场历史，不含伴随，不引入检查点。',
'GPU NVIDIA RTX5070(12GB)，FP32/complex64；预热3次，计时10次取中位数，CUDA同步。物理为4源同时batch；FNO为顺序4次单源，合计4源工作量，未改网络。FNO固定500时间帧、modes=(16,8,64)、width=8、4层、8输入通道、8389129参数，随机权重，仅计时，不训练、不用于梯度门。',
'| 网格材料点数 | Nt×dt | 物理单步中位数 | 物理无历史总时长中位数 | FNO四源前向 | FNO四源前向+输入反传 |','|---|---|---:|---:|---:|---:|']
f0=json.loads((OUT/'scale0_fno_forward.json').read_text());b0=json.loads((OUT/'scale0_fno_backward.json').read_text())
lines.append(f"| 256×128 | 2000×30ns | 未测 | 未测 | {f0['median_seconds']:.6f}s | {b0['median_seconds']:.6f}s |")
for scale,n,nt,dt in [(1,'512×256',4000,15),(2,'1024×512',8000,7.5)]:
 r=json.loads((OUT/f'scale{scale}_physical_streaming.json').read_text())
 lines.append(f"| {n} | {nt}×{dt}ns | {r['single_step_median_seconds']*1000:.4f}ms | {r['median_seconds']:.6f}s | {'OOM' if scale==1 else '未尝试'} | {'OOM（带梯度前向阶段）' if scale==1 else '未尝试'} |")
lines+=['512档FNO无梯度前向OOM；带梯度前向+反向工作负载也OOM，但错误在前向GELU、尚未执行反向。第二次以torch.AcceleratorError而非torch.OutOfMemoryError抛出，保存原日志并按错误文字明确归类为显存不足。两次失败后停止T4，1024档FNO不再尝试，不能记成实测OOM。',
'证据等级：**实验验证（仅这些计时及OOM现象）**。白话：不用保存每个时刻的整幅声场，物理正演能在大网格上跑；这不表示需要历史波场的反演也能跑。',
'不报告加速比：虽然源数已统一4源，物理纯正演只保留最终状态，而FNO输出500帧全场；此前物理完整历史输出Nt帧。这些输出和存储负担不同，且R2质量未过关，不能据此声称FWI加速。所有结论仅限二维标量声学，外推弹性波/三维为“有理论依据未验证”。',
f"T4补测独立墙钟{status['elapsed_seconds']/60:.2f}分钟，不计入第二轮7小时反演预算。原始输出`.build/fwi-quality-guards-round2-20261013/t4/`。已停止。"]
assert '## 第二轮获准补测' not in path.read_text(encoding='utf-8')
with path.open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
with (ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md').open('a',encoding='utf-8') as f:f.write('\nT4补测：四源无历史纯正演512/1024档成功；512档FNO前向与带梯度前向均OOM，累计两次后停止。详见[[T4 物理与R2尺度计时 2026-10-13]]。不报告加速比。\n')
print('T4 documented',status['elapsed_seconds'],flush=True)
