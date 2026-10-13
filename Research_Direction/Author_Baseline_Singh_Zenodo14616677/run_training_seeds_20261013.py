"""Four additional paired initialization seeds; all other settings unchanged."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MPLBACKEND']='Agg'
import sys,time,json,random,threading
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
SUP=BASE/'singh_zenodo';AUTHOR=SUP/'singh_zenodo'
DATA=ROOT/'.build/singh-coverage-paired-20261012'
BUILD=ROOT/'.build/singh-training-seeds-20261013';BUILD.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(SUP),str(AUTHOR)];torch.set_num_threads(4)
TOTALSTART=time.time();DEADLINE=TOTALSTART+3450
def timeout():
    print('STOP 60-minute budget guard',flush=True);os._exit(124)
timer=threading.Timer(3450,timeout);timer.daemon=True;timer.start()
record=ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md'
with record.open('a',encoding='utf-8') as f:
    f.write('\n\n## 2026-10-13 训练种子复现（正式开跑）\n裁剪与调度：论文和发布代码不一致，统一按代码执行（已知差异，已豁免）。具体：预训练梯度裁剪5e-4（论文5e-5）；预训练beta置零，RMSprop学习率恒定5e-4（论文beta=0.2）；迁移学习梯度裁剪5e-5（论文1e-5）。\n长期规则更新：论文与发布代码不一致默认按代码执行，开跑前在统一记录注明，不停跑；仅代码与作者Readme/发布权重矛盾、组间设置不一致或按代码物理不成立时停止询问。论文预训练学习率8e-4与本轮指定代码5e-4的差异已注明，本轮两组统一5e-4。100轮符合Readme；时间网格豁免继续有效。此规则替代此前“所有不一致都停跑”的流程。\n预注册不变：5种子实验组平均1%虚警工作点检出数减对照≥2，且至少4/5种子实验组更多，才记对训练种子稳健，否则不稳健；无论判定均更新汇总表，不启动FNO。\n种子映射：seed0为已有Torch初始化30；新增seed1/2/3/4为31/32/33/34，两组配对同种子；数据顺序Python种子2、NumPy种子2、划分Torch种子2固定；迁移脚本种子99固定。只变预训练网络初始化及随后训练的Torch随机状态，不改变数据内容、顺序和划分。每模型100轮、RMSprop lr=5e-4、batchSize=20、迁移5步，用最终第100轮权重，不根据测试成绩挑轮次。\n')
old=(BASE/'run_coverage_100epochs_20261013.py').read_text(encoding='utf-8')
traincode=old[old.index('def checkpoint(ns):'):old.index("for group in ['control','experiment']:")]
traincode=traincode.replace('code=code.replace(\'epochs = 1\',\'epochs = 100\')',"code=code.replace('epochs = 1','epochs = 100').replace('torch.manual_seed(30)',f'torch.manual_seed({INITSEED})')")
assert 'INITSEED' in traincode
exec(compile(traincode,'same_100_epoch_training','exec'),globals())
previous=(BASE/'run_coverage_paired_20261012.py').read_text(encoding='utf-8')
transfer=previous[previous.index('# Reuse author TransferLearningFWI'):].replace(';timer.cancel()','')
all_times={}
for seed in range(1,5):
    INITSEED=30+seed;OUT=BUILD/f'seed{seed}';OUT.mkdir(exist_ok=True)
    START=time.time();timings={}
    print('SEED_START',seed,'INIT',INITSEED,flush=True)
    for group in ['control','experiment']:
        timings[group]={'training':train(group)}
        (OUT/'timings.json').write_text(json.dumps(timings,indent=2),encoding='utf-8')
    exec(compile(transfer,'unchanged_paired_transfer','exec'),globals())
    all_times[str(seed)]=timings
    (BUILD/'run_times.json').write_text(json.dumps(dict(seeds=all_times,wall_seconds=time.time()-TOTALSTART),indent=2),encoding='utf-8')
    print('SEED_FINISHED',seed,'elapsed_total',time.time()-TOTALSTART,flush=True)
print('ALL_SEEDS_FINISHED',time.time()-TOTALSTART,flush=True);timer.cancel()
