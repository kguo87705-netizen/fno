"""Authorized correction: reuse data, change only pretraining epochs 1 -> 100."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MPLBACKEND']='Agg'
import sys,time,json,random,threading
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
SUP=BASE/'singh_zenodo';AUTHOR=SUP/'singh_zenodo'
DATA=ROOT/'.build/singh-coverage-paired-20261012'
OUT=ROOT/'.build/singh-coverage-100epochs-20261013';OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(SUP),str(AUTHOR)]
torch.set_num_threads(4)
START=time.time();DEADLINE=START+2300
def timeout():
    print('STOP 40-minute budget guard',flush=True);os._exit(124)
timer=threading.Timer(2300,timeout);timer.daemon=True;timer.start()
record=ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md'
text=record.read_text(encoding='utf-8')
marker='### 同规模覆盖对照：本轮结果'
assert marker in text
a=text.index(marker)
tail=text[a:]
tail=tail.replace(marker,'### 同规模覆盖对照：1轮结果【无效：训练轮数与作者不一致；数字保留，结论作废】',1)
tail=tail.replace('结论：未支持；','作废结论（不得引用）：未支持；',1)
tail=tail.replace('明确判断：1%档','作废判断（不得引用）：1%档',1)
text=text[:a]+tail
text+='\n\n## 2026-10-13 重做：同数据200样本、100轮预训练\n上一轮状态：无效：训练轮数与作者不一致；所有数字保留，仅撤销关于覆盖作用的结论。Readme明确作者800样本权重训练100轮。\n本轮只将epochs=1改为100，其余沿用上轮：RMSprop lr=5e-4、batchSize=20、初始化Torch种子30，样本抽样Python/NumPy/Torch种子2、划分Torch种子2、80/20划分；调度公式和原脚本训练前beta=0的实际行为保留。复用两组已生成200样本，不生成新数据，从相同初始权重重新训练，不续接1轮模型。\n主判据不变：1%合并虚警下实验组比对照组至少多检出2个缺陷才记支持；0.5%及MSE为补充。100轮结束只用最终权重，不以验证或测试成绩选轮次。\n收敛检查（开跑前声明，仅描述，不改变训练）：验证损失有限且明显较初期降低，最后10轮均值相对前10轮变化绝对值≤5%、最后10轮变异系数≤10%时记“末段近似平台”；未满足就写未确认收敛，不擅自延长训练或把100轮等同收敛。\n以后规则：脚本与论文或Readme设置不一致，开跑前先告知负责人。\n'
record.write_text(text,encoding='utf-8')
oldpath=DATA/'comparison_scores.json'
old=json.loads(oldpath.read_text(encoding='utf-8'))
old['decision']['voided_previous_status']=old['decision']['status']
old['decision']['status']='无效：训练轮数与作者不一致'
old['decision']['voided']=True
oldpath.write_text(json.dumps(old,indent=2),encoding='utf-8')
timings={}
def checkpoint(ns):
    e=int(ns['epoch'])
    if e%10==0 or e==99:
        print('TRAIN_TRACE '+json.dumps(dict(group=ns['group'],epoch=e+1,validation=float(ns['validationCostHistory'][e]),training=float(ns['trainingCostHistory'][e]))),flush=True)
    n=e+1
    np.savez(OUT/ns['group']/'loss_history.npz',training=ns['trainingCostHistory'][:n],validation=ns['validationCostHistory'][:n])
def train(group):
    random.seed(2);np.random.seed(2);torch.manual_seed(2)
    folder=OUT/group;folder.mkdir(exist_ok=True)
    for stem in ['gradient','material','measurement']:
        assert len(list((DATA/group/'data').glob(stem+'*.h5')))==200
    code=(SUP/'Pretraining.py').read_text(encoding='utf-8-sig')
    assert code.count('epochs = 1')==1
    code=code.replace('epochs = 1','epochs = 100')
    code=code.replace('destinationFolder = "data"','destinationFolder = '+repr(str(DATA/group/'data')))
    code=code.replace('numberOfSamples = 800','numberOfSamples = 200').replace('random.sample(range(800), numberOfSamples)','random.sample(range(numberOfSamples), numberOfSamples)')
    code=code[:code.index('# Save the pretrained model')]
    code=code.replace('    if epoch % 10 == 0:','    checkpoint(globals())\n    if epoch % 10 == 0:')
    t=time.perf_counter();ns=dict(__file__=str(AUTHOR/'Pretraining.py'),__name__='__main__',checkpoint=checkpoint,group=group)
    exec(compile(code,'author_pretraining_100','exec'),ns)
    torch.cuda.synchronize();elapsed=time.perf_counter()-t
    assert ns['epochs']==100 and ns['lr']==5e-4 and ns['batchSize']==20
    assert len(ns['validationCostHistory'])==100
    v=ns['validationCostHistory'];assert np.isfinite(v).all()
    tail=float(v[-10:].mean());prev=float(v[-20:-10].mean())
    change=(tail-prev)/prev;cv=float(v[-10:].std()/tail)
    convergence=dict(last10_mean=tail,previous10_mean=prev,relative_change=change,last10_cv=cv,
                     late_plateau=bool(abs(change)<=.05 and cv<=.10 and tail<v[:10].mean()))
    torch.save(ns['model'].state_dict(),folder/'model_Unet_200')
    stats=dict(seconds=elapsed,epochs=100,training_loss=float(ns['trainingCostHistory'][-1]),validation_loss=float(v[-1]),convergence=convergence)
    print('TRAIN_FINISHED '+json.dumps(dict(group=group,**stats)),flush=True)
    del ns;torch.cuda.empty_cache();return stats
for group in ['control','experiment']:
    timings[group]={'training':train(group)}
    (OUT/'timings.json').write_text(json.dumps(timings,indent=2),encoding='utf-8')
# Same transfer wrapper as the preceding paired run; no data-generation code is executed.
previous=(BASE/'run_coverage_paired_20261012.py').read_text(encoding='utf-8')
transfer=previous[previous.index('# Reuse author TransferLearningFWI'):]
exec(compile(transfer,'same_five_step_transfer','exec'),globals())
