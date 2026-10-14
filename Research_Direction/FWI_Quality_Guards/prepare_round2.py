"""Freeze outcome-blind stratified subset and the owner's revised endpoint."""
from common import *
from scipy.optimize import milp, LinearConstraint, Bounds
from collections import Counter

ROUND=ROOT/'.build/fwi-quality-guards-round2-20261013'
NOTE=ROOT/'PINO超声成像知识库/05实践与复习/T2 独立测试探索轮 2026-10-13.md'
assert not (ROUND/'selection.json').exists(), 'Never redraw a frozen test subset'
rows=json.loads((BUILD/'accepted_geometries.json').read_text())
boards=[r for r in rows if r['split']=='test' and not r['blank']]
assert len(boards)==60
counts=np.array([[sum(p['category']==k for p in r['components']) for r in boards] for k in range(8)])
rng=np.random.default_rng(20261014)
# Exact type stratification: 16 boards, eight defects of each of eight types.
# Random costs break ties without reading observations or reconstruction results.
solution=milp(rng.uniform(size=60), integrality=np.ones(60),bounds=Bounds(0,1),
 constraints=LinearConstraint(np.vstack([np.ones(60),counts]),[16]+[8]*8,[16]+[8]*8),
 options={'time_limit':60,'mip_rel_gap':0})
assert solution.success, solution.message
chosen=[r['id'] for r,x in zip(boards,solution.x) if x>.5]
assert len(chosen)==16 and np.all(counts[:,solution.x>.5].sum(axis=1)==8)
blanks=sorted(r['id'] for r in rows if r['split']=='test' and r['blank'])
cases=chosen+[blanks[0]]
bank=json.loads((BUILD/'frozen_thresholds.json').read_text())
thresholds={g:{cap:bank['results'][key][cap]['threshold'] for cap in ['0.01','0.005']} for g,key in [('A','0'),('B','B'),('C','100')]}
stamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
selection=dict(frozen_at_utc=stamp,seed=20261014,rule='binary selection of 16/60 with exactly eight defects/type; seeded random linear costs; no outcome inputs',cases=cases,defective_cases=chosen,blank=blanks[0],counts_by_category={str(k):8 for k in range(8)},total_defects=64)
dump(ROUND/'selection.json',selection)
cfg=dict(base_commit='137095a',frozen_at_utc=stamp,status='preregistered before any test inversion or outcome',
 decision_timing='Primary endpoint chosen by owner after validation descriptions, before any test outcomes',
 primary='Each defect detected iff within-defect abnormal pixel fraction >=0.5; pooled detection rate for holes 2/3/4/6 mm and for slits 1/2/3 cells separately',
 secondary='Mean within-defect abnormal pixel fraction for every type; descriptive only; report discordant direction',
 main_rule='epoch100 versus native epoch0 paired by training seed and geometry: pooled hole OR slit detection rate lower by >=0.10 on average and same direction in >=4/5 seeds',
 floor='If epoch0 pooled slit detection rate <0.10, slit arm cannot test; use holes only',
 evidence='Exploratory only: eight defects/type; no formal H2 decision. Formal requires full test with >=30/type.',
 interpretation='A/B/C tests initialization effects; prior-strength trend only on the epoch axis. B/C are different distributions, not ordered strength.',
 groups=dict(A='native expanded epoch0',B='existing ellipse200 epoch100 weights',C='existing expanded200 epoch100 weights'),
 seeds=[30,31,32,33,34],thresholds=thresholds,threshold_selection_seed=30,threshold_policy='Frozen validation bank; never retune per test, seed, or intermediate step; report actual FPR even above target',
 settings=dict(structure='author Unet [1,16,32,64,128], kernel2, BN train with batch1',input='same normalized initial gradient from fine observations and coarse solver',updates=35,optimizer='Adam',lr=5e-4,weight_decay=0,betas=[.9,.999],eps=1e-8,scheduler='(0.2*epoch+1)^(-0.5)',clip=5e-5,costScaling=1e10,gamma_min=GMIN,snapshots=[0,5,10,20,35],Nx=255,Ny=127,Nt=2000,dt=30e-9,source_batch=4,precision='float32',deterministic_algorithms=True),
 runtime_limit_seconds=7*3600,expected_seconds=[5.3*3600,5.9*3600],failures_stop=2,T3=False,gradient_gate=False,
 limitations=['One blank material/acquisition reference; all existing blank files identical, not independent.', 'No noise; fine and coarse share solver family.', 'Pretraining inputs/materials on coarse grid; historical training observations use 510x254 refinement, new observations511x255; distribution difference.'])
dump(ROUND/'config.json',cfg)
# New standalone copy: only approved test guard and output directory differ.
source=(HERE/'invert.py').read_text()
old="    if case.startswith('test_'):raise RuntimeError('Test inversion forbidden in first round')"
new="    approved=json.loads((ROOT/'.build/fwi-quality-guards-round2-20261013/selection.json').read_text())['cases']\n    assert case in approved, 'Case outside approved frozen subset'"
assert source.count(old)==1
source=source.replace(old,new)
old="out=BUILD/destination/f'seed{seed}_epoch{epoch}'/case"
assert source.count(old)==1
source=source.replace(old,"out=ROOT/'.build/fwi-quality-guards-round2-20261013'/destination/f'seed{seed}_epoch{epoch}'/case")
target=HERE/'invert_round2.py';assert not target.exists();target.write_text(source,encoding='utf-8')
text=f'''---
tags: [FWI, H2, 探索性, 独立测试]
date: 2026-10-13
base_commit: 137095a
---
# T2 独立测试探索轮

## 开跑前预注册修订（UTC {stamp}）
负责人在看过验证集描述、尚未看到任何测试结果时决定：主指标为缺陷检出率，单缺陷内部异常像素比例≥50%计检出；小孔2/3/4/6 mm合并、狭缝1/2/3格合并，分别报告检出数/总数。全部分组另报平均像素召回，只作描述，不推翻主指标，方向冲突照实写。
主判据：同几何、同种子的epoch100相对epoch0，小孔或狭缝检出率平均下降≥10个百分点，且≥4/5种子方向一致。epoch0狭缝检出率<10%时狭缝项记“无法检验”，只用小孔。此轮每类仅8个缺陷，**探索性，不作H2正式判定**；正式判定仍需每类30个的完整测试集。
A/B/C只检验初始化对漏检的影响；强度趋势只依据epoch轴。A原生epoch0，B已有单椭圆200权重，C已有扩展200 epoch100权重，配对种子30–34。此轮不补epoch10/30，不启动T3、梯度门或FNO训练。

## 固定抽样与配置
从60块有缺陷测试板中按8种类型精确分层选16块，每类恰8个、总64个缺陷；固定种子20261014产生随机线性权重，用二元约束求解满足类型配额的组合，完全不读取重建结果。另取编号最小的已有空板。选择：{', '.join(cases)}。
统一作者U-Net、初始梯度输入、BN train(batch=1)、Adam lr=5e-4、weight_decay=0、betas=(0.9,0.999)、eps=1e-8、调度$(0.2e+1)^{{-0.5}}$、裁剪5e-5、costScaling=1e10、35步；保存0/5/10/20/35步。只换初始化，不重训。
验证seed30已冻结阈值：A {thresholds['A']}，B {thresholds['B']}，C {thresholds['C']}。所有种子、中间步和测试均使用各组固定阈值，实际虚警超过目标也不重选。异常像素为$|\\gamma-1|\\ge t(1-\\gamma_{{min}})$，$\\gamma_{{min}}=10^{{-5}}$。MSE在内部256×128评分，虚警背景排除缺陷外两格。
细网格观测511×255、4000×15ns；粗网格反演255×127、2000×30ns，均4源；新观测与既有预训练分布有离散差异。已豁免差异继续按代码，未改作者代码或已有结果。
预算255次35步，预计5.3–5.9小时，上限7小时；串行运行，单项失败两次停止；T4单独计时、不占本轮反演预算。

## 局限与空板成本（不生成）
现有空板在材料、采集和无噪声条件下完全相同，只有1个独立参考；验证空板也相同，因此不声称空板泛化验证独立。更多重复文件不会增加独立信息。正式测试前建议负责人决定是否引入独立噪声或采集/背景扰动；本轮不生成、不新增该变量。
若未来批准10种独立空板条件，按已有约6秒/细网格观测、75.35秒/35步反演估算，生成约1分钟，A/B/C五种子评估150次约3.14小时，另加预处理；仅为成本情景，不是已获批任务。

## 结果
尚未运行；结果将追加，不修改以上门槛。
'''
NOTE.write_text(text,encoding='utf-8')
with (ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md').open('a',encoding='utf-8') as f:
 f.write('\n\n## T2 第二轮主指标与范围修订（开跑前）\n主指标明确为缺陷检出率；决定在看到验证集描述后、任何测试结果前作出。固定16块有缺陷板+1空板，A/B/C五种子，探索性；见[[T2 独立测试探索轮 2026-10-13]]。T3与梯度门继续停止。\n')
print(json.dumps(selection,ensure_ascii=False),flush=True)
