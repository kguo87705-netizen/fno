"""CPU-only retrospective sensitivity; never imports or invokes a wave solver."""
import os,json,time,csv
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.build/author-baseline-env/matplotlib-cache'))
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

ROUND=ROOT/'.build/fwi-quality-guards-round2-20261013'
FIRST=ROOT/'.build/fwi-quality-guards-20261013'
OUT=ROUND/'posthoc-20261014'
NOTE=ROOT/'PINO超声成像知识库/05实践与复习/T2 独立测试探索轮 2026-10-13.md'
START=time.perf_counter();LIMIT=30*60
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
cfg=read(ROUND/'config.json');sel=read(ROUND/'selection.json');old=read(ROUND/'scores.json')
groups=['A','B','C'];seeds=cfg['seeds'];steps=[0,5,10,20,35]
thresholds=[float(x) for x in np.round(np.arange(.05,.95001,.05),2)]
caps=[.0001,.0005,.001,.002];targets=[50,55,60]
types=['小孔2mm','小孔3mm','小孔4mm','小孔6mm','狭缝1格','狭缝2格','狭缝3格','椭圆']
immutable=[ROUND/'config.json',ROUND/'selection.json',ROUND/'scores.json',ROUND/'summary.json',FIRST/'frozen_thresholds.json',FIRST/'accepted_geometries.json']
original_bytes={p:p.read_bytes() for p in immutable}
assert not (OUT/'summary.json').exists(),'Do not overwrite an existing posthoc analysis'
text=NOTE.read_text(encoding='utf-8');assert '\n## 事后敏感性分析\n' not in text
stamp=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
header=f'''
## 事后敏感性分析

**事后分析，不替代预注册主判据。** 本小节全部结果为探索性；记录时间（北京时间）：{stamp}。
只读取现有255次反演，不重新反演。原冻结阈值、抽样和主判据不变；以下扫描结果不写回原阈值、原scores或原summary。

### 计算前固定的事后分析规则
第35步分别对A/B/C各5种子扫描0.05–0.95、步长0.05。虚警背景仍排除所有缺陷外两格，合并虚警为17板背景异常像素总数/背景像素总数，含1空板。缺陷检出定义仍为内部异常像素占比≥50%。
离散扫描通常不能恰好达到指定虚警。因此四个工作点解释为**实际合并虚警≤0.01%/0.05%/0.1%/0.2%**：选择满足上限的最低阈值（即最宽松可行档，不用检出数量打破平局）；不插值、不外推。若扫描范围内无可行档记“无可用阈值”；若最低扫描档仍低于上限，明确标记范围右端未覆盖目标。另报各自实际虚警，不能声称精确同虚警。
逐缺陷配对在同几何、同训练种子下比较64个标签，报告左组胜/负/平；5次模型重建不视为320个独立缺陷。小孔与狭缝各类型召回同时保留。
速度账只用原第35步验证冻结阈值，读取已保存0/5/10/20/35步的检出数，取第一个达到50/55/60的保存点；不重新选阈值、不插值，也不把未达到者当35步达标。这是事后质量工作量，不能当作可部署的真实早停准则。
一次物理调用指4源batch求解；完成n个更新，每板n次正演+n次伴随=2n次，17板共34n次。35步另有每板1次最终波形核验正演，单列；输入初始梯度另需每板1正演+1伴随（17板34次），属于已完成、各组共享的T1预处理，不混进更新次数。若按单源求解工作量，以上调用再×4。
'''
OUT.mkdir(parents=True,exist_ok=True)
with NOTE.open('a',encoding='utf-8') as f:f.write(header)
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def mean_range(x,places=1):return '无可用结果' if not x else f'{np.mean(x):.{places}f} [{min(x):.{places}f},{max(x):.{places}f}]'
def wilson(k,n):
 z=1.959963984540054;p=k/n;den=1+z*z/n
 mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [float(max(0,mid-half)),float(min(1,mid+half))]
metadata={r['id']:r for r in read(FIRST/'accepted_geometries.json')}
ids=[];cache={};file_stats={};scans={}
for g in groups:
 scans[g]={}
 for seed in seeds:
  epoch=0 if g=='A' else 100
  false=np.zeros(len(thresholds),np.int64);denominator=0;detected=[];pixels=[]
  for case in sel['cases']:
   assert time.perf_counter()-START<LIMIT,'30-minute analysis limit'
   path=ROUND/('group_'+g)/f'seed{seed}_epoch{epoch}'/case/'fields.npz'
   file_stats[path]=(path.stat().st_size,path.stat().st_mtime_ns)
   with np.load(path) as f:
    pred=f['gamma_step35'];labels=f['labels'];assert np.array_equal(pred,f['prediction'])
    if case not in cache:
     masks=[labels==i for i in range(1,len(metadata[case]['components'])+1)]
     bg=~ndimage.binary_dilation(labels>0,structure=np.ones((3,3)),iterations=2)
     cache[case]=(labels.copy(),masks,bg)
    canonical,masks,bg=cache[case];assert np.array_equal(labels,canonical)
    denominator+=int(bg.sum());d=np.abs(pred-1)
    per=np.zeros((len(thresholds),len(masks)),float)
    for j,t in enumerate(thresholds):
     abnormal=d>=t*(1-1e-5)  # identical scalar comparison to the registered scorer
     false[j]+=int((abnormal&bg).sum())
     for k,mask in enumerate(masks):per[j,k]=float(abnormal[mask].mean())
    pixels.append(per);detected.append(per>=.5)
    if g=='A' and seed==seeds[0]:ids.extend(dict(case=case,label=i,category=p['category']) for i,p in enumerate(metadata[case]['components'],1))
  flags=np.concatenate(detected,axis=1);pixel=np.concatenate(pixels,axis=1)
  assert flags.shape==(19,64)
  rows=[]
  for j,t in enumerate(thresholds):
   typescores={}
   for k in range(8):
    mask=np.array([c['category']==k for c in ids]);n=int(mask.sum());tp=int(flags[j,mask].sum());assert n==8
    typescores[str(k)]=dict(detected=tp,total=n,recall=tp/n,mean_pixel_recall=float(pixel[j,mask].mean()),wilson95=wilson(tp,n))
   rows.append(dict(threshold=t,false_pixels=int(false[j]),background_pixels=denominator,false_alarm=float(false[j]/denominator),detected=int(flags[j].sum()),flags=flags[j].astype(int).tolist(),types=typescores))
  assert np.all(np.diff(false)<=0) and np.all(np.diff(flags.sum(axis=1))<=0)
  frozen=next(r for r in rows if r['threshold']==cfg['thresholds'][g]['0.01'])
  registered=old[g][str(seed)]['0.01']['35']
  assert frozen['detected']==registered['detected'] and frozen['false_alarm']==registered['false_alarm']
  assert frozen['flags']==[x['detected'] for x in registered['components']]
  scans[g][str(seed)]=rows
  print('SCANNED',g,seed,flush=True)
dump(OUT/'threshold_scan.json',dict(disclaimer='事后分析，不替代预注册主判据',thresholds=thresholds,defects=ids,scans=scans))
work={};paired={}
for cap in caps:
 key=f'{cap:.4f}';work[key]={};paired[key]={}
 for g in groups:
  work[key][g]=[]
  for seed in seeds:
   eligible=[r for r in scans[g][str(seed)] if r['false_alarm']<=cap]
   chosen=eligible[0] if eligible else None
   work[key][g].append(dict(seed=seed,selected=chosen,cap_above_scanned_max=bool(chosen and chosen['threshold']==thresholds[0] and chosen['false_alarm']<cap)))
 for left,right in [('A','B'),('A','C'),('B','C')]:
  pp=[]
  for l,r in zip(work[key][left],work[key][right]):
   assert l['seed']==r['seed']
   if l['selected'] is None or r['selected'] is None:
    pp.append(dict(seed=l['seed'],available=False));continue
   x=np.array(l['selected']['flags']);y=np.array(r['selected']['flags'])
   def pair(mask):
    return dict(win=int((x[mask]>y[mask]).sum()),loss=int((x[mask]<y[mask]).sum()),tie=int((x[mask]==y[mask]).sum()))
   cats=np.array([c['category'] for c in ids])
   pp.append(dict(seed=l['seed'],available=True,all=pair(np.ones(64,bool)),holes=pair(cats<4),slits=pair((cats>=4)&(cats<=6))))
  paired[key][left+'/'+right]=pp
speed={}
for g in groups:
 speed[g]=[]
 for seed in seeds:
  trajectory={str(n):old[g][str(seed)]['0.01'][str(n)]['detected'] for n in steps}
  for target in targets:
   first=next((n for n in steps if trajectory[str(n)]>=target),None)
   operating=None if first is None else old[g][str(seed)]['0.01'][str(first)]
   speed[g].append(dict(seed=seed,target=target,first_saved_step=first,trajectory=trajectory,
    actual_false_alarm_at_first=None if operating is None else operating['false_alarm'],
    forward_calls_per_board=first,adjoint_calls_per_board=first,
    update_calls_per_board=None if first is None else 2*first,
    update_calls_17_boards=None if first is None else 34*first,
    final_wave_check_forward_calls_17_boards=None if first is None else (17 if first==35 else 0),
    total_excluding_shared_initial_gradient=None if first is None else 34*first+(17 if first==35 else 0)))

plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
fig,axes=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
colors=plt.cm.tab10(np.arange(5))
maxf=max(r['false_alarm']*100 for g in groups for ss in scans[g].values() for r in ss)
for ax,g,title in zip(axes,groups,['A 随机初始化','B 单椭圆200','C 扩展200']):
 for color,seed in zip(colors,seeds):
  rr=scans[g][str(seed)];ax.plot([r['false_alarm']*100 for r in rr],[r['detected'] for r in rr],'-o',ms=2.6,lw=1.1,color=color,label=f'seed{seed}')
 for cap in caps:ax.axvline(cap*100,color='gray',lw=.6,ls='--',alpha=.7)
 ax.set_xscale('symlog',linthresh=.001);ax.set_xlim(0,max(.22,maxf*1.03));ax.set_ylim(-1,65)
 ax.set_title(title);ax.set_xlabel('测试实际合并虚警（%）');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0].set_ylabel('检出缺陷数 / 64')
fig.suptitle('事后分析，不替代预注册主判据｜第35步、阈值0.05–0.95、各5种子')
fig.text(.5,.015,'虚线：0.01% / 0.05% / 0.1% / 0.2%；线段仅连接离散扫描点，不用于插值',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(OUT/'detection_vs_actual_false_alarm.png',dpi=300);plt.close(fig)
with (OUT/'curve_points.csv').open('w',encoding='utf-8-sig',newline='') as f:
 writer=csv.writer(f);writer.writerow(['group','seed','threshold','actual_pooled_false_alarm_percent','detected_of_64'])
 for g in groups:
  for seed in seeds:
   writer.writerows([g,seed,r['threshold'],r['false_alarm']*100,r['detected']] for r in scans[g][str(seed)])

lines=['\n### 同虚警上限的事后比较','下表为可用种子的均值[范围]；缺档时明确标注n，不把缺档当零检出。采用各自测试扫描阈值，**不是**新的冻结阈值。配对仅使用两组都可用的同种子；不作正式H2检验。',
'| 实际虚警上限 | A检出/64 | B检出/64 | C检出/64 | A/B胜/负/平（均值） | A/C胜/负/平（均值） | B/C胜/负/平（均值） |','|---|---:|---:|---:|---:|---:|---:|']
compact={}
for cap in caps:
 key=f'{cap:.4f}';counts={g:[x['selected']['detected'] for x in work[key][g] if x['selected']] for g in groups}
 triples={pair:'/'.join(f"{np.mean([x['all'][k] for x in rr if x['available']]):.1f}" for k in ['win','loss','tie']) if any(x['available'] for x in rr) else '无配对' for pair,rr in paired[key].items()}
 values={g:mean_range(counts[g])+(f'（仅{len(counts[g])}/5可用）' if len(counts[g])!=5 else '') for g in groups}
 lines.append(f"| {cap*100:g}% | {values['A']} | {values['B']} | {values['C']} | {triples['A/B']} | {triples['A/C']} | {triples['B/C']} |")
 compact[key]=dict(detections={g:dict(mean=float(np.mean(v)) if v else None,range=[min(v),max(v)] if v else None,available=len(v)) for g,v in counts.items()},pair_means=triples)
lines+=['\n#### 逐种子选档与实际虚警（缺档不当成零检出）','| 上限 | 组 | 种子 | 事后阈值 | 实际虚警 | 检出/64 | 右端未覆盖目标 |','|---|---|---:|---:|---:|---:|---|']
for cap in caps:
 for g in groups:
  for row in work[f'{cap:.4f}'][g]:
   x=row['selected']
   lines.append(f"| {cap*100:g}% | {g} | {row['seed']} | {x['threshold'] if x else '无可用阈值'} | {format(x['false_alarm']*100,'.6f')+'%' if x else '未覆盖'} | {x['detected'] if x else '未测'} | {row['cap_above_scanned_max']} |")
lines+=['\n#### 逐缺陷配对明细','| 上限 | 左/右 | 种子 | 全部胜/负/平 | 小孔胜/负/平 | 狭缝胜/负/平 |','|---|---|---:|---:|---:|---:|']
for cap in caps:
 for pair,rr in paired[f'{cap:.4f}'].items():
  for row in rr:
   triple=lambda family:'/'.join(str(row[family][k]) for k in ['win','loss','tie']) if row['available'] else '无配对'
   lines.append(f"| {cap*100:g}% | {pair} | {row['seed']} | {triple('all')} | {triple('holes')} | {triple('slits')} |")
lines+=['\n#### 分类型召回（各类8缺陷，五种子均值[范围]）','| 上限 | 组 | 类型 | 检出率 |','|---|---|---|---:|']
for cap in caps:
 for g in groups:
  for k,label in enumerate(types):
   vals=[100*x['selected']['types'][str(k)]['recall'] for x in work[f'{cap:.4f}'][g] if x['selected']]
   lines.append(f"| {cap*100:g}% | {g} | {label} | {mean_range(vals,2)}% |")
lines+=['\n### 冻结阈值下的保存步速度账','未达到者不进入达标步数均值；因此不能只比较成功者的均值来宣称加速。4源batch逻辑调用的数量不等于实测墙钟加速。','| 组 | 目标检出 | 种子30/31/32/33/34首次保存步 | 达标数/5 | 达标者步数均值[范围] |','|---|---:|---|---:|---:|']
speed_compact={}
for g in groups:
 speed_compact[g]={}
 for target in targets:
  rr=[x for x in speed[g] if x['target']==target];hits=[x['first_saved_step'] for x in rr if x['first_saved_step'] is not None]
  lines.append(f"| {g} | ≥{target} | {' / '.join(str(x['first_saved_step']) if x['first_saved_step'] is not None else '未达到' for x in rr)} | {len(hits)}/5 | {mean_range(hits)} |")
  speed_compact[g][str(target)]=dict(steps=[x['first_saved_step'] for x in rr],reached=len(hits),mean=float(np.mean(hits)) if hits else None)
lines+=['\n| 组 | 种子 | 目标 | 首次保存步 | 每板正演+伴随 | 17板更新调用 | 17板额外35步核验正演 | 合计（不含共享初始梯度） | 首次点合并实际虚警 |','|---|---:|---:|---|---:|---:|---:|---:|---:|']
for g,rr in speed.items():
 for x in rr:
  val=lambda k:str(x[k]) if x[k] is not None else '未达到'
  fpr='未达到' if x['actual_false_alarm_at_first'] is None else f"{100*x['actual_false_alarm_at_first']:.4f}%"
  lines.append(f"| {g} | {x['seed']} | ≥{x['target']} | {val('first_saved_step')} | {val('update_calls_per_board')} | {val('update_calls_17_boards')} | {val('final_wave_check_forward_calls_17_boards')} | {val('total_excluding_shared_initial_gradient')} | {fpr} |")
lines+=['物理账核对：完整35步每板实际36正演+35伴随=71次；17板1207次/组/种子，全部255次共18105次4源batch调用。这里是历史已有调用，不是本轮新计算。共享初始梯度17板另34次；其数据生成和训练成本未被这个速度账取代。',
'\n### 局限与材料MSE','事后在测试真值上选择可行阈值，结果只用于敏感性，不替代独立验证冻结的主分析。离散步长0.05可能导致实际虚警差距；逐种子选档和范围缺口全部披露。仅16有缺陷板、每类8缺陷、1空板参考；不把同一缺陷五次重建当独立样本。',
'本轮阈值扫描不改变材料场，MSE沿用原17板均值：A0.006906、B0.002826、C0.002315；平均像素召回与逐种子Wilson数值随选档保存在threshold_scan.json的types字段。',
'图：`'+str((OUT/'detection_vs_actual_false_alarm.png').relative_to(ROOT)).replace('\\','/')+'`；点表curve_points.csv；完整数值summary.json/threshold_scan.json。']
elapsed=time.perf_counter()-START
assert elapsed<LIMIT
assert all(p.read_bytes()==raw for p,raw in original_bytes.items())
assert all((p.stat().st_size,p.stat().st_mtime_ns)==v for p,v in file_stats.items())
result=dict(disclaimer='事后分析，不替代预注册主判据',started_beijing=stamp,elapsed_seconds=elapsed,wall_budget_seconds=LIMIT,base_commit='98b3eb6',operating_rule='lowest threshold with actual test pooled FPR <= cap; no interpolation/extrapolation',caps=caps,thresholds=thresholds,workpoints=work,paired=paired,compact=compact,speed=speed,speed_compact=speed_compact,source_batch=4,additional_inversions=0,immutable_inputs_unchanged=True)
dump(OUT/'summary.json',result)
lines.append(f'本轮CPU分析脚本耗时{elapsed:.2f}秒，新增反演0次；冻结阈值、抽样、主判据及255份原始材料场均保持不变。未启动补齐44板、T3、梯度门或FNO训练，未推送GitHub。已停止。')
with NOTE.open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(compact=compact,speed_compact=speed_compact,elapsed_seconds=elapsed),ensure_ascii=False),flush=True)
