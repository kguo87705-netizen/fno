"""Only frozen-threshold scoring on approved test fields; no optimization."""
from collections import Counter,defaultdict
from scipy import ndimage
from common import *
from evaluate import wilson
ROUND=ROOT/'.build/fwi-quality-guards-round2-20261013'
NOTE=ROOT/'PINO超声成像知识库/05实践与复习/T2 独立测试探索轮 2026-10-13.md'
TYPES=['孔2mm','孔3mm','孔4mm','孔6mm','狭缝1格','狭缝2格','狭缝3格','椭圆']
cfg=json.loads((ROUND/'config.json').read_text());sel=json.loads((ROUND/'selection.json').read_text())
metadata={r['id']:r for r in json.loads((BUILD/'accepted_geometries.json').read_text())}
status=json.loads((ROUND/'status.json').read_text());assert status['completed']==255 and status['status']=='completed; stopped'
def score(group,seed,cap,step):
 t=cfg['thresholds'][group][cap];epoch=0 if group=='A' else 100
 false=bgcount=0;components=[];boards=[];mse=[];blank=None;wall=0
 for case in sel['cases']:
  folder=ROUND/('group_'+group)/f'seed{seed}_epoch{epoch}'/case
  with np.load(folder/'fields.npz') as f:
   pred=f[f'gamma_step{step}'];truth=f['truth'];labels=f['labels']
   assert pred.shape==truth.shape==labels.shape==(256,128)
   assert np.isfinite(pred).all() and pred.min()>=GMIN-1e-6 and pred.max()<=1+1e-6
   assert len(metadata[case]['components'])==int(labels.max())
   if step==35:assert np.array_equal(pred,f['prediction'])
   m=float(np.mean((pred.astype(float)-truth)**2));mse.append(m)
   abnormal=abs(pred-1)>=t*(1-GMIN)
   bg=~ndimage.binary_dilation(labels>0,structure=np.ones((3,3)),iterations=2)
   nfalse=int((abnormal&bg).sum());nbg=int(bg.sum());false+=nfalse;bgcount+=nbg
   detections=[]
   for i,p in enumerate(metadata[case]['components'],1):
    mask=labels==i;assert mask.any();pixel=float(abnormal[mask].mean());det=int(pixel>=.5)
    components.append(dict(case=case,label=i,category=p['category'],detected=det,pixel_recall=pixel))
    detections.append(det)
   boards.append(dict(case=case,material_mse=m,false_alarm=nfalse/nbg,detected=sum(detections),total=len(detections)))
   if metadata[case]['blank']:blank=nfalse/nbg
   if step==35:
    metrics=json.loads((folder/'metrics.json').read_text());assert abs(metrics['material_mse']-m)<1e-12
    wall+=metrics['wall_seconds']
 def pool(cats):
  items=[x for x in components if x['category'] in cats];n=len(items);k=sum(x['detected'] for x in items)
  return dict(detected=k,total=n,detection_rate=k/n,wilson95=wilson(k,n),mean_pixel_recall=float(np.mean([x['pixel_recall'] for x in items])))
 return dict(group=group,seed=seed,cap=cap,threshold=t,step=step,detected=sum(x['detected'] for x in components),total=len(components),material_mse=float(np.mean(mse)),defective_material_mse=float(np.mean([x['material_mse'] for x in boards if x['total']])),false_alarm=false/bgcount,blank_false_alarm=blank,holes=pool(range(4)),slits=pool(range(4,7)),types={str(k):pool([k]) for k in range(8)},boards=boards,components=components,wall_seconds=wall)
results={}
for group in ['A','B','C']:
 results[group]={}
 for seed in cfg['seeds']:
  results[group][str(seed)]={cap:{str(step):score(group,seed,cap,step) for step in [0,5,10,20,35]} for cap in ['0.01','0.005']}
dump(ROUND/'scores.json',results)
def rows(group,cap='0.01',step=35):return [results[group][str(s)][cap][str(step)] for s in cfg['seeds']]
def stats(x):return dict(mean=float(np.mean(x)),range=[float(min(x)),float(max(x))])
summary={}
for group in results:
 rr=rows(group)
 summary[group]=dict(material_mse=stats([r['material_mse'] for r in rr]),defective_material_mse=stats([r['defective_material_mse'] for r in rr]),detected=stats([r['detected'] for r in rr]),false_alarm=stats([r['false_alarm'] for r in rr]),blank_false_alarm=stats([r['blank_false_alarm'] for r in rr]),holes=stats([r['holes']['detection_rate'] for r in rr]),slits=stats([r['slits']['detection_rate'] for r in rr]),types={str(k):dict(detection=stats([r['types'][str(k)]['detection_rate'] for r in rr]),pixel=stats([r['types'][str(k)]['mean_pixel_recall'] for r in rr])) for k in range(8)})
comparisons={}
for left,right in [('A','B'),('A','C'),('B','C')]:
 comparisons[left+'_'+right]=[]
 for l,r in zip(rows(left),rows(right)):
  d=[x['detected']-y['detected'] for x,y in zip(l['components'],r['components'])]
  board=[x['detected']-y['detected'] for x,y in zip(l['boards'],r['boards']) if x['total']]
  comparisons[left+'_'+right].append(dict(seed=l['seed'],left_win=sum(x>0 for x in d),left_loss=sum(x<0 for x in d),tie=sum(x==0 for x in d),board_win=sum(x>0 for x in board),board_loss=sum(x<0 for x in board),board_tie=sum(x==0 for x in board)))
exploratory={}
for family in ['holes','slits']:
 drops=[a[family]['detection_rate']-c[family]['detection_rate'] for a,c in zip(rows('A'),rows('C'))]
 pixel_drops=[a[family]['mean_pixel_recall']-c[family]['mean_pixel_recall'] for a,c in zip(rows('A'),rows('C'))]
 floor=family=='slits' and summary['A']['slits']['mean']<.1
 exploratory[family]=dict(epoch0_minus_epoch100=drops,mean_drop=float(np.mean(drops)),directions_decline=sum(x>0 for x in drops),floor_unable_to_test=floor,threshold_pattern=bool(not floor and np.mean(drops)>=.10 and sum(x>0 for x in drops)>=4),secondary_pixel_drop=pixel_drops,secondary_mean_drop=float(np.mean(pixel_drops)),discordant_mean_direction=bool(np.mean(drops)*np.mean(pixel_drops)<0))
report=dict(evidence='实验验证（仅本轮数值观察；探索性，不作H2正式判定）',summary=summary,paired=comparisons,exploratory_pattern=exploratory,run_seconds=status['elapsed_seconds'],failure_count=len(status['failures']),thresholds=cfg['thresholds'],limitation='Eight defects/type, five dependent repeats per geometry; Wilson intervals reported separately by seed, do not treat320 correlated defects as320 independent samples. Only one independent blank reference; its validation counterpart is identical.')
dump(ROUND/'summary.json',report)
lines=['\n## 第二轮结果（探索性；已停止）\n','证据等级：实验验证（仅本轮数值观察）；H2仍为假设，不作正式判定。',f"运行255/255次，反演墙钟{status['elapsed_seconds']/3600:.3f}小时，失败{len(status['failures'])}次。测试阈值未更改；每组在所有训练种子共用各自在验证seed30冻结的阈值。0.5%和1%工作点阈值相同，故这两档结果相同；实际虚警仍另报。",'MSE总体按17块板等权平均（含1空板）；原始summary另报16块有缺陷板的均值，每板值见下表。五种子不增加独立几何数量。','白话：同一批缺陷，用相同网络和物理优化，只换起始权重，比较漏检是否随之改变；这批样本还不足以作正式H2结论。','\n| 组别 | MSE均值[范围] | 检出/64均值[范围] | 小孔检出率 | 狭缝检出率 | 实际虚警均值[范围] | 空板虚警均值 |','|---|---:|---:|---:|---:|---:|---:|']
for g,s in summary.items():
 def fmt(k,scale=1,places=4):
  x=s[k];return f"{x['mean']*scale:.{places}f} [{x['range'][0]*scale:.{places}f}, {x['range'][1]*scale:.{places}f}]"
 lines.append(f"| {g} | {fmt('material_mse',places=6)} | {fmt('detected',places=1)} | {fmt('holes',100,2)}% | {fmt('slits',100,2)}% | {fmt('false_alarm',100,3)}% | {s['blank_false_alarm']['mean']*100:.3f}% |")
lines+=['\n### 逐种子主指标与配对','| 种子 | A检出/64 | B检出/64 | C检出/64 | A−C小孔(pp) | A−C狭缝(pp) |','|---|---:|---:|---:|---:|---:|']
for seed,a,b,c in zip(cfg['seeds'],rows('A'),rows('B'),rows('C')):
 lines.append(f"| {seed} | {a['detected']} | {b['detected']} | {c['detected']} | {(a['holes']['detection_rate']-c['holes']['detection_rate'])*100:.2f} | {(a['slits']['detection_rate']-c['slits']['detection_rate'])*100:.2f} |")
for family,x in exploratory.items():
 lines.append(f"{family}：epoch0−epoch100平均{x['mean_drop']*100:.2f}个百分点，下降方向{x['directions_decline']}/5；地板无法检验={x['floor_unable_to_test']}；探索性数值模式满足门槛={x['threshold_pattern']}。次要像素召回平均差{x['secondary_mean_drop']*100:.2f}个百分点，主次方向冲突={x['discordant_mean_direction']}。不据此正式支持或反驳H2。")
lines+=['\n### 各类型召回与Wilson 95%区间（每个种子各自8缺陷）','Wilson区间描述单个种子下的几何采样不确定性；同板缺陷仍相关，不把五次重建当五份独立测试。','| 组别 | 种子 | 类型 | 检出/8 | 检出率 Wilson95% | 平均像素召回 |','|---|---:|---|---:|---:|---:|']
for g in ['A','B','C']:
 for r in rows(g):
  for k in range(8):
   x=r['types'][str(k)];lo,hi=x['wilson95']
   lines.append(f"| {g} | {r['seed']} | {TYPES[k]} | {x['detected']}/8 | {x['detection_rate']*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}] | {x['mean_pixel_recall']*100:.2f}% |")
lines+=['\n### 同缺陷配对胜/负/平','| 左组/右组 | 种子 | 缺陷胜/负/平 | 每板胜/负/平 |','|---|---:|---:|---:|']
for name,rr in comparisons.items():
 for r in rr:lines.append(f"| {name} | {r['seed']} | {r['left_win']}/{r['left_loss']}/{r['tie']} | {r['board_win']}/{r['board_loss']}/{r['board_tie']} |")
lines+=['\n### 每案例实际虚警与MSE（最终35步）','| 组别 | 种子 | 板 | MSE | 实际虚警 | 检出 |','|---|---:|---|---:|---:|---:|']
for g in ['A','B','C']:
 for r in rows(g):
  for b in r['boards']:lines.append(f"| {g} | {r['seed']} | {b['case']} | {b['material_mse']:.6f} | {b['false_alarm']*100:.3f}% | {b['detected']}/{b['total']} |")
lines+=['\n### 中间步（描述性，沿用第35步冻结阈值）','| 组别 | 步数 | MSE五种子均值 | 检出/64均值 | 虚警均值 | 小孔检出率 | 狭缝检出率 |','|---|---:|---:|---:|---:|---:|---:|']
for g in ['A','B','C']:
 for step in [0,5,10,20,35]:
  rr=rows(g,step=step)
  lines.append(f"| {g} | {step} | {np.mean([r['material_mse'] for r in rr]):.6f} | {np.mean([r['detected'] for r in rr]):.1f} | {np.mean([r['false_alarm'] for r in rr])*100:.3f}% | {np.mean([r['holes']['detection_rate'] for r in rr])*100:.2f}% | {np.mean([r['slits']['detection_rate'] for r in rr])*100:.2f}% |")
lines+=['\n### 未解决与下一步','正式H2判定尚未做；每类型只有8个缺陷，空板只有1个独立参考，阈值只由验证seed30校准，其他种子实际虚警可能越界。A/B/C不是同一先验强度轴；未执行epoch10/30。FNO作为先验或正演代理的角色未定，本轮不能推出FNO的效果；若完整测试确认预训练造成漏检，FNO先验也必须通过相同小缺陷质量护栏。', '建议负责人决定是否批准完整测试和独立空板条件；T3与梯度门未启动。已停止。']
assert '## 第二轮结果（探索性；已停止）' not in NOTE.read_text(encoding='utf-8')
with NOTE.open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(report,ensure_ascii=False),flush=True)
