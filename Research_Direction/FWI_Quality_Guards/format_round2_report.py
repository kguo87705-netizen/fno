"""Owner-requested presentation order; consumes saved scores, changes no endpoint."""
from common import *
ROUND=ROOT/'.build/fwi-quality-guards-round2-20261013'
note=ROOT/'PINO超声成像知识库/05实践与复习/T2 独立测试探索轮 2026-10-13.md'
report=json.loads((ROUND/'summary.json').read_text(encoding='utf-8'));data=json.loads((ROUND/'scores.json').read_text(encoding='utf-8'))
cfg=json.loads((ROUND/'config.json').read_text(encoding='utf-8'));sel=json.loads((ROUND/'selection.json').read_text(encoding='utf-8'))
types=['孔2mm','孔3mm','孔4mm','孔6mm','狭缝1格','狭缝2格','狭缝3格','椭圆']
def rows(g,step=35):return [data[g][str(s)]['0.01'][str(step)] for s in cfg['seeds']]
def interval(x,scale=1,places=2):return f"{np.mean(x)*scale:.{places}f} [{min(x)*scale:.{places}f}, {max(x)*scale:.{places}f}]"
lines=['\n## 第二轮结果（探索性；已停止）',
'证据等级：实验验证（仅本轮数值观察，全部结论标为探索性）；H2仍为假设，不作正式判定。',
f"255/255次完成，反演墙钟{report['run_seconds']/3600:.3f}小时，反演失败{report['failure_count']}次。0.5%与1%验证工作点选中相同阈值，测试结果相同；不等于测试实际虚警相同或均达标。",
'白话：先看哪些缺陷被找到了，再看误报多少，最后看平均图像误差，避免把漏检换来的低误差当成质量提升。',
'### 1. 缺陷检出率（主）',
'以下“均值[范围]”跨5个训练种子；独立测试几何仍是16块有缺陷板。每种类型8缺陷；小孔合并32、狭缝合并24、椭圆8，总64。',
'| 组 | 检出/64 | 小孔检出率 | 狭缝检出率 | 椭圆检出率 |','|---|---:|---:|---:|---:|']
for g in ['A','B','C']:
 rr=rows(g)
 lines.append(f"| {g} | {interval([r['detected'] for r in rr],places=1)} | {interval([r['holes']['detection_rate'] for r in rr],100)}% | {interval([r['slits']['detection_rate'] for r in rr],100)}% | {interval([r['types']['7']['detection_rate'] for r in rr],100)}% |")
lines+=['\n| 种子 | A检出/64 | B检出/64 | C检出/64 | A−C小孔(pp) | A−C狭缝(pp) |','|---|---:|---:|---:|---:|---:|']
for s,a,b,c in zip(cfg['seeds'],rows('A'),rows('B'),rows('C')):
 lines.append(f"| {s} | {a['detected']} | {b['detected']} | {c['detected']} | {(a['holes']['detection_rate']-c['holes']['detection_rate'])*100:.2f} | {(a['slits']['detection_rate']-c['slits']['detection_rate'])*100:.2f} |")
for family,x in report['exploratory_pattern'].items():
 lines.append(f"{family}：epoch0−epoch100检出率平均差{x['mean_drop']*100:.2f}个百分点，下降方向{x['directions_decline']}/5，地板无法检验={x['floor_unable_to_test']}；数值模式满足探索门槛={x['threshold_pattern']}。不作正式H2判定。")
lines+=['\n#### 分组Wilson 95%区间（按种子分别报）','同板缺陷相关；Wilson区间是约定的二项区间，未校正板内聚类；五种子不是五倍独立几何。','| 组 | 种子 | 类型 | 检出/总数 | 检出率 Wilson95% |','|---|---:|---|---:|---:|']
for g in ['A','B','C']:
 for r in rows(g):
  for label,x in [('小孔合并',r['holes']),('狭缝合并',r['slits'])]+[(types[k],r['types'][str(k)]) for k in range(8)]:
   lo,hi=x['wilson95'];lines.append(f"| {g} | {r['seed']} | {label} | {x['detected']}/{x['total']} | {x['detection_rate']*100:.2f}% [{lo*100:.2f}, {hi*100:.2f}] |")
lines+=['\n#### 同几何配对胜/负/平','左组相对右组；缺陷配对以检出/未检出比较，每板配对以检出数量比较。','| 左/右 | 种子 | 缺陷胜/负/平 | 每板胜/负/平 |','|---|---:|---:|---:|']
for pair,rr in report['paired'].items():
 for r in rr:lines.append(f"| {pair} | {r['seed']} | {r['left_win']}/{r['left_loss']}/{r['tie']} | {r['board_win']}/{r['board_loss']}/{r['board_tie']} |")
lines+=['\n| 左/右 | 种子 | 类型 | 缺陷胜/负/平 |','|---|---:|---|---:|']
for left,right in [('A','B'),('A','C'),('B','C')]:
 for a,b in zip(rows(left),rows(right)):
  for label,cats in [('小孔合并',range(4)),('狭缝合并',range(4,7))]+[(types[k],[k]) for k in range(8)]:
   pairs=[(x,y) for x,y in zip(a['components'],b['components']) if x['category'] in cats]
   assert all(x['case']==y['case'] and x['label']==y['label'] for x,y in pairs)
   wins=sum(x['detected']>y['detected'] for x,y in pairs);losses=sum(x['detected']<y['detected'] for x,y in pairs)
   lines.append(f"| {left}/{right} | {a['seed']} | {label} | {wins}/{losses}/{len(pairs)-wins-losses} |")
lines+=['\n### 2. 测试实际虚警（合并、逐板、空板）',
'不在测试集重选阈值。合并虚警按背景像素面积加权，含空板；逐板与空板单报。以下固定的是验证校准工作点，不声称测试实际虚警逐板等于或低于1%。',
'| 组 | 冻结阈值 | 合并虚警均值[范围] | 空板虚警均值[范围] | 合并虚警>1%的种子数 |','|---|---:|---:|---:|---:|']
for g in ['A','B','C']:
 rr=rows(g);lines.append(f"| {g} | {cfg['thresholds'][g]['0.01']} | {interval([r['false_alarm'] for r in rr],100,3)}% | {interval([r['blank_false_alarm'] for r in rr],100,3)}% | {sum(r['false_alarm']>.01 for r in rr)}/5 |")
lines+=['\n| 组 | 种子 | 板 | 实际虚警 |','|---|---:|---|---:|']
for g in ['A','B','C']:
 for r in rows(g):
  for b in r['boards']:lines.append(f"| {g} | {r['seed']} | {b['case']} | {b['false_alarm']*100:.3f}% |")
lines+=['空板只有1个独立材料/采集参考，且其无噪声观测与验证空板相同；跨种子变化是模型变化，不是独立空板变化。B验证第0步1.038%仍保留为历史描述，没有为此重选阈值。',
'\n### 3. 材料MSE','每板内部256×128材料格点；17板等权总体（含空板），同时单列16块有缺陷板。','| 组 | 17板MSE均值[范围] | 16有缺陷板MSE均值[范围] |','|---|---:|---:|']
for g in ['A','B','C']:
 rr=rows(g);lines.append(f"| {g} | {interval([r['material_mse'] for r in rr],places=6)} | {interval([r['defective_material_mse'] for r in rr],places=6)} |")
lines+=['\n| 组 | 板 | MSE五种子均值[范围] |','|---|---|---:|']
for g in ['A','B','C']:
 for i,case in enumerate(sel['cases']):lines.append(f"| {g} | {case} | {interval([r['boards'][i]['material_mse'] for r in rows(g)],places=6)} |")
lines+=['\n### 4. 0/5/10/20/35步描述性轨迹','全部使用各组第35步的验证冻结阈值，不给中间步重新选阈值；只观察保存时点，不断言保存间隙的行为。','| 组 | 反演步 | 检出/64均值[范围] | 小孔检出率 | 狭缝检出率 | 实际虚警均值 |','|---|---:|---:|---:|---:|---:|']
transitions={}
for g in ['A','B','C']:
 for step in [0,5,10,20,35]:
  rr=rows(g,step);lines.append(f"| {g} | {step} | {interval([r['detected'] for r in rr],places=1)} | {np.mean([r['holes']['detection_rate'] for r in rr])*100:.2f}% | {np.mean([r['slits']['detection_rate'] for r in rr])*100:.2f}% | {np.mean([r['false_alarm'] for r in rr])*100:.3f}% |")
 transitions[g]=[]
 for s in cfg['seeds']:
  snapshots=[data[g][str(s)]['0.01'][str(step)]['components'] for step in [0,5,10,20,35]]
  flags=np.array([[c['detected'] for c in cc] for cc in snapshots],dtype=bool)
  categories=np.array([c['category'] for c in snapshots[0]])
  rr=dict(seed=s)
  for label,mask in [('all',np.ones(64,bool)),('holes',categories<4),('slits',(categories>=4)&(categories<=6))]+[(str(k),categories==k) for k in range(8)]:
   f=flags[:,mask]
   rr[label]=dict(initial=int(f[0].sum()),final=int(f[-1].sum()),recovered_final=int((~f[0]&f[-1]).sum()),lost_final=int((f[0]&~f[-1]).sum()),never_at_saved_steps=int((~f.any(axis=0)).sum()),transient_only=int((f.any(axis=0)&~f[-1]).sum()))
  transitions[g].append(rr)
dump(ROUND/'descriptive_transitions.json',transitions)
lines+=['\n#### 初值漏检是否找回','找回=第0步漏检、第35步检出；丢失=第0步检出、第35步漏检；始终漏检只指五个保存时点均漏检。','| 组 | 种子 | 类型 | 初始检出 | 最终检出 | 找回 | 丢失 | 保存时点始终漏检 |','|---|---:|---|---:|---:|---:|---:|---:|']
for g,rr in transitions.items():
 for r in rr:
  for label,key in [('全部','all'),('小孔','holes'),('狭缝','slits')]+[(types[k],str(k)) for k in range(8)]:
   x=r[key];lines.append(f"| {g} | {r['seed']} | {label} | {x['initial']} | {x['final']} | {x['recovered_final']} | {x['lost_final']} | {x['never_at_saved_steps']} |")
lines+=['\n### 5. 平均像素召回（次要，不推翻主判据）','每个缺陷内异常像素比例先独立计算，再在类型内等权平均，最后报五种子均值[范围]。','| 组 | 类型 | 平均像素召回 |','|---|---|---:|']
for g in ['A','B','C']:
 for label,key in [('小孔合并','holes'),('狭缝合并','slits')]:lines.append(f"| {g} | {label} | {interval([r[key]['mean_pixel_recall'] for r in rows(g)],100)}% |")
 for k in range(8):lines.append(f"| {g} | {types[k]} | {interval([r['types'][str(k)]['mean_pixel_recall'] for r in rows(g)],100)}% |")
for family,x in report['exploratory_pattern'].items():lines.append(f"{family}：主检出率A−C为{x['mean_drop']*100:.2f}pp，次要像素召回A−C为{x['secondary_mean_drop']*100:.2f}pp，均值方向不一致={x['discordant_mean_direction']}。")
lines+=['\n### 局限、未解决与下一步','所有结果探索性；每类仅8缺陷，尚不作正式H2判定；测试案例少、训练种子波动明显，尚不足以可靠判断较小的检出差异。不同组独立验证阈值的保守程度、实际测试虚警差异和离散分布差异都可能影响检出，不能把任一差值全部归因于先验。',
'A/B/C检验初始化效应；先验强度结论只依据epoch轴，未执行epoch10/30。若完整测试以后支持H2，FNO先验也应接受相同的小缺陷护栏，本轮不能推出FNO有效、无效或必然需要某一种修正。',
'建议负责人决定下一轮范围和独立空板条件；T3、梯度门和FNO训练均未启动。已停止。']
text=note.read_text(encoding='utf-8');marker='\n## 第二轮结果（探索性；已停止）'
# The archived literal diff contains this title inside a Python string, not a section.
# The scorer's broad substring guard stopped note writing after all scores were saved.
# Append if the real section is absent; otherwise replace only the generated section.
prefix=text.split(marker,1)[0] if marker in text else text
note.write_text(prefix+'\n'.join(lines)+'\n',encoding='utf-8')
print('OWNER_REPORT_ORDER_APPLIED',flush=True)
