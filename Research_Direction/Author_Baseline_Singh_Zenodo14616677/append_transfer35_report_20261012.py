"""Append one authorized 35-step transfer run; do not open another branch."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.build/singh-transfer35-author-20261012'
r=json.loads((OUT/'comparison_scores.json').read_text(encoding='utf-8'))
steps=[5,10,20,35]
names={'conventional':'常规FWI（35）','nn':'NN（35）',**{f't{s}':f'迁移学习（{s}）' for s in steps}}
def wp(m,cap):
 a=r['working_points'][m][cap]
 return '不可达' if not a else f"t={a['threshold']:.2f}；{a['detected_total']}/12；狭缝{100*a['thin_slit_recall']:.2f}%"
def fmt(a):return ' / '.join(f'{x:.6f}' for x in a)
lines=['\n\n## 迁移学习延长至35步：能否找回漏检（2026-10-12）','',
'唯一训练改动：TransferLearningFWI.py的epochs从5改为35；原作者文件保留，运行时适配路径。lr=5e-4确实传入Adam；原调度5e-4·(0.2e+1)^(-0.5)不变，不重置优化器；第5/10/20/35次更新分别使用约3.727e-4、2.988e-4、2.282e-4、1.790e-4，后段继续衰减。',
'每案例只运行一次35步，不是分别重启5/10/20/35步。保存四个更新后材料场；额外网络读出后恢复所有BatchNorm缓冲区，不改变后续训练状态。没有追加物理求解、正则项、源或新训练样本。四例第5步均与上一轮更新后5步材料场一致，优化器实际步数和学习率已检查。',
'查文献（30分钟以内）：[原论文§3.4–3.7](https://link.springer.com/article/10.1007/s00466-025-02600-w)已有延长迭代可能修正不良初值的定性结果，也报告案例4部分侧孔到55步才恢复、某些缺陷100步仍恢复不佳。本轮不重复声称发现这一一般现象，只检查此前缺少的固定虚警指标；不延长到35步以外。',
'评分不变：内部256×128物理材料点；阈值0.10–0.90、步长0.05；真值连通域召回≥50%为检出；背景排除真值外两格；每方法/时点分别在1%与0.5%合并虚警上限内最大化检出数，并列取最低虚警。阈值是在同组四案例上选取，工作点不是完全相同的实际虚警值。','',
'| 方法（步数） | 整体MSE：案例1/2/3/4 | 1%合并上限 | 0.5%合并上限 |','|---|---|---|---|']
for m,n in names.items():lines.append(f"| {n} | {fmt(r['material_mse'][m])} | {wp(m,'0.01')} | {wp(m,'0.005')} |")
lines+=['','### 各案例虚警面积比（%）','',
'| 方法 | 合并上限 | 实际合并虚警% | 案例1/2/3/4虚警% | 漏检对象 |','|---|---|---|---|---|']
for m,n in names.items():
 for cap in ['0.01','0.005']:
  a=r['working_points'][m][cap]
  if a:
   fp=fmt([100*a['cases'][str(c)]['false_alarm_fraction'] for c in range(1,5)])
   lines.append(f"| {n} | {100*float(cap):g}% | {100*a['false_alarm_fraction']:.6f} | {fp} | {'；'.join(a['missed_defects']) or '无'} |")
cn={'case3:thin_slit':'案例3细狭缝','case4:left_circle_d6':'案例4左侧6毫米孔','case4:central_circle_d4':'案例4中央4毫米孔','case4:central_circle_d3':'案例4中央3毫米孔','case4:central_circle_d2':'案例4中央2毫米孔','case4:right_circle_d4':'案例4右侧4毫米孔'}
lines+=['','### 首次检出保存时点（不是精确首次跨阈值的迭代）','',
'只在5/10/20/35步评分；“始终漏检”仅指这四个保存时点，不能证明所有中间步或35步以后也漏检。每个时点重新选择阈值，因此检出变化包含材料场与阈值变化，不能单独归因于网络表示。','',
'| 缺陷 | 1%上限：首次保存步；5/10/20/35状态 | 0.5%上限：首次保存步；5/10/20/35状态 |','|---|---|---|']
for name,label in cn.items():
 vals=[]
 for cap in ['0.01','0.005']:
  a=r['selected_defects'][cap][name]
  first=str(a['first_saved_step']) if a['first_saved_step'] else '始终漏检'
  history='/'.join('检出' if x['detected'] else '漏检' for x in a['history'])
  vals.append(first+'；'+history)
 lines.append('| '+label+' | '+' | '.join(vals)+' |')
lines+=['','对应缺陷内召回率（%，按5/10/20/35步）：']
for cap in ['0.01','0.005']:
 for name,label in cn.items():
  vals=[100*x['recall'] for x in r['selected_defects'][cap][name]['history']]
  lines.append(f"- {100*float(cap):g}%，{label}："+fmt(vals)+'。')
new={}
for cap in ['0.01','0.005']:
 first=r['working_points']['t5'][cap];last=r['working_points']['t35'][cap]
 recovered=[];lost=[]
 for c in range(1,5):
  for i,(a,b) in enumerate(zip(first['cases'][str(c)]['detected'],last['cases'][str(c)]['detected'])):
   if not a and b:recovered.append(f'case{c}:label{i+1}')
   if a and not b:lost.append(f'case{c}:label{i+1}')
 ever=[]
 for c in range(1,5):
  for i,a in enumerate(first['cases'][str(c)]['detected']):
   if not a and any(r['working_points'][f't{s}'][cap]['cases'][str(c)]['detected'][i] for s in [10,20,35]):
    ever.append(f'case{c}:label{i+1}')
 new[cap]=ever
 lines.append(f"{100*float(cap):g}%上限：5步至35步检出{first['detected_total']}/12→{last['detected_total']}/12；找回{len(recovered)}个此前漏检，丢失{len(lost)}个此前检出。")
if any(new.values()):
 conclusion='判断：此前漏检可以被迭代部分修正，先验偏差没有表现为完全锁死；35步内仍漏检的对象只能判当前设置下尚未恢复，不能判永久无法恢复。白话：多迭代能找回一部分，但并非所有小缺陷都回得来。'
else:
 conclusion='判断：本轮延长至35步未在最终工作点找回5步漏检对象，当前设置呈持续漏检；但有限步数、衰减学习率与固定评分不足以证明永久锁死。白话：这次多迭代没有把漏掉的缺陷找回来，不能据此说以后永远找不回。'
lines+=['',conclusion,
'因果边界：未控制或单独比较预训练数据覆盖、网络表示能力、优化路径和衰减学习率，不能把持续漏检唯一归因于单椭圆训练集。局限仍是4个合成案例、同组真值挑阈值，以及覆盖式检出不等于裂缝形状准确恢复。',
'检出曲线：`.build/singh-transfer35-author-20261012/detection_vs_steps.png`；案例3/4材料演化：`.build/singh-transfer35-author-20261012/case3_case4_steps.png`。',
'完整17档评分、各缺陷召回及材料场分别见同目录comparison_scores.json、step5/step10/step20/step35各案例final_fields.npz；计算计时见case_times.json。',
'已停止；不延长步数、改变调度、扩充训练集或另开支线。']
p=ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md'
with p.open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(conclusion)
print('APPENDED',len(lines),'lines')
