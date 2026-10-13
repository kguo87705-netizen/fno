"""Append results of this authorized comparison to the single research note."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.build/singh-pretrained-author-20261012'
r=json.loads((OUT/'comparison_scores.json').read_text(encoding='utf-8'))
rawtime=json.loads((OUT/'initial_only_timing.json').read_text(encoding='utf-8'))
names={'conventional':'常规FWI','nn':'NN','initial':'U-Net初值','initial_fwi':'U-Net初值+常规FWI','transfer_post5':'迁移学习FWI（更新5次后）','transfer':'迁移脚本原保存场（更新5次前）'}
methods=['conventional','nn','initial','initial_fwi','transfer_post5']
steps={'conventional':35,'nn':35,'initial':0,'initial_fwi':35,'transfer_post5':5}
nn=[json.loads((ROOT/f'.build/singh-nnfwi-author-20261012/case{c}/metrics.json').read_text(encoding='utf-8')) for c in range(1,5)]
conv=[json.loads((ROOT/f'.build/author-singh-case{c}-35epochs-20261008/run_report.json').read_text(encoding='utf-8')) for c in range(1,5)]
times={'conventional':{'kind':'历史单例总墙钟（非纯迭代）','values':[x['elapsed_seconds'] for x in conv]},
       'nn':{'kind':'此前单例迭代计时','values':[x['fit_seconds'] for x in nn]},
       'initial':{'kind':'本轮读权重/预存梯度+推理，不含梯度生成与预训练','values':[x['initial_online_seconds'] for x in rawtime]},
       'initial_fwi':{'kind':'本轮单例迭代计时，不含初值准备','values':[x['fit_seconds'] for x in r['timing'] if x['method']=='initial_fwi']},
       'transfer_post5':{'kind':'本轮单例5步迭代+更新后一次网络读出，不含初值准备','values':[x['fit_seconds']+x['post5_forward_seconds'] for x in r['timing'] if x['method']=='transfer']}}
def point(m,cap):
 a=r['working_points'][m][cap]
 return '扫描范围内不可达' if a is None else f"t={a['threshold']:.2f}；检出{a['detected_total']}/12；狭缝召回{100*a['thin_slit_recall']:.2f}%"
lines=['\n\n## 作者预训练U-Net初值及迁移学习基线（2026-10-12）','',
'问题：800样本预训练U-Net的初值本身质量如何，接作者常规FWI或迁移学习FWI后是否改善？本轮不训练新网络、不读取data.rar、不恢复FNO正演代理或加源方向。',
'输入来自负责人提供的singh_zenodo_transfer.zip：model_Unet_pretrained_800、TransferLearningFWI.py、conventionalFWIWithInitialGuess.py、Pretraining.py；已解压到约定目录。原文件未改写。',
'超参数核对：初值+常规脚本 lr=0.05，传入Adam((gammaPred,),lr=lr)，epochs=35；迁移脚本 lr=0.0005，作为Adam(model.parameters(),lr)的第二个位置参数真正传入，epochs=5；迁移脚本声明l2=1e-6但未传入weight_decay，实际weight_decay=0。均保留原样，未调参。',
'必要运行适配只有输入/输出和权重路径、初值常规脚本遗漏的model.to(device)，以及保存评分字段的钩子；初值归一化、默认train模式（含BatchNorm）、优化器、学习率调度、梯度裁剪与物理源使用均沿用作者。迁移脚本生成F并在反演中使用F，虽另加载source.h5到f但未使用f，本轮未私自替换。',
'输出口径：作者迁移循环先预测gammaPred、再更新网络权重，末尾保存的gammaPred只反映前4次更新；作者原保存场完整保留。主表另以已完成5次优化器更新后的网络读出评分，与此前NN按更新后权重评分一致；该补充仅一次网络前向，不新增优化或物理求解，并保留原train模式。原保存场的评分另列，不能混作5次更新后效果。',
'预训练及训练/测试来源：[论文](https://link.springer.com/article/10.1007/s00466-025-02600-w) §2.4.2说明800个单椭圆缺陷样本、100epochs；本地作者Readme也说明现成权重为800样本、100epochs。§3.2明确案例1不是预训练样本；§3.1/3.3–3.5描述案例2–4为多缺陷/矩形/手工十字组合，结构上不属于每样本单椭圆训练集。四例排除依据是作者论文说明及几何定义，不是本轮对训练归档逐文件查重；未读取708MB的data.rar。论文中100个测试案例不在训练集的句子单独指100例，未误当作四例的直接声明。',
'提供的Pretraining.py当前写epochs=1，并把800样本作80/20划分；它不能单独证明现成权重的实际训练历史。本轮按论文/Readme使用现成权重，不运行该预训练脚本、不把其epochs改为100。',
'零迭代初值输入是预存的首轮伴随梯度（归一化到[-1,1]），不是直接输入原始波形；其生成需要一次正演与伴随过程。初值计时不含这项历史成本，也不含800样本生成和离线训练，因此不据此声称端到端加速。独立零迭代读出与两脚本原始初值比对一致。','',
'| 方法 | 更新步数 | 整体MSE：案例1/2/3/4 | 1%合并虚警上限 | 0.5%合并虚警上限 | 四例计时合计秒 |','|---|---|---|---|---|---|']
for m in methods:
 lines.append(f"| {names[m]} | {steps[m]} | {' / '.join(f'{x:.6f}' for x in r['material_mse'][m])} | {point(m,'0.01')} | {point(m,'0.005')} | {sum(times[m]['values']):.3f} |")
lines+=['','### 各案例虚警面积比与漏检对象','',
'| 方法 | 合并上限 | 实际合并虚警% | 案例1/2/3/4虚警% | 漏检对象 |','|---|---|---|---|---|']
for m in methods+['transfer']:
 for cap in ['0.01','0.005']:
  a=r['working_points'][m][cap]
  if a:
   fp=' / '.join(f"{100*a['cases'][str(c)]['false_alarm_fraction']:.4f}" for c in range(1,5))
   lines.append(f"| {names[m]} | {100*float(cap):g}% | {100*a['false_alarm_fraction']:.4f} | {fp} | {'；'.join(a['missed_defects']) or '无'} |")
lines+=['','### 原迁移脚本保存场（保留原样）','',
f"材料MSE：{' / '.join(f'{x:.6f}' for x in r['material_mse']['transfer'])}；1%工作点：{point('transfer','0.01')}；0.5%工作点：{point('transfer','0.005')}。",'',
'### 计时口径（不混作同质量端到端速度）']
for m in methods:
 lines.append(f"- {names[m]}：{times[m]['kind']}；各例 "+' / '.join(f'{x:.3f}s' for x in times[m]['values'])+'。')
for method in ['initial_fwi','transfer']:
 rows=[x for x in r['timing'] if x['method']==method]
 lines.append(f"- {method}本轮各例总墙钟（含读数据、初值和保存）："+' / '.join(f"{x['case_wall_seconds']:.3f}s" for x in rows)+'。')
lines+=['- 零迭代纯网络推理各例：'+' / '.join(f"{x['inference_seconds']:.6f}s" for x in rawtime)+'；冷启动和CUDA初始化影响见独立初值总计时，不隐藏。','',
f"初值到35步常规FWI的MSE变化（每例最终减初值）：{' / '.join(f'{b-a:+.6f}' for a,b in zip(r['material_mse']['initial'],r['material_mse']['initial_fwi']))}。", 
'结论：初值+常规35步虽然四例MSE均比初值本身变差，但两档虚警上限下检出由5/12提高至9/12；不能把MSE变差等同于所有重建质量都变差。迁移5步后的MSE有3/4案例≤NN的1.25倍，但检出7/12仍少于NN在两工作点的12/12、10/12，细狭缝仍未检出。白话：预训练提供了更干净的起点，却会漏小缺陷；继续物理迭代能补回部分缺陷，也可能带回背景伪影。',
'本轮仅4个合成案例，检测阈值仍在同组真值上回顾性选取；检出覆盖比例不保证细裂缝形状恢复，合并虚警不保证各案例局部虚警合格。',
'重建图：`.build/singh-pretrained-author-20261012/pretrained_reconstruction_comparison.png`。逐阈值评分、末次更新前后材料场与计时分别见同目录comparison_scores.json、各案例final_fields.npz及initial_only_timing.json。',
'常规与NN全17档扫描结果与上轮完全一致；两脚本优化器真实步数与lr已检查，初值独立读出与保存初值一致。已完成并停止，不继续调参或启动其他方向。']
p=ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md'
with p.open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({'times':times,'mse':r['material_mse'],'working_points':{m:{cap:{'threshold':a['threshold'],'detected':a['detected_total'],'slit':a['thin_slit_recall'],'percase_fp':[a['cases'][str(c)]['false_alarm_fraction'] for c in range(1,5)]} if a else None for cap,a in r['working_points'][m].items()} for m in methods}},indent=2))
print('APPENDED',len(lines),'lines')
