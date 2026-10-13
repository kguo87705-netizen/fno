"""Append this authorized round and <=20-line weekly summary to the one note."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'.build/singh-multiscale-author-20261012'
r=json.loads((OUT/'comparison_scores.json').read_text(encoding='utf-8'))
best=r['best_multiscale']
names={'conventional':'常规 FWI','nn':'NN','tv_k1':'TV k=1','ms250':'多尺度250 kHz','ms350':'多尺度350 kHz'}
def mse(m):return ' / '.join(f'{x:.5f}' for x in r['material_mse'][m])
def wp(m,cap):
 a=r['working_points'][m][cap]
 return '无满足上限的阈值' if a is None else f"t={a['threshold']:.2f}；{a['detected_total']}/12；狭缝{100*a['thin_slit_recall']:.2f}%"
lines=['\n\n## 低频先行多尺度对照（2026-10-12）','',
'问题：作者常规FWI只改前15步的数据频带、总共35步，能否使至少3个案例的整体材料MSE≤NN的1.25倍，同时保留缺陷检出？',
'文献检索（30分钟以内）：Bunks et al., 1995, Multiscale seismic waveform inversion, Geophysics 60(5),1457–1473, DOI https://doi.org/10.1190/1.1443880 提出了低频到高频的尺度推进；出版商全文入口受限，另核对原始研究 https://academic.oup.com/jge/article/15/6/2330/5209789 的引言与公式，确认低通数据及其梯度作用。文献支持标准策略，不给出本项目四案例、35步和指定频带的现成答案。',
'源频谱：采样dt=30ns、2000点，频率间隔16.67 kHz，主峰500 kHz；250/350 kHz以下能量占比3.896%/16.986%，带内峰值幅度分别为全谱峰值的45.33%/77.99%。两档均有实际低频信号，250 kHz更弱；未从无信号频段凭空生成低频。能量占比是理想截止下的频谱累计值，不是平滑滤波后的能量占比。',
'实现：以作者conventionalFWI.py为基础，lr=0.08、Adam、clipGrad=1e-5、costScaling=1e12、γ边界及源/传感器梯度屏蔽均保留；不加TV、不重置Adam。前15步对观测与模拟波形使用相同低通，后20步全频。滤波固定四阶数字Butterworth双向幅频响应 |H|²（截止处幅度1/2），用8192点零延拓FFT施加实偶响应再裁回2000点；无相位偏移、无幅度归一化。伴随源使用Lᵀ(L模拟−L观测)，没有只过滤失配却继续使用全频残差。作者原有代价积分及连续伴随近似未另作修正。',
'最佳档定义：按四案例整体材料MSE的算术平均选两档中较低者，是这四案例上的回顾性oracle选择。检出沿用上轮的17档阈值扫描、真值连通域及背景区，检出数并列取更低虚警；阈值同样在这四案例上选取。','',
'| 方法 | 材料MSE：案例1/2/3/4 | 平均MSE | ≤1.25×NN案例数 | 1%上限工作点 | 0.5%上限工作点 |','|---|---|---|---|---|---|']
for m in names:
 lines.append(f"| {names[m]} | {mse(m)} | {r['mean_material_mse'][m]:.5f} | {r['mse_pass_count'][m]}/4 | {wp(m,'0.01')} | {wp(m,'0.005')} |")
lines+=['',f"按平均MSE，最佳档为{names[best]}；它有{r['mse_pass_count'][best]}/4案例满足MSE门槛。其平均MSE为{r['mean_material_mse'][best]:.5f}，常规为{r['mean_material_mse']['conventional']:.5f}；1%和0.5%虚警上限下检出数分别为{r['working_points'][best]['0.01']['detected_total']}/12、{r['working_points'][best]['0.005']['detected_total']}/12，均低于常规、NN与TV k=1。白话：本轮低频打底没有让重建接近NN，限制误报后反而丢掉更多缺陷。",'250 kHz档在0.10–0.90的17档扫描中没有满足1%或0.5%上限的阈值；最低合并虚警为1.1397%，因此对应工作点应报不可达，不能扩展阈值范围补救或填作零检出。','',
'### 各案例虚警（面积比%，按案例1/2/3/4排列）','',
'| 方法 | 合并上限 | 选定阈值 | 合并实际虚警 % | 各案例虚警 % | 漏检对象 |','|---|---|---|---|---|---|']
for m in names:
 for cap in ['0.01','0.005']:
  a=r['working_points'][m][cap]
  if a:
   local=' / '.join(f"{100*a['cases'][str(c)]['false_alarm_fraction']:.4f}" for c in range(1,5))
   lines.append(f"| {names[m]} | {100*float(cap):g}% | {a['threshold']:.2f} | {100*a['false_alarm_fraction']:.4f} | {local} | {'；'.join(a['missed_defects']) or '无'} |")
lines+=['','两档多尺度各四案例均35步、各运行一次，未调整学习率或滤波器来挽救结果。各案例总墙钟时间（含读数据及保存）如下：']
for fc in [250000,350000]:
 vals=[a['case_wall_seconds'] for a in r['case_wall_seconds'] if a['cutoff_hz']==fc]
 lines.append(f"- {fc//1000} kHz："+' / '.join(f'{x:.2f}s' for x in vals)+f'；合计{sum(vals):.2f}s。')
lines+=['',f"本轮八案例总墙钟（逐案例累计）{sum(a['case_wall_seconds'] for a in r['case_wall_seconds']):.2f}s，低于30分钟预算；无失败重跑。",'源频谱图：`.build/singh-multiscale-author-20261012/source_spectrum_lowpass.png`。',
'材料场对比：`.build/singh-multiscale-author-20261012/multiscale_reconstruction_comparison.png`。',
'1%工作点二值叠图：`.build/singh-multiscale-author-20261012/multiscale_1pct_detection_comparison.png`。',
'完整17档扫描、逐缺陷召回、所有MSE及运行时间：`.build/singh-multiscale-author-20261012/comparison_scores.json`。',
'检查：沿用评分实现，常规/NN/TV全17档结果与上轮完全一致；滤波自伴随内积检查通过，后20步滤波返回原张量；全部8例有限值且γ边界满足约束。未追加参数搜索，已停止，多源实验未启动。','',
'## 本周小结（不超过20行）']
weekly=[
'本周的问题：在作者四个合成案例上，35步常规FWI加入标准手段，能否接近NN重参数化FWI？',
'统一口径：内部256×128材料点、不含ghost cells；至少3例MSE≤NN的1.25倍才算接近。',
'补充检出：真值缺陷内异常像素占比≥50%算检出；背景排除缺陷外两格，比较合并虚警1%和0.5%工作点。',
'常规FWI：1%/0.5%上限下分别检出7/12、5/12；细狭缝均未检出。',
'作者NN：对应检出12/12、10/12；狭缝召回66.92%、28.46%，只在1%档达到检出门槛。',
'TV：oracle共同档k=1只有2/4案例达到MSE门槛；案例1、3 MSE低于NN，案例2、4约NN的2.5倍。',
'TV结论：虚警率对齐后检出不如NN，其MSE优势是靠抑制缺陷换来的；1%/0.5%上限下为8/12、7/12，狭缝均未检出。',
'TV k=10：强平滑导致全部12个缺陷漏检，零虚警不能解释为优于其他方法。',
f"多尺度：250/350 kHz两档、15步低频加20步全频；按平均MSE选出的{names[best]}，MSE门槛通过{r['mse_pass_count'][best]}/4。",
f"多尺度最佳档：1%工作点{wp(best,'0.01')}；0.5%工作点{wp(best,'0.005')}。",
'局限：只有四个二维标量合成案例；正则权重、截止频率与检测阈值均由同组真值回顾性选择，未做独立测试。',
'局限：检出只统计覆盖比例，宽块伪影可能命中狭缝；合并虚警上限不保证每个案例局部虚警达标。',
'局限：只评价所定35步和两档频带，保留作者代价/梯度实现；不能据此否定所有多尺度方案或断言唯一失败机制。',
('可以跟老师说：在固定35步、四个合成案例上，TV主要让图更干净却漏掉缺陷，本轮低频先行也未达到接近NN的门槛；NN的检出优势仍需与裂缝形状准确性分开看。' if r['mse_pass_count'][best]<3 else '可以跟老师说：在固定35步、四个合成案例上，本轮低频先行达到预定MSE接近门槛，但还必须结合检出与虚警判断是否真正保留小缺陷。'),
'本周到此停止；是否做加源实验，由负责人看完小结后决定。']
assert len(weekly)<=20
lines+=weekly
p=ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md'
with p.open('a',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print('\n'.join(weekly))
print('APPENDED',len(lines),'lines; weekly',len(weekly),'lines')
