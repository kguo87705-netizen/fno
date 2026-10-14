"""Append interpretation and measured future budget; no new experiments."""
from common import *
ROUND=ROOT/'.build/fwi-quality-guards-round2-20261013'
note=ROOT/'PINO超声成像知识库/05实践与复习/T2 独立测试探索轮 2026-10-13.md'
r=json.loads((ROUND/'summary.json').read_text(encoding='utf-8'))
s=json.loads((ROUND/'status.json').read_text(encoding='utf-8'))
seconds_per_job=s['elapsed_seconds']/255
workers=np.array([x['wall_seconds'] for x in s['finished']]);overhead=seconds_per_job-float(workers.mean())
p90=float(np.quantile(workers,.9))+overhead
budget=dict(measured_end_to_end_seconds_per_inversion=seconds_per_job,p90_worker_plus_mean_overhead=p90,
 remaining_defective_boards=44,reuse_completed_16_plus_one_blank=True,
 full_ABC=dict(inversions=660,mean_hours=660*seconds_per_job/3600,p90_hours=660*p90/3600),
 primary_AC_only=dict(inversions=440,mean_hours=440*seconds_per_job/3600,p90_hours=440*p90/3600),
 example_ten_new_blank_conditions=dict(inversions=150,mean_hours=150*seconds_per_job/3600),
 status='cost estimate only; no next round authorized or started')
dump(ROUND/'next_round_budget.json',budget)
text=note.read_text(encoding='utf-8')
heading='#### 分组Wilson 95%区间（按种子分别报）'
assert text.count(heading)==1
interpretation='''### 主指标解读（探索性）
A(随机原生epoch0)五种子均检出64/64；B单椭圆200为61.0[60,62]/64，C扩展200为60.8[59,63]/64。A比B、C的每种子总检出数都更多；C对B按种子总检出数为胜3/负2/平0，均值却低0.2个，说明少数种子的差距大小会影响均值，不能称扩展数据稳定胜出。
C相对A：小孔平均下降3.75pp、下降方向4/5；狭缝平均下降8.33pp、下降方向5/5。两项效应量均未达预登记10pp，**未达到探索性数值门槛**；这不反证H2，也不能据此正式支持H2。只比较epoch0/100，未做epoch10/30，不声称单调强度曲线。
地板规则用预训练epoch0的A组完成35步反演后的狭缝检出率（100%），不是A的反演第0步；本轮不触发地板效应。
白话：预训练的最终平均图像误差更低，但还少找出几个缺陷；这批样本里差距没有达到事先规定的幅度。

'''
text=text.replace(heading,interpretation+heading)
text+='''
## 收尾问题与解释边界
数值评分已保存后，analyze_round2.py的笔记防重复断言误把本笔记保存的历史diff里同名标题当成已有结果章节，因此停止写笔记；反演与评分JSON未失败、未重算。保留该脚本原样，使用已计划的新format_round2_report.py按真实章节起始位置写出结果。另一次查看JSON的命令未指定UTF-8，Windows默认GBK解码报错；改为显式UTF-8后成功。两者均为报告读写问题，不改变数值、样本、阈值、判据或反演结果。分析脚本仍只有披露的两次历史修改。
主次指标方向一致：C对A的小孔像素召回低8.11pp、狭缝低6.30pp；只作描述，不替代主指标。
轨迹显示B/C分别从43.6/36.2个初始检出增加到61.0/60.8个，平均找回17.4/24.6个初值漏检，仍有3.0/3.2个在所有保存时点漏检，平均最终丢失0个。**本设置下“初值漏检全被锁死”不成立**；这一观察也不证明所有小缺陷都能被恢复。
最强替代解释：C的冻结阈值更保守，实际合并虚警远低于A；方法处于同一验证虚警上限下，但不是相同的测试实际虚警。离散与几何分布差异也存在，因此不能把检出差值全部归因于先验。阈值没有为追求结论而再调。
FNO先验路线的含义：目前没有达到H2预设幅度，也没有证明FNO先验保质；后续若完整测试支持先验造成漏检，应单独验证残差或物理精修能否恢复缺陷。FNO正演代理仍需独立质量与梯度门，R2不因本轮计时而获得资格。本轮未启动这些工作。
'''
text+=f'''
## 下一轮实测预算（仅估算，待负责人决定）
本轮255次串行端到端平均{seconds_per_job:.2f}s/次（含进程启动），P90工作耗时加平均启动开销约{p90:.2f}s/次；主反演总计{s['elapsed_seconds']/3600:.3f}小时。T4独立2.37分钟。
| 范围 | 新增反演次数 | 均值估算 | P90估算 |
|---|---:|---:|---:|
| 补齐其余44板，A/B/C五种子 | 660 | {budget['full_ABC']['mean_hours']:.2f}h | {budget['full_ABC']['p90_hours']:.2f}h |
| 若只补H2主对照A/C五种子 | 440 | {budget['primary_AC_only']['mean_hours']:.2f}h | {budget['primary_AC_only']['p90_hours']:.2f}h |
已有16板与1空板可复用；不重新生成已有细网格观测，不重训。ABC可考虑分两批22板，每批约{330*seconds_per_job/3600:.2f}h（P90约{330*p90/3600:.2f}h）；只是预算，不启动。
正式测试前仍需负责人决定独立空板的含义与数量。10个新的独立条件若获批准，三组五种子评估约{150*seconds_per_job/3600:.2f}h，观测约1分钟另加预处理；重复现有同一无噪声空板不能增加独立信息。本轮没有生成。
本周已明确的T3、梯度门、FNO训练和新增支线继续停止；未提交或推送GitHub。已停止。
'''
note.write_text(text,encoding='utf-8')
unified=ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md'
with unified.open('a',encoding='utf-8') as f:
 f.write(f'\n\n## T2独立测试探索轮完成（2026-10-14凌晨汇总）\n255/255次、五种子、16有缺陷板+1空板，耗时{s["elapsed_seconds"]/3600:.3f}小时；各类型8缺陷，总64。A/B/C检出均值64.0/61.0/60.8，范围64–64/60–62/59–63。C相对A小孔下降3.75pp、狭缝下降8.33pp，未达10pp预设幅度；所有结论探索性，H2仍不正式判定。各分组Wilson、配对、实际虚警、MSE、轨迹和像素召回见[[T2 独立测试探索轮 2026-10-13]]。T4补测与局限见[[T4 物理与R2尺度计时 2026-10-13]]。未启动下一轮、T3或梯度门，未推送GitHub。已停止。\n')
print(json.dumps(budget,ensure_ascii=False),flush=True)
