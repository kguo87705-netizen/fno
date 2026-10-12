"""Save verified paired results to the established local Obsidian vault."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--vault',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.run/'comparison.json').read_text());m=json.loads((a.run/'material_response.json').read_text())['models'];use=json.loads((a.run/'feature_use.json').read_text());verification=json.loads((a.run/'artifact_verification.json').read_text());assert verification['status']=='passed'
    note=a.vault/'05实践与复习/普通FNO时间Fourier输入同预算对照 2026-10-08.md';s=note.read_text(encoding='utf-8')
    s=s.replace('status: 两组固定20轮训练执行中，结果待冻结','status: 两组固定20轮完成，材料响应审计已保存')
    v=r['final_validation'];tr=r['final_training'];percent=lambda x:f'{100*x:.3f}%'
    results='## 固定最终轮结果\n\n两组各完成 20 轮、960 次更新，主要结果为预定最终轮。下表为全部 24 次验证发射合并能量的指标。\n\n| 验证指标 | 八通道零特征 | 八通道 sin/cos |\n| --- | ---: | ---: |\n'
    for label,key in [('散射波场相对 L2','scatter_field_relative_l2'),('散射接收相对 L2','scatter_trace_relative_l2'),('完整波场相对 L2','full_field_relative_l2'),('完整接收相对 L2','full_trace_relative_l2')]:results+=f'| {label} | {percent(v[0][key])} | {percent(v[1][key])} |\n'
    results+=f"\n验证零散射参照：散射场/接收均为 100%；完整场/接收为 {percent(v[0]['homogeneous_field_relative_l2'])}/{percent(v[0]['homogeneous_trace_relative_l2'])}。训练散射场/接收：零特征 {percent(tr[0]['scatter_field_relative_l2'])}/{percent(tr[0]['scatter_trace_relative_l2'])}，sin/cos {percent(tr[1]['scatter_field_relative_l2'])}/{percent(tr[1]['scatter_trace_relative_l2'])}。\n\n"
    results+='| 验证材料差分指标 | 零特征 | sin/cos |\n| --- | ---: | ---: |\n'
    for label,key,fmt in [('差分响应相对 L2（零响应=1）','secant_response_relative_l2',lambda x:f'{x:.6f}'),('预测/参考差分范数','response_norm_ratio',percent),('差分余弦','response_cosine',lambda x:f'{x:.6f}')]:results+=f'| {label} | {fmt(m[0][key])} | {fmt(m[1][key])} |\n'
    results+=f"\n冻结 sin/cos 模型再移除两条特征，输出变化范数仅为参考散射波场的 {percent(use['feature_removal_field_change_over_true_scatter'])}、散射接收的 {percent(use['feature_removal_trace_change_over_true_scatter'])}；移除后散射接收误差为 {percent(use['zeroed_scatter_trace_relative_l2'])}。这项诊断不替代独立训练的零特征组。\n\n"
    results+='![[普通FNO时间Fourier输入同预算对照.png]]\n\n图中两组场误差使用共同色阶；误差曲线接近 1 的变化不可误读为大幅收益。单个波形仅作示例，结论取全部验证发射。\n\n'
    if max(x['response_norm_ratio'] for x in m)<.1 and min(x['scatter_trace_relative_l2'] for x in v)>.95:
        results+='当前输入特征在单种子固定预算下没有带来实质性的散射学习改善，材料响应仍很弱。训练散射也拟合不足，不能简单归结为验证泛化失败。两组的小数值差异没有重复种子或置信区间支撑，不能宣称可靠优劣；本结果也不证明时间 Fourier 特征普遍无效，更不证明或排除频率偏置。暂不开展代理 FWI。\n\n'
    else:results+='本轮结果仅回答固定预算下的输入特征作用；是否具有稳定收益仍需独立重复。材料差分不是局部梯度，继续定位材料条件映射，暂不由本轮直接进入代理 FWI。\n\n'
    spectra=json.loads((a.run/'scatter_target_spectrum.json').read_text())['summary']
    results+='## 补充：散射目标频谱覆盖\n\n此前完整场的覆盖率不能直接套用到散射残差，本轮重新计算了全部训练/验证目标的全 FFT 矩形掩膜能量。\n\n| 散射目标能量覆盖 | 训练48发射 | 验证24发射 |\n| --- | ---: | ---: |\n'
    for label,key in [('时间频谱范围','temporal_retained_energy_fraction'),('联合x/y/t矩形范围','joint_retained_energy_fraction')]:results+=f"| {label} | {percent(spectra['train'][key])} | {percent(spectra['validation'][key])} |\n"
    results+='\n掩膜为 x[-16,15]、y[-8,7]、t[-63,63]；它不是谱层精确 Hermitian 输出，局部支路和非线性还可产生额外频率，不能用来给出严格 FNO 误差下界。主要时间能量仍在范围内，未见仅靠缺失载频即可解释约100%散射误差的证据。特征移除确实改变了输出，材料差分仍很弱；不能把特征利用等同于有效材料条件映射。\n\n'
    results+='## 下一步：同源双工件材料条件诊断\n\n把问题缩到训练工件 000、001 的同一发射 0，已核查除 gamma 外的输入完全相同，完整接收差异为平均波形范数的 33.885%。两组仍用八通道零特征普通 FNO：一组输入真实 gamma，另一组将 gamma 换成全 1，但保留各自不同的真值。两组同初值、交替样本顺序、旧尺度、接收权重 1、Adam lr=0.001，各固定 800 次更新，每个工件各暴露 400 次。不读取验证或测试。\n\n增加两工件散射真值的算术平均参照：在其他输入相同、等权二次损失下，它是材料无关预测的最优解。该参照用已拟合训练真值计算，是诊断下界，不能部署为未知工件预测器。真实材料组能否降到平均参照以下，以及两材料预测差分是否匹配参考差分，是主要判据。已算得平均参照监督损失0.0636916、零散射损失0.138478，前者为后者45.994%；这说明共同波形也能降低损失，下降本身不能证明学到材料区分。\n\n这一步是条件映射可学性诊断，不能将与旧 12 工件实验的差异单独归因于工件数，也不是泛化验收。协议已保存为 `next_material_conditioning_protocol.json`，标为 planned_not_run；本轮仅完成了输入一致性、真值差异和平均参照核查，没有执行新训练。优先把基础映射做清楚，再决定扩大预算、调整容量或引入复杂结构。\n\n'
    results+='## 验证与限制\n\n最终两组检查点重新加载后，全部 24 次验证发射指标复算一致；960 次更新、20 轮排列、同初值与旧归一化通过检查。已知零散射和线性时间信号的还原/能量检查、输入频率/单位圆与材料自动微分连接、差分能量恒等式、七个作者原文件及数据/基底哈希通过。旧结果文件哈希另行核查；图像已实际查看，附件副本及新增笔记链接已检查。没有新的测试成绩、物理梯度、实测或反演验证。\n\n'
    marker='## 可复现文件';assert s.count(marker)==1 and '## 固定最终轮结果' not in s;s=s.replace(marker,results+marker);note.write_text(s,encoding='utf-8')
    image=a.vault/'90附件/普通FNO时间Fourier输入同预算对照.png';shutil.copyfile(a.run/'time_feature_comparison.png',image)
    prior=a.vault/'05实践与复习/普通FNO散射目标多工件验证 2026-10-08.md';s=prior.read_text(encoding='utf-8').replace('上述时间输入对照现已启动','上述时间输入对照现已完成');prior.write_text(s,encoding='utf-8')
    plan=a.vault/'00导航/FNO与超声全波形反演科研计划 2026-10-07.md';s=plan.read_text(encoding='utf-8')
    s=s.replace('status: 散射多工件验证完成，材料响应不足，下一步时间Fourier输入对照','status: 时间Fourier同预算对照完成，下一步双工件材料条件诊断')
    anchor='- 尚未完成：更多状态';assert anchor in s
    s=s.replace(anchor,f"- 后续已完成：[[普通FNO时间Fourier输入同预算对照 2026-10-08]]，两组八通道各 20 轮。验证散射接收 {percent(v[0]['scatter_trace_relative_l2'])}/{percent(v[1]['scatter_trace_relative_l2'])}，材料差分范数 {percent(m[0]['response_norm_ratio'])}/{percent(m[1]['response_norm_ratio'])}；下一步同源双工件真实/置空材料输入诊断。\n"+anchor)
    s=s.replace('- [ ] 两组八通道普通 FNO','- [x] 两组八通道普通 FNO')
    s=s.replace('- [ ] 继续核查材料扰动响应','- [ ] 固定两个训练工件同一发射，真实 gamma/全 1 gamma 两组各 800 次更新，与两真值平均参照比较材料区分能力；仅训练诊断，协议已保存。\n- [ ] 继续核查材料扰动响应')
    s=s.replace('下一步时间 Fourier 输入单因素对照；物理梯度与完整代理反演尚未验收','八通道时间 Fourier 输入同预算对照已完成，见 [[普通FNO时间Fourier输入同预算对照 2026-10-08]]；下一步双工件材料条件诊断，物理梯度与完整代理反演尚未验收')
    plan.write_text(s,encoding='utf-8')
    readme=Path(__file__).parent/'README_local.md';s=readme.read_text(encoding='utf-8').replace('下一步计划（未运行）：八通道普通FNO','当时预先计划（后续已执行）：八通道普通FNO')
    s+=f"\n\n## 时间 Fourier 输入同预算对照\n\n`train_time_feature_control.py` 在八通道普通 FNO 上配对零特征与固定 0.5MHz sin/cos 特征，初始权重哈希相同，原 20 轮顺序相同；两组各 960 次更新，散射目标、旧尺度与接收权重1保持不变。输出 `.build/basic-fno-time-feature-control-20261008`。最终验证散射接收 {percent(v[0]['scatter_trace_relative_l2'])}/{percent(v[1]['scatter_trace_relative_l2'])}，材料差分范数 {percent(m[0]['response_norm_ratio'])}/{percent(m[1]['response_norm_ratio'])}。\n\n`audit_validation_material_response.py` 按检查点配置和特征模式加载；原六通道默认兼容。`audit_time_feature_use.py` 仅冻结实验模型后移除特征，不能代替两组训练对照。`summarize_time_feature_control.py` 保存图和汇总，`verify_time_feature_control.py` 重新加载两组并复算全部验证发射。复现命令及详细结论见知识库“普通FNO时间Fourier输入同预算对照 2026-10-08”。\n\n下一步计划 `next_material_conditioning_protocol.json`：两个训练工件同一源，真实 gamma/全1 gamma 输入对照，各800更新，与两个训练真值的平均预测比较；不读取验证/测试。这只是可学性诊断，尚未运行，不作代理FWI或泛化结论。\n"
    readme.write_text(s,encoding='utf-8');print(str(note.resolve()))


if __name__=='__main__':main()
