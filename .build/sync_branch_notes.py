from pathlib import Path
v=Path('PINO超声成像知识库')
new='基础DeepONet分支冻结与时间基诊断 2026-10-08'
p=v/'00导航/FNO与超声全波形反演科研计划 2026-10-07.md'
s=p.read_text(encoding='utf-8')
s=s.replace('status: DeepONet时间特征及4000次预算仍未拟合，下一步branch冻结优化对照','status: 分支冻结仍未有效拟合，下一步多频时间表示受控对照')
s=s.replace('下一步只改变branch是否在初值冻结，配对检查联合优化，不增加工件或同时更改表示。','分支冻结对照后续已执行，见下条。')
anchor='- 补充调研：'
s=s.replace(anchor, f'- 后续已完成：[[{new}]]，联合／冻结初始branch各4000次，同初值与点序列。散射接收98.07092%／96.32091%，场100.40794%／100.00832%，仍未有效拟合。仅训练真值逐接收拟合的时间Fourier基参照误差3.09%，不代表网络精度；下一步131输入trunk的零／多频特征配对，两组均冻结初始branch。\n'+anchor,1)
s=s.replace('下一步冻结branch优化诊断；',f'分支冻结仍未有效拟合，见 [[{new}]]，下一步多频时间表示受控对照；')
s=s.replace('- [ ] 固定sincos七通道模型、训练000／源0、相同初值与点序列、4000次预算，配对联合训练／冻结初始branch，仅改变branch是否训练；先检查优化耦合，不作材料泛化或物理反演声明。',f'- [x] 完成七通道sincos联合／冻结初始branch配对各4000次，接收98.07%／96.32%，仍未拟合；冻结权重完整性、优化器参数选择与重载检查通过，见 [[{new}]]。\n- [ ] 执行预先保存的多频时间表示对照：两组均131输入、冻结相同初始branch，126零特征／k=1..63的sin-cos特征；同训练000／源0、初值／点序列／尺度／4000次预算，固定最终轮，不读验证或旧测试。')
p.write_text(s,encoding='utf-8')
p=v/'00导航/开始学习.md'; s=p.read_text(encoding='utf-8');s=s.replace('下一步branch冻结优化诊断。',f'分支冻结结果与时间基诊断见 [[{new}]]，仍未有效拟合；下一步多频时间特征同预算对照。');p.write_text(s,encoding='utf-8')
p=v/'05实践与复习/基础DeepONet时间特征与预算对照 2026-10-08.md';s=p.read_text(encoding='utf-8');s=s.replace('协议已保存到预算结果目录`next_branch_freeze_protocol.json`，尚未运行。',f'当时协议保存到预算结果目录`next_branch_freeze_protocol.json`；后续已完成，见 [[{new}]]。历史协议的planned_not_run字段保留，执行证据另存，不覆盖原记录。');p.write_text(s,encoding='utf-8')
p=Path('Research_Direction/Author_Baseline_Singh_Zenodo14616677/README_local.md');s=p.read_text(encoding='utf-8');s=s.replace('下一步`.build/basic-deeponet-budget-control-20261008/next_branch_freeze_protocol.json`，planned_not_run：','当时下一步（后续已执行，见下节）`.build/basic-deeponet-budget-control-20261008/next_branch_freeze_protocol.json`，历史planned_not_run保留：')
s+='''

## DeepONet分支冻结与训练时间基诊断

`run_deeponet_branch_control.py --data .build/operator-pilot-v3-20261008 --out .build/basic-deeponet-branch-control-20261008/joint --arm joint --plan .build/basic-deeponet-budget-control-20261008/next_branch_freeze_protocol.json`执行七通道sincos从同初值重新训练4000次；冻结组将arm及out子目录改为`frozen_initial_branch`。两组seed2/19、抽样点、旧尺度、损失与预算匹配；冻结组仅trunk和bias进入Adam，branch全部保持初值。最终散射场100.40794%／100.00832%，接收98.07092%／96.32091%。小幅接收改善仍不构成有效拟合。

`verify_deeponet_branch_control.py --data ... --run 相应组目录`检查完整网格重载、全部4000步抽样SHA、Adam步数、作者文件及冻结branch逐值不变与优化器参数选择，两组通过。`audit_deeponet_branch_state.py --run ...`审计监督优化状态，不能替代物理梯度；冻结branch的权重梯度为零是设定结果，材料输入梯度仍可穿过冻结函数。

`audit_receiver_temporal_basis.py`仅用训练000/source0，对每个接收器使用整条训练真值独立求最小二乘系数：单载波4列／16局部载波窗口34列／时间Fourier常数加63对sin-cos共127列，500点拟合后重建2000点误差98.2906%／63.5899%／3.09159%。最小二乘正规方程及Fourier与截断FFT数值一致性通过。它是训练内线性时间基解析参照，不是预测、材料算子或非线性DeepONet能力下界；稀疏真值重建误差仅0.05210%。结果`.build/receiver-temporal-basis-20261008`。

`summarize_deeponet_branch_control.py`保存对照汇总与图像，已实际视觉核查并同步Obsidian。下一协议`.build/basic-deeponet-branch-control-20261008/next_multifrequency_protocol.json`仍planned_not_run：131输入匹配trunk，两组均冻结相同初始branch，126零特征／k=1..63时间sin-cos特征，4000次固定预算与同点序列。新初值配对，旧七通道不是因果对照。本轮不读取验证或旧测试，不接入代理FWI。详见知识库“基础DeepONet分支冻结与时间基诊断 2026-10-08”。
'''
p.write_text(s,encoding='utf-8')
print('Updated plan, home, prior note and local README.')
