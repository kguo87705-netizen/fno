"""Freeze one no-training audit after partial geometry repair and material loss."""
import json
from pathlib import Path
from run_depth_coverage_storage_retry import OUT
from run_frozen_depth_shift import WIDE,BASE,ROOT
from run_reduced_material_scan import sha
r=json.loads((OUT/'summary.json').read_text());assert r['decision']=='coverage_improves_but_fails_screens'
review='''### 三个竞争解释与反证

| 解释 | 支持证据 | 冲突/未知 | 若成立应观察到；反证 |
| --- | --- | --- | --- |
| H1 单对比度几何覆盖训练改变了原材料映射，遗忘不仅是固定偏置 | 原名义a=.3/.7接收3.83%/5.67%→覆盖后14.17%/34.07%；缓存材料有限响应误差也升高 | 重复臂也退化到8.10%/14.04%；无法把全部变化归给几何覆盖，且局部材料Jacobian未知 | 若几何训练梯度与材料锚点梯度相反，支持训练方向的局部冲突；若一致则削弱这个具体机制。原结果仍不证明材料锚点重放必然解决 |
| H2 接收拟合与全场拟合的参数方向存在冲突或不平衡 | 训练几何±2接收约8%，散射场仍约47%；开发场31%，接收8% | 最后30步平均field loss=.01953、receiver loss=.001033，不能凭标量损失说“接收损失更大/主导”；尚无梯度证据 | 冻结状态若field/receiver梯度反向或合成方向不利于field，支持局部欧氏梯度冲突；若各状态方向一致且场误差广泛，就转向表示/优化，而非直接调权重 |
| H3 原300步覆盖只部分优化了全场，不能据此断言容量不足 | 训练全场尚差，开发场改善较小，但接收及几何响应已明显改善 | 三几何field梯度是否相互抵消、总梯度是否仍有信号未知；更多轮数尚未执行 | 若场梯度方向一致且较强，有限优化预算解释仍合理但需后续控制；若相互抵消或局部平坦，也不能直接证明不可约容量，需区分损失与表示 |

与上一轮相比：几何覆盖的作用获得配对支持，单纯多训练名义几何未修复；但按既定整体筛查只能叫部分改善，因为全场不通过。材料宽区间响应变差削弱“只是加了共同固定偏置”的解释。材料方向局部Jacobian、网络自身AD正确与物理导数正确仍未知，不能混为一次审查。

### 已有缓存能回答的材料问题

R/cached_nominal_material_response.json以已有名义a=.3/.5/.7的保存接收波形计算，无新增预测/训练。宽区间.3→.5、.5→.7、.3→.7的有限响应相对误差分别为：原网络7.51%/9.94%/4.62%，名义重复18.20%/16.19%/15.68%，几何覆盖34.03%/30.14%/30.30%。三个区间均保留，不择优报告；这是有限材料变化，不是局部材料Jacobian或未见深度材料敏感度。下一步不重复生成这些标签。

### 继续与修改计划的反方审查

继续几何训练：本轮训练几何全场还没拟合好，重复加轮可能延续材料遗忘，而且会把当前固定300步终点变成开发反馈优化。不能因接收和几何响应通过就跳过场误差或材料评价；线性在开发散射场约6.3%/6.5%、接收约6.6%，当前覆盖网络在两空间都未超越它，几何响应16.88%也略差于线性14.84%。尚无神经质量/总成本优势。

修改为固定状态梯度审查：不改变任何模型、数据或参数，不做优化；只新增诊断测量。先看训练目标方向，而不是同时加材料重放、换权重、改网络。使用旧训练材料锚点标签作为明确诊断信息，不参与验证真值停止或训练。欧氏参数梯度冲突只是局部一阶解释，Adam预条件和历史动量可能改变实际更新，不能将角度直接当作已证明的训练因果。原材料有限响应已可从缓存回答；全场与接收/材料锚点的参数梯度还没有保存，才需新增反向计算。

### 最多三个候选比较

| 候选 | 区分的解释；固定/唯一变化 | 简单参照 | 成本 | 成功/失败/无法判定及触发 |
| --- | --- | --- | --- | --- |
| A 冻结损失/参数梯度审查（唯一优先） | H1/H2/H3局部优化方向；原网络与覆盖final300冻结，只测五旧训练状态的目标方向 | 原网络、单损失梯度、现有线性与零散射误差参照 | 10前向、最多20反向；0优化/0物理新标签，预计<1分钟，上限2分钟 | 完整有限梯度与区域误差可核验即信息有效；几何/材料方向冲突优先重放控制，field/receiver冲突优先损失控制。无冲突不能叫失败，应转向优化/表示。梯度为零/非有限/哈希不符则无法判相关角度 |
| B 同初权重配对材料锚点重放 | 唯一改变两个训练槽的对比度，几何曝光匹配 | 重复a=.5槽及线性 | 至少两臂数百更新，尚未预注册预算 | 材料响应/场同时改善才支持收益；会新增训练，先用A确定必要性 |
| C 物理局部材料差分审查 | 检验真正材料导数；固定几何和checkpoint，只改微小对比度 | 原网络与物理双epsilon差分 | 多个物理和网络预测，无训练 | 有助导数判断，但当前全场及宽区间材料映射已退化，先解释训练方向；本轮不执行 |

### 唯一优先A具体协议（已保存未运行）

固定原width8与覆盖final300两checkpoints；各评估五旧训练状态：a=.5深度−2/0/+2，加名义几何a=.3/.7。固定IEEE、输入、源、损失和原归一化。a=.3/.7在本轮只是诊断锚点，±1开发数据不进入梯度计算。

对每状态同一前向输出计算原field loss和receiver loss，并分别对网络参数求梯度。报告两者范数、范数比、实内积余弦、合成梯度对field损失的欧氏一阶方向；复杂权重使用实部/虚部对应的实欧氏内积，不能丢弃虚部。梯度无定义或极小范数时报告未知，不伪造余弦。

按三几何等权汇总完整目标梯度，按两材料锚点等权汇总，再报二者余弦及几何负梯度对材料目标的一阶方向；不据结果改样本权重。区域误差固定使用已有全场区划：opposite_boundary y[0:2]、interior[2:126]、array_side[126:128]，分别报告误差能量/参考能量和全误差占比，核查三区之和。区域分析不能称有限孔径是根因。

两checkpoint所有参数前后哈希必须不变；无optimizer、0实际参数更新、无标签生成/下载。最多10前向＋20反向，2分钟预算，OOM/非有限停止，保留部分。结果成功仅支持冻结状态的训练目标方向与空间分布解释，不证明材料恢复、物理导数或梯度干预的长期效果。

### 什么新证据会改变建议

若发现恢复初权重/Adam/数据或终态回读不一致，撤回配对判断先修复。若梯度审查显示材料锚点与几何方向冲突，再预注册几何曝光匹配的材料重放控制；若field/receiver方向冲突，先隔离损失；若二者均无冲突且场梯度有一致信号，再考虑固定额外优化预算而非加结构。无论哪种，局部材料Jacobian及真实恢复必须另行验证；不能把A的方向角度当作实际Adam更新效果。
'''
plan={'status':'planned_not_run','name_cn':'冻结检查点的损失、参数梯度与材料锚点冲突审查',
 'output_directory':'.build/fno-frozen-loss-gradient-audit-20261009',
 'bottleneck_cn':'**当前瓶颈：几何覆盖在固定300步下改善了接收波形与有限几何响应，但全场未达筛查，原材料映射明显退化，尚无可靠代理或神经收益证据。** 对这组配对结果置信度高；训练目标冲突、优化或表示机制仍未知。',
 'navigation_cn':'覆盖final300接收约8.04%/8.26%、几何响应16.88%，但场约31%未过旧筛查；名义材料有限响应退化。下一步只做冻结损失/参数梯度审查，未运行；不加网络/轮数。所有新文件实际存D盘。',
 'models':['original width8','depth_coverage final300'],
 'states':['a=.5 depth-2','a=.5 depth0','a=.5 depth2','nominal a=.3','nominal a=.7'],
 'fixed':'IEEE,old field/trace scales,field+receiver losses,input encoding,source0; checkpoint parameters immutable.',
 'measurement':['field/receiver individual parameter gradient norms and real Euclidean cosine','summed objective gradient direction relative to field objective','equal-mean geometry total gradients versus equal-mean material-anchor gradients','fixed three-region full-field residual energies and error fractions'],
 'complex_gradients':'Flatten real and imaginary components as real coordinates; use full real Euclidean inner products.',
 'gradient_zero_rule':'If either Euclidean norm <=1e-12 or nonfinite, cosine/directional interpretation unknown; preserve norms rather than inventing sign.',
 'interpretation':'Negative inner products support local raw-gradient competition, not actual Adam trajectory or proven cause. No conflict does not establish capacity failure. Do not tune weights from this diagnosis.',
 'budget':'10 forward,at most20 backwards,0 optimizer updates,0 physical generation;expected<1minute,cap2min;no training/download,stop OOM/nonfinite.',
 'verification':'Checkpoint/source SHA before/after unchanged; finite gradients; recompute inner-product summaries from saved real-coordinate sufficient sums; region energy partition exact within declared float64 accumulation tolerance. Scope excludes physical material derivatives.',
 'review_markdown_cn':review,
 'frozen_sha256':{str(p):sha(p) for p in [OUT/'protocol.json',OUT/'raw.pt',OUT/'summary.json',OUT/'verification.json',OUT/'cached_nominal_material_response.json',OUT/'depth_coverage_update300.pt',WIDE/'final.pt',BASE/'data_manifest.json',ROOT/'run_receiver_loss_control.py',ROOT/'time_feature_input.py',ROOT/'basic_fno.py']}}
assert not (OUT/'next_experiment_protocol.json').exists();(OUT/'next_experiment_protocol.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
print('Next audit protocol saved, not run')
