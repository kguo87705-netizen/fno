"""Summarize saved first-round outputs only; never perform a test inversion."""
from common import *

def append_note(task,title,text):
    p=ROOT/'PINO超声成像知识库/05实践与复习'/f'{task} {title} 2026-10-13.md'
    with p.open('a',encoding='utf-8') as h:h.write('\n\n## 本轮结果\n'+text+'\n')

if __name__=='__main__':
    cfg=json.loads((BUILD/'config.json').read_text())
    state=json.loads((BUILD/'round_status.json').read_text())
    gen=json.loads((BUILD/'generation_times.json').read_text())
    check=json.loads((BUILD/'continuous_geometry_checks.json').read_text())
    rows=json.loads((BUILD/'accepted_geometries.json').read_text())
    # Structural validation: all accepted fields and observations parse, finite, and
    # match the continuous-geometry labels. No test reconstruction is accessed.
    for r in rows:
        d=np.load(data_path(r['id']));truth,lab=raster(r['components'],255,127)
        assert d['measurement'].shape==(4,24,2000)
        assert d['truth'].shape==(256,128) and d['initial_gradient'].shape==(256,128)
        assert np.array_equal(d['truth'],truth[1:-1,1:-1]) and np.array_equal(d['labels'],lab[1:-1,1:-1])
        assert all(np.isfinite(d[k]).all() for k in ['measurement','truth','initial_gradient'])
    t1=f'''**实验验证：**90份几何/观测文件已生成，包含80个独立有缺陷几何，以及1个无缺陷参考的10份重复记录；结构与有限值检查通过。

| 集合 | 有缺陷/空板 | 缺陷总数 | 八类型各自数量 |
|---|---:|---:|---:|
| 验证 | 20/5 | 80 | 10 |
| 测试 | 60/5 | 240 | 30 |

首批观测及粗网格初始梯度耗时{gen['wall_seconds']/60:.2f}分钟。连续距离复核发现3个旧候选不满足间距（T1第一次检查失败）；已保留原元数据/波形，单次修正并重生成通过。采样边界复核最小距离：边界{1000*check['minima_m'][0]:.4f}mm、源{1000*check['minima_m'][1]:.4f}mm、缺陷间{1000*check['minima_m'][2]:.4f}mm。修正案例用保守外接圆条件保证连续间距。

最终几何入口：`accepted_geometries.json`；波形入口由`geometry_data_overrides.json`解析到`data/`或`corrected_data/`。测试集反演次数为0。

局限：仍是同类二维标量求解器的细/粗网格试验，不证明真实采集泛化；历史训练观测510×254与当前511×255也存在离散差异。固定采集且无噪声时10个空板文件的观测/初始梯度完全相同，仅有1个独立空板几何；不能按10份独立样本推断置信度。保留预先约定的5+5文件权重，不用看到结果后改校准口径。下一步只能在冻结阈值之后、经批准评价测试反演。
'''
    append_note('T1','独立几何与细网格观测',t1)
    retrain=[json.loads(p.read_text()) for p in sorted((BUILD/'checkpoints').glob('*/reproduction.json'))]
    text='| 初始化种子 | 100轮训练秒数 | 与旧epoch100逐张量一致 |\n|---|---:|---|\n'
    for r in retrain:text+=f"| {r['seed']} | {r['seconds']:.2f} | {'是' if r['exact_match'] else '否'} |\n"
    feasibility=json.loads((BUILD/'feasibility.json').read_text()) if (BUILD/'feasibility.json').exists() else None
    if feasibility:
        text+='\n### 原生epoch0可行性关\n\n$D=1-J_{35}/J_0$，每种子先对固定四几何平均，再判断是否达到50%。\n\n| 种子 | 平均失配下降 |\n|---|---:|\n'
        for r in feasibility['per_seed']:text+=f"| {r['seed']} | {100*r['mean_reduction']:.2f}% |\n"
        text+=f"\n**实验验证：**可行性关{'通过' if feasibility['passed'] else '未通过'}，{feasibility['seed_passes']}/5种子达标。这只检验35步优化是否能动，不验证H2。\n"
    runtime=json.loads((BUILD/'runtime_probe.json').read_text())
    text+=f"\n串行35步端到端{runtime['serial']['wall_seconds']:.2f}s，优化{runtime['serial']['fit_seconds']:.2f}s；平均GPU利用率{runtime['mean_gpu_percent']:.2f}%，设备显存峰值{runtime['peak_device_memory_mib']:.0f}MiB。并发未启动：单进程分配峰值{runtime['serial']['peak_allocated_bytes']/2**30:.2f}GiB，两进程超出12GB显存，未声称完成并发等价性检查。\n"
    frozen=json.loads((BUILD/'frozen_thresholds.json').read_text()) if (BUILD/'frozen_thresholds.json').exists() else None
    if frozen:
        text+='\n### 验证集阈值（seed30，已冻结）\n\n| 初始化 | 虚警上限 | 阈值 | 验证MSE均值 | 检出/80 | 验证虚警 | 空板虚警 |\n|---|---:|---:|---:|---:|---:|---:|\n'
        groupkeys=['0','B','100'] if 'B' in frozen['results'] else ['0','100']
        groupnames={'0':'A原生epoch0','B':'B单椭圆200','100':'C扩展200 epoch100'}
        for e in groupkeys:
            for cap in ['0.01','0.005']:
                r=frozen['results'][e][cap]
                if r is None:text+=f'| {groupnames[e]} | {float(cap):.1%} | 无工作点 | 未测 | 未测 | 未测 | 未测 |\n';continue
                text+=f"| {groupnames[e]} | {float(cap):.1%} | {r['threshold']:.2f} | {r['mean_material_mse']:.6f} | {r['detected']}/80 | {r['false_alarm']:.4%} | {r['blank_false_alarm']:.4%} |\n"
        text+='\n| 初始化（1%档） | 类型 | 检出/数量 | 分组召回 | Wilson95% | 平均缺陷内像素召回 |\n|---|---|---:|---:|---|---:|\n'
        names=['孔2mm','孔3mm','孔4mm','孔6mm','狭缝1格','狭缝2格','狭缝3格','椭圆']
        for e in groupkeys:
            r=frozen['results'][e]['0.01']
            if r is None:continue
            for k,g in sorted(r['groups'].items()):
                lo,hi=g['wilson95'];text+=f"| {groupnames[e]} | {names[int(k)]} | {g['detected']}/{g['total']} | {g['recall']:.1%} | [{lo:.1%}, {hi:.1%}] | {g['mean_pixel_recall']:.1%} |\n"
        text+='\n以上均为选阈值所用验证集的描述结果，不是独立测试证据；不能据此宣布H2通过或失败。逐案例虚警、完整扫描和分组区间均保存在`frozen_thresholds.json`。Wilson区间按缺陷计数，未处理同板相关性。\n'
        text+='\n### 验证集逐板结果（1%上限工作点）\n\n| 验证板 | A MSE | A虚警 | B MSE | B虚警 | C MSE | C虚警 | A/B/C检出数 |\n|---|---:|---:|---:|---:|---:|---:|---|\n'
        folders={'0':BUILD/'calibration/seed30_epoch0','B':BUILD/'calibration_B/seed30_epoch100','100':BUILD/'calibration/seed30_epoch100'}
        for i in range(25):
            case=f'validation_{i:03d}';cells=[];counts=[]
            for key in ['0','B','100']:
                if key not in frozen['results']:cells+=['未测','未测'];counts.append('未测');continue
                m=json.loads((folders[key]/case/'metrics.json').read_text())
                selected=frozen['results'][key]['0.01']
                c=next(r for r in selected['cases'] if r['case']==case) if selected else None
                cells+=[f"{m['material_mse']:.6f}",f"{c['false_alarm']:.4%}" if c else '无工作点']
                counts.append(str(sum(c['detected'])) if c and c['detected'] else ('空板' if c else '未测'))
            text+='| '+case+' | '+' | '.join(cells)+' | '+'/'.join(counts)+' |\n'
        text+='\n### 同一验证几何的配对描述（只含20块有缺陷板，seed30）\n\n| 左组/右组 | 左组胜 | 左组负 | 平 |\n|---|---:|---:|---:|\n'
        for left,right in [('0','B'),('0','100'),('B','100')]:
            if left not in frozen['results'] or right not in frozen['results']:continue
            a=frozen['results'][left]['0.01'];c=frozen['results'][right]['0.01']
            if a is None or c is None:continue
            delta=[sum(x['detected'])-sum(y['detected']) for x,y in zip(a['cases'][:20],c['cases'][:20])]
            text+=f"| {groupnames[left]}/{groupnames[right]} | {sum(d>0 for d in delta)} | {sum(d<0 for d in delta)} | {sum(d==0 for d in delta)} |\n"
        text+='\n这里只描述每板检出数的胜负，不作验证集显著性检验，不代替训练种子4/5方向判据。\n'
    descpath=BUILD/'intermediate_descriptive.json'
    if descpath.exists():
        desc=json.loads(descpath.read_text());text+=f"\n### 中间步（描述性）\n\n以下0/5/10/20/35是反演迭代步，不是预训练epoch轴0/10/30/100。新保存的B/C中间步在共同{len(desc['common_BC_cases'])}块验证板上，用各组第35步阈值评分，不重新选阈值。A未保存中间步，C前两例未保存；不重跑。详见`intermediate_descriptive.json`。\n\n| 组别 | 步数 | MSE | 检出 | 虚警 |\n|---|---:|---:|---:|---:|\n"
        for group,steps in desc['results'].items():
            if 'status' in steps:continue
            for step,r in steps.items():text+=f"| {group} | {step} | {r['mean_material_mse']:.6f} | {r['detected']}/{r['total']} | {r['false_alarm']:.4%} |\n"
        text+='\n固定各自最终阈值不代表中间步实际虚警均≤1%；在保存的检查点之间是否短暂检出未测。端点和保存步转移计数：`'+json.dumps(desc['transitions'],ensure_ascii=False)+'`。\n'
    text+='\nH2状态：**假设，未验证**。A/B/C只检验初始化是否影响漏检，强度下降仅依据epoch轴。首轮无测试集反演，无epoch10/30的反演曲线，无近背景备选，无T3。若后续支持H2，仅说明此U-Net初始化在本设置下有压制缺陷的证据；FNO先验应接受同一质量护栏，不能外推为所有预训练FNO必然无效或必须使用某种修正。对历史四案例比较使用表述：测试案例少、训练种子波动明显，尚不足以可靠判断较小的检出差异。TV仅为实验观察（4案例，回顾性阈值），新测试集尚未复核。\n'
    append_note('T2','H2初始化与可行性',text)
    scale='| 网格 | 测量 | 源批次 | 输出帧 | 中位秒 | 峰值GiB | 状态 |\n|---|---|---:|---:|---:|---:|---|\n'
    for p in sorted((BUILD/'scale').glob('*.json')):
        r=json.loads(p.read_text());kind=r['kind'];sources=1 if kind.startswith('fno') else 4
        scale+=f"| {r['Nx']+1}×{r['Ny']+1} | {kind} | {sources} | {r['output_frames']} | {format(r['median_seconds'],'.4f') if r['status']=='ok' else '未测'} | {format(r['peak_allocated_bytes']/2**30,'.2f') if r['status']=='ok' else '未测'} | {r['status']} |\n"
    software=json.loads((BUILD/'software.json').read_text())
    scale+='\n软件与精度：'+json.dumps(software,ensure_ascii=False)+'；张量FP32/complex64，cuDNN TF32允许，未宣称所有卷积均为严格IEEE FP32。\n'
    scale+='\nFNO反向用输出平方均值标量，只测完整反向图对输入求梯度的开销；未验证真实FWI波形损失梯度，也未执行梯度门。\n'
    scale+='\n**实验验证：**已完成项仅是各自工作量下的计时。物理四源批次、FNO单输入批次，且物理Nt全帧/FNO500帧，不能相除宣称等任务加速。源批次未统一是本轮T4实现缺口。512档两次物理OOM后按规则停止T4；其余512的FNO项及全部1024项未运行。T1细网格只存接收波形，而T4使用作者完整波场存储路径，所以前者可运行不代表后者可容纳。OOM不等于某算法原理上不可扩展。无检查点、无接收点专用结构、无FNO训练。\n\n下一轮如继续T4，需先批准统一源批次口径和全场内存约束下的测量范围；这不是本轮自动续跑项。弹性/三维外推证据等级：有理论依据未验证。\n'
    append_note('T4','物理与R2尺度计时',scale)
    events=json.loads((BUILD/'round_events.json').read_text())
    if (BUILD/'B_events.json').exists():events+=json.loads((BUILD/'B_events.json').read_text())
    measured=[r['seconds'] for r in events if r['name'].startswith(('feasibility_','calibration_'))]+[runtime['serial']['wall_seconds']]
    wall=float(np.mean(measured));p90=float(np.percentile(measured,90))
    second=dict(mean_measured_case_seconds=wall,p90_case_seconds=p90,unique_inversions_measured=len(measured),H2_AC_primary=dict(methods=2,seeds=5,cases=65,inversions=650,hours=650*wall/3600,p90_hours=650*p90/3600),ABC_test=dict(methods=3,seeds=5,cases=65,inversions=975,hours=975*wall/3600,p90_hours=975*p90/3600),suggested_block=dict(defective=16,blank=1,inversions=255,hours=255*wall/3600,p90_hours=255*p90/3600,proposed_budget_hours=7),references_later='author800/conventional/initial+conventional/CNNGamma need their own new-geometry timing; not measured this round; blank duplicates not independent observations')
    dump(BUILD/'second_round_budget.json',second)
    summary=dict(round=state,geometry_validation_passed=True,weights_all_exact=all(r['exact_match'] for r in retrain),training_seconds=sum(r['seconds'] for r in retrain),feasibility=feasibility,frozen_thresholds=bool(frozen),second_round_budget=second,test_inversions=0,H2='unverified',T3=False,FNO_training=False)
    dump(BUILD/'first_round_summary.json',summary)
    record=ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md'
    with record.open('a',encoding='utf-8') as h:
        h.write('\n\n## 2026-10-13 独立评价首轮入口\n负责人已批准新定位及首轮8小时范围，旧“只追加一份文件”由每实验Obsidian笔记要求替代。\n见[[T1 独立几何与细网格观测 2026-10-13]]、[[T2 H2初始化与可行性 2026-10-13]]、[[T4 物理与R2尺度计时 2026-10-13]]。\n测试反演0次，H2未验证；T3未启动；FNO仅随机权重计时，未训练。结果JSON入口`.build/fwi-quality-guards-20261013/first_round_summary.json`。\n')
    print('FIRST_ROUND_SUMMARY',json.dumps({k:v for k,v in summary.items() if k not in ['feasibility','second_round_budget']},ensure_ascii=False),flush=True)
