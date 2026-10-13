"""Read saved metrics and fields only; no training, inversion, or rescoring."""
import time,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import ndimage
from PIL import Image
START=time.perf_counter()
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
OUT=ROOT/'.build/singh-teacher-report-20261013';OUT.mkdir(parents=True,exist_ok=True)
mult=json.loads((ROOT/'.build/singh-multiscale-author-20261012/comparison_scores.json').read_text(encoding='utf-8'))
cov=json.loads((ROOT/'.build/singh-coverage-100epochs-20261013/comparison_scores.json').read_text(encoding='utf-8'))
record=ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md'
header='| 方法 | 案例1整体MSE | 案例2整体MSE | 案例3整体MSE | 案例4整体MSE | 1%虚警下检出数/12 | 狭缝召回 | 案例3虚警率 |'
table=[header,'|---|---|---|---|---|---|---|---|']
for name,key,source in [('常规 FWI（35步）','conventional',mult),('TV k=1（35步）','tv_k1',mult),('多尺度350 kHz（35步）','ms350',mult),('NN-FWI（35步）','nn',mult),('作者800样本模型迁移5步','author800',cov)]:
    vals=source.get('material_mse',{}).get(key)
    p=source.get('working_points',{}).get(key,{}).get('0.01')
    cells=[f'{x:.6f}' for x in vals] if vals is not None else ['未测']*4
    cells+=([f"{p['detected_total']}/12",f"{100*p['thin_slit_recall']:.2f}%",f"{100*p['cases']['3']['false_alarm_fraction']:.4f}%"] if p else ['未测']*3)
    table.append('| '+name+' | '+' | '.join(cells)+' |')
for name in ['对照组200（100轮，5种子均值）','扩展组200（100轮，5种子均值）']:
    table.append('| '+name+' | '+' | '.join(['未测']*7)+' |')
notes=['评分均为内部256×128材料格点，不含ghost cells；方法各自选择合并虚警≤1%的工作点，不保证案例3局部虚警≤1%。',
       '本表只摘录已有comparison_scores.json；未执行新的MSE、阈值扫描或种子均值计算。两个200样本组仅有seed0，不能填写5种子均值。',
       'TV权重、截止频率与检测阈值曾在同组四案例上选择，属于回顾性评价；检出是缺陷内像素覆盖≥50%，不等于形状恢复准确。',
       '现行代码存在未豁免的预训练裁剪/调度及迁移裁剪差异，新增种子复现未启动；这不是科学上的“不稳健”。']
md='## 老师汇报总表（2026-10-13；仅已有记录）\n\n'+'\n'.join(table)+'\n\n'+'\n'.join(notes)+'\n'
(OUT/'summary_table.md').write_text(md,encoding='utf-8')
# Plot only saved post-five-update materials. No detection is recomputed.
fields=[]
for key,folder in [('author800',ROOT/'.build/singh-transfer35-author-20261012/step5'),('control',ROOT/'.build/singh-coverage-100epochs-20261013/control'),('experiment',ROOT/'.build/singh-coverage-100epochs-20261013/experiment')]:
    a=np.load(folder/'case4/final_fields.npz')
    truth=a['truth'][1:-1,1:-1]
    predkey='reconstruction' if key=='author800' else 'post5_reconstruction'
    fields.append(a[predkey][1:-1,1:-1])
    if len(fields)==1:gt=truth.copy()
    else:assert np.array_equal(gt,truth)
assert gt.shape==(256,128)
labels,n=ndimage.label(gt<.5,structure=np.ones((3,3)));assert n==6
hole_indices=[0,2,3,4,5]
dx,dy=100/255,50/127
centers=[]
for idx in hole_indices:
    pixels=np.argwhere(labels==idx+1)
    centers.append((float(pixels[:,0].mean()*dx),float(pixels[:,1].mean()*dy)))
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=[font_manager.FontProperties(fname=str(font)).get_name(),'DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(1,4,figsize=(18,5.6),sharex=True,sharey=True)
fig.subplots_adjust(left=.045,right=.925,bottom=.19,top=.82,wspace=.08)
titles=['真值（✓表示孔存在）','作者800样本模型\n迁移5步，阈值0.45','单椭圆对照 seed0\n200样本、100轮，迁移5步，阈值0.65','扩展组 seed0\n200样本、100轮，迁移5步，阈值0.20']
keys=[None,'author800','control','experiment']
targets=[(14,3),(68,31),(68,22),(68,12),(82,43)]
for ax,field,title,key in zip(axes,[gt]+fields,titles,keys):
    im=ax.imshow(field.T,cmap='coolwarm',vmin=0,vmax=1,extent=(-dx/2,100+dx/2,50+dy/2,-dy/2),interpolation='nearest')
    ax.contour(np.arange(256)*dx,np.arange(128)*dy,(gt<.5).T,levels=[.5],colors='black',linewidths=.45,alpha=.65)
    hits=[True]*5 if key is None else [cov['working_points'][key]['0.01']['cases']['4']['detected'][i] for i in hole_indices]
    for center,target,hit,d in zip(centers,targets,hits,[6,4,3,2,4]):
        # Place glyph beside each diameter callout; connector points to the exact hole.
        ax.annotate(f'{d} mm',xy=center,xytext=target,fontsize=8.7,ha='center',va='center',
                    bbox=dict(boxstyle='round,pad=.18',fc='white',ec='gray',alpha=.95),
                    arrowprops=dict(arrowstyle='-',color='dimgray',lw=.7))
        ax.text(target[0]-8,target[1],'\u2713' if hit else '\u2717',fontfamily='DejaVu Sans',fontsize=15,
                weight='bold',color='#006838' if hit else '#a00000',ha='center',va='center',
                bbox=dict(fc='white',ec='none',alpha=.9,pad=.1))
    ax.set(title=title,xlim=(0,100),ylim=(50,0),xlabel='x（毫米）')
axes[0].set_ylabel('y（毫米）')
cax=fig.add_axes([.945,.235,.013,.49]);cb=fig.colorbar(im,cax=cax);cb.set_label('材料系数 γ（统一色标）')
fig.suptitle('案例4：圆孔检出对比（沿用已保存的1%合并虚警工作点）',fontsize=15,y=.96)
fig.text(.5,.07,'✓：检出；✗：漏检。真值栏✓仅表示存在。黑线为真值轮廓；各方法阈值在同组四案例上选择。',ha='center',fontsize=10)
fig.text(.5,.025,'孔标签直径从左到右对应：左6 mm、中央4/3/2 mm、右4 mm；检出不代表形状准确恢复。',ha='center',fontsize=9)
png=OUT/'case4_four_methods_300dpi.png';fig.savefig(png,dpi=300);plt.close(fig)
with Image.open(png) as pic:
    assert pic.size==(5400,1680)
    dpi=pic.info.get('dpi');assert dpi and abs(dpi[0]-300)<.1 and abs(dpi[1]-300)<.1
    pic.verify()
elapsed=time.perf_counter()-START
seedtable=['| 训练种子 | 对照组1%检出 | 扩展组1%检出 |','|---|---|---|','| seed0（已有） | 5/12 | 9/12 |']
seedtable += [f'| seed{i} | 未测 | 未测 |' for i in range(1,5)]
existing=record.read_text(encoding='utf-8')
section='\n\n## 老师汇报材料 2026-10-13（仅汇总与作图）'
if section in existing:
    record.write_text(existing[:existing.index(section)],encoding='utf-8')
with record.open('a',encoding='utf-8') as f:
    f.write('\n\n## 老师汇报材料 2026-10-13（仅汇总与作图）\n')
    f.write('时间网格：论文 1000×60 ns，代码 2000×30 ns，总时长相同；60 ns 在发布网格上违反 CFL，统一按代码执行（已知差异，已豁免）。\n')
    f.write('昨晚第2–4段已取消，尤其不开始FNO。第1项种子复现先报告未豁免差异后不开跑：预训练表4裁剪5e-5 vs代码5e-4、beta=0.2 vs实际置0；迁移表3裁剪1e-5 vs代码5e-5。指定lr=5e-4仍保留，不自行改为论文8e-4。既有9对5结果保留。\n')
    f.write('判定：训练种子稳健性未验证，不能记稳健或不稳健；5种子均值/范围、孔检出频次/5、案例3均值及狭缝均值均未测。\n\n'+'\n'.join(seedtable)+'\n\n'+md)
    f.write('案例4图：`.build/singh-teacher-report-20261013/case4_four_methods_300dpi.png`；四栏真值/作者800样本迁移5步/对照seed0/扩展seed0，同色标0–1，五孔✓✗直接摘取既有1%工作点，未重新评分。\n')
    f.write(f'PNG结构检查通过：5400×1680像素、300 dpi；汇总和导出脚本耗时{elapsed:.2f}秒（不含人工阅读及后续目视检查），新增训练/数据生成/反演均0分钟。图将进行目视检查；不将尚未检查的内容称已通过。\n已停止，不开始FNO。\n')
print(json.dumps(dict(summary=str(OUT/'summary_table.md'),figure=str(png),seconds=elapsed,dpi=dpi,table_rows=len(table)-2),ensure_ascii=False))
print('\n'.join(table))
