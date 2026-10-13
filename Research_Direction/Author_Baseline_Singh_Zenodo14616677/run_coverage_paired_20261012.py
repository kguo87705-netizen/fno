"""One authorized paired coverage experiment, fixed author training settings."""
import os
os.environ['OMP_NUM_THREADS']='4';os.environ['MPLBACKEND']='Agg'
import sys,time,json,random,threading,shutil
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
SUP=BASE/'singh_zenodo';AUTHOR=SUP/'singh_zenodo'
OUT=ROOT/'.build/singh-coverage-paired-20261012'
sys.path[:0]=[str(SUP),str(AUTHOR)]
# Reuse only helper definitions, never rerun pilot.
pilotcode=(BASE/'run_coverage_pilot_20261012.py').read_text(encoding='utf-8')
exec(compile(pilotcode[:pilotcode.index('start=time.time()')],'pilot_helpers','exec'),globals())
import Utilities
from scipy import ndimage
START=time.time()
DEADLINE=START+3000 # Leaves room for already completed preparation and final scoring inside 60 minutes.
def timeout():
    print('STOP paired reserved budget exhausted',flush=True);os._exit(124)
timer=threading.Timer(3000,timeout);timer.daemon=True;timer.start()
record=ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md'
with record.open('a',encoding='utf-8') as f:
    f.write('\n生成规则（执行前固定）：每组200；对照200个作者Utilities.generateGamma单椭圆，实验复用前100个，再加入34个单圆孔、33个单狭缝、33个2–5缺陷组合。圆孔直径连续均匀2–6毫米；狭缝宽度均匀抽取粗网格1/2/3格、长度均匀6–18毫米、方向均匀0–π；组合由圆孔/狭缝随机混合，各缺陷留边界且至少相隔2粗网格，禁止相交故无十字。排除与任一测试材料完全相同的掩膜；不调用测试坐标、轮廓或组合模板，组合不含矩形或椭圆，无法复制案例2矩形加圆、案例3椭圆组合、案例4十字加五孔。相同类型的随机小孔/狭缝允许出现，否则无法研究类型覆盖；不承诺绝无外观相似。两组各自按作者种子2做80/20划分，模型从相同种子30初始化。\n计时：一次内存分配失败；保持源项公式/差分更新不变，细网格改为仅保存接收波形后重试成功，单样本含初始化保存8.0084秒、200重复样本计时训练一轮含读入验证3.8009秒（权重丢弃）。300个独立生成样本、两组各200，按8.01秒/样本加训练、迁移及评分余量估算约45分钟，执行200档。\n')
truths=[]
import pandas as pd
for c in range(1,5):truths.append(pd.read_hdf(AUTHOR/f'data_casestudy/material{c}.h5').values<.5)
metadata=[]
def build_material(case,group,Nx,Ny,dx,dy,Lx,Ly,gamma0):
    if group=='control':
        for attempt in range(1000):
            g,*p=Utilities.generateGamma(Nx,Ny,dx,dy,Lx,Ly,gamma0)
            coarse=g[0,0,1:-1:2,1:-1:2].numpy()<.5
            if not any(np.array_equal(coarse,t) for t in truths):break
        else:raise RuntimeError('Ellipse rejection limit')
        metadata.append(dict(group=group,index=case,type='author_single_ellipse',parameters=[float(x) for x in p]))
        return g.to('cuda')
    typ=['circle','slit','combination'][(case-100)%3]
    n=1 if typ!='combination' else np.random.randint(2,6)
    xx,yy=np.meshgrid(np.linspace(-dx,Lx+dx,Nx+3),np.linspace(-dy,Ly+dy,Ny+3),indexing='ij')
    mask=np.zeros(xx.shape,bool);spec=[]
    for j in range(n):
        kind=typ if typ!='combination' else np.random.choice(['circle','slit'])
        for attempt in range(1000):
            cx=np.random.uniform(.008,Lx-.008);cy=np.random.uniform(.008,Ly-.008)
            if kind=='circle':
                diameter=np.random.uniform(.002,.006)
                part=(xx-cx)**2+(yy-cy)**2 <= (diameter/2)**2
                params=dict(type=kind,x=cx,y=cy,diameter=diameter)
            else:
                width_grid=int(np.random.randint(1,4));width=width_grid*(2*dx)
                length=np.random.uniform(.006,.018);theta=np.random.uniform(0,np.pi)
                along=(xx-cx)*np.cos(theta)+(yy-cy)*np.sin(theta)
                across=-(xx-cx)*np.sin(theta)+(yy-cy)*np.cos(theta)
                part=(abs(along)<=length/2)&(abs(across)<=width/2)
                params=dict(type=kind,x=cx,y=cy,width_grid=width_grid,width=width,length=length,theta=theta)
            if part[:5,:].any() or part[-5:,:].any() or part[:,:5].any() or part[:,-5:].any():continue
            if np.any(part & ndimage.binary_dilation(mask,structure=np.ones((3,3)),iterations=4)):continue
            if not part[1:-1:2,1:-1:2].any():continue
            mask|=part;spec.append(params);break
        else:raise RuntimeError('Geometry rejection limit')
    coarse=mask[1:-1:2,1:-1:2]
    assert not any(np.array_equal(coarse,t) for t in truths)
    g=np.ones(mask.shape,np.float32);g[mask]=gamma0
    metadata.append(dict(group=group,index=case,type=typ,components=spec))
    return torch.from_numpy(g)[None,None].to('cuda')
genbase=(ROOT/'.build/author-observation-control-20261008/downloaded_generation_sources/GenerateMeasurementDataTrain.py').read_text(encoding='utf-8-sig')
def generate(group):
    folder=OUT/group/'data';folder.mkdir(parents=True,exist_ok=True)
    if group=='experiment':
        for c in range(100):
            for stem in ['material','gradient','measurement','parametersMaterial']:
                os.link(OUT/'control'/'data'/f'{stem}{c}.h5',folder/f'{stem}{c}.h5')
    code=genbase.replace('destinationFolder = "data"','destinationFolder = '+repr(str(folder)))
    code=code.replace('if destinationFolder not in os.listdir(os.getcwd()):','if not os.path.isdir(destinationFolder):')
    code=code.replace('numberOfCases = 5','numberOfCases = 200').replace('for case in range(numberOfCases):','for case in range('+('0' if group=='control' else '100')+', numberOfCases):')
    code=code.replace('F_ = Utilities.getSource(','F_ = compact_source(').replace('Nx_, Ny_, dx_, dy_, Lx, Ly, N_, dt_).to(device)','Nx_, Ny_, dx_, dy_, Lx, Ly, N_, dt_)')
    a=code.index('    gamma_ = torch.ones(');b=code.index('    gamma_ = gamma_.to(device)',a)
    code=code[:a]+f"    gamma_ = build_material(case, '{group}', Nx_, Ny_, dx_, dy_, Lx, Ly, gamma0)\n    x0=y0=a=b=theta=float('nan')\n\n"+code[b:]
    a=code.index('    Um = forwardSolver_.forwardNSteps(');b=code.index('    gamma = torch.ones',a)
    code=code[:a]+'    Um = receiver_forward(forwardSolver_,u0_.clone(),u1_.clone(),gamma_,F_,N_,selx_,sely_)\n\n'+code[b:]
    code=code.replace('        print(case)','        print("GENERATED '+group+'",case,"elapsed",time.time()-RUN_START,flush=True)')
    np.random.seed(2);random.seed(2);torch.manual_seed(2)
    t=time.perf_counter()
    ns=dict(__file__=str(AUTHOR/'GenerateMeasurementDataTrain.py'),__name__='__main__',build_material=build_material,compact_source=compact_source,receiver_forward=receiver_forward,RUN_START=START)
    with torch.no_grad():exec(compile(code,'author_generation_adapted','exec'),ns)
    del ns;torch.cuda.empty_cache()
    (OUT/'geometry_samples.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return time.perf_counter()-t
def train(group):
    random.seed(2);np.random.seed(2);torch.manual_seed(2)
    code=(SUP/'Pretraining.py').read_text(encoding='utf-8-sig')
    code=code.replace('destinationFolder = "data"','destinationFolder = '+repr(str(OUT/group/'data')))
    code=code.replace('numberOfSamples = 800','numberOfSamples = 200').replace('random.sample(range(800), numberOfSamples)','random.sample(range(numberOfSamples), numberOfSamples)')
    code=code[:code.index('# Save the pretrained model')]
    t=time.perf_counter();ns=dict(__file__=str(AUTHOR/'Pretraining.py'),__name__='__main__')
    exec(compile(code,'author_pretraining_200','exec'),ns)
    torch.cuda.synchronize();elapsed=time.perf_counter()-t
    assert ns['epochs']==1 and ns['lr']==5e-4 and ns['batchSize']==20
    torch.save(ns['model'].state_dict(),OUT/group/'model_Unet_200')
    stats=dict(seconds=elapsed,epochs=int(ns['epochs']),training_loss=float(ns['trainingCostHistory'][-1]),validation_loss=float(ns['validationCostHistory'][-1]))
    del ns;torch.cuda.empty_cache();return stats
timings={}
for group in ['control','experiment']:
    print('GROUP_START',group,flush=True)
    timings[group]={'generate_seconds':generate(group),'training':train(group)}
    (OUT/'timings.json').write_text(json.dumps(timings,indent=2),encoding='utf-8')
print('GENERATION_TRAINING_FINISHED',time.time()-START,flush=True)
# Reuse author TransferLearningFWI path adaptations and post-update field extraction.
helper=(BASE/'run_pretrained_baselines_20261012.py').read_text(encoding='utf-8')
helper=helper[:helper.index("if __name__=='__main__':")]
hns={'__file__':str(BASE/'run_pretrained_baselines_20261012.py'),'__name__':'helpers'}
exec(compile(helper,'transfer_helpers','exec'),hns)
for group in ['control','experiment']:
    code=(SUP/'TransferLearningFWI.py').read_text(encoding='utf-8-sig')
    code=code.replace('cases = np.linspace(1, 4, 4, dtype=int)','cases = np.array([selected_case],dtype=int)')
    for old in ['destinationFolder = "data_casestudy"','destinationFolder = "data"']:code=code.replace(old,'destinationFolder = outputFolder')
    code=code.replace('if destinationFolder not in os.listdir(os.getcwd()):','if not os.path.isdir(destinationFolder):')
    for stem in ['material','gradient','measurement','source.h5']:code=code.replace(f'destinationFolder + "/{stem}',f'inputFolder + "/{stem}')
    a=code.index('model.load_state_dict(torch.load(');b=code.index('map_location=device))',a)+len('map_location=device))')
    code=code[:a]+'model.load_state_dict(torch.load(pretrainedPath,map_location=device,weights_only=True))\n'+code[b:]
    code=code.replace('        start0 = time.perf_counter()','        before(globals())\n        start0 = time.perf_counter()')
    code=code[:code.index('        ##Plotting and saving figures over iterations')]+'        finish(globals())\n'
    rows=[]
    for c in range(1,5):
        if time.time()>DEADLINE-100:raise TimeoutError('Budget insufficient for remaining transfer')
        folder=OUT/group/f'case{c}';folder.mkdir(exist_ok=True)
        hns['CASE_START']=time.perf_counter()
        ns=dict(__file__=str(AUTHOR/'TransferLearningFWI.py'),__name__='__main__',selected_case=c,method='transfer',inputFolder=str(AUTHOR/'data_casestudy'),outputFolder=str(folder),pretrainedPath=str(OUT/group/'model_Unet_200'),before=hns['before'],finish=hns['finish'])
        exec(compile(code,'author_transfer_five','exec'),ns)
        rows.append(ns['final_metrics']);del ns;torch.cuda.empty_cache()
    timings[group]['transfer']=rows
    (OUT/'timings.json').write_text(json.dumps(timings,indent=2),encoding='utf-8')
timings['paired_wall_seconds']=time.time()-START
(OUT/'timings.json').write_text(json.dumps(timings,indent=2),encoding='utf-8')
print('PAIRED_FINISHED',time.time()-START,flush=True);timer.cancel()
