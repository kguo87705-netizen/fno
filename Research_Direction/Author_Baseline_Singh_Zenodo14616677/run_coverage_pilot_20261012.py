import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['MPLBACKEND']='Agg'
import sys,time,json,threading,random
from pathlib import Path
import numpy as np
import torch
import Utilities
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
SUP=BASE/'singh_zenodo'
AUTHOR=SUP/'singh_zenodo'
OUT=ROOT/'.build/singh-coverage-paired-20261012'
OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(SUP),str(AUTHOR)]
torch.set_num_threads(4)
def compact_source(frequency,cycles,amplitude,sx,sy,Nx,Ny,dx,dy,Lx,Ly,N,dt):
    wave=Utilities.generateSineBurst(frequency,cycles,amplitude)(np.linspace(0,(N-1)*dt,N))
    spatial=torch.zeros((len(sx),1,Nx+3,Ny+3),device='cuda')
    for i,(ix,iy) in enumerate(zip(sx,sy)):spatial[i,0,ix,iy]=1
    wave=torch.from_numpy(((1/dx)*(1/dy)*wave).astype(np.float32)).to('cuda')
    return spatial,wave
def small_source(frequency,cycles,amplitude,sx,sy,Nx,Ny,dx,dy,Lx,Ly,N,dt):
    spatial,wave=compact_source(frequency,cycles,amplitude,sx,sy,Nx,Ny,dx,dy,Lx,Ly,N,dt)
    return spatial[...,None]*wave
Utilities.getSource=small_source
def receiver_forward(solver,u0,u1,gamma,source,N,selx,sely):
    spatial,wave=source
    out=torch.zeros((len(u0),len(selx),N),device=u0.device)
    for i in range(N):
        u2=solver.forward(u0.clone(),u1.clone(),gamma,spatial*wave[i])
        u0[:],u1[:]=u1,u2
        out[:,:,i]=u2[:,0,selx,sely]
    return out
torch.manual_seed(2);np.random.seed(2);random.seed(2)
start=time.time()
def stop():
    print('STOP pilot 15 minute limit',flush=True);os._exit(124)
timer=threading.Timer(900,stop);timer.daemon=True;timer.start()
record=ROOT/'PINO超声成像知识库/05实践与复习/常规FWI正则对照 2026-10.md'
with record.open('a',encoding='utf-8') as f:
    f.write('\n\n## 获书面授权：同规模训练分布对照（运行前判据）\n判据固定：实验组在合并虚警≤1%工作点比对照组至少多检出2个缺陷，记为“数据覆盖是主因之一”；不足2个记“未支持”，不据此断言U-Net有问题。0.5%工作点、各例虚警与MSE为补充，不替换主判据。\n两组默认各200，若预算不足同时缩至至少100；作者Pretraining.py实际epochs=1，RMSprop lr=5e-4、梯度裁剪5e-4、80/20划分、batchSize=N//10照用。种子：几何NumPy/Python/Torch=2、网络初始化Torch=30、划分Torch=2，两组一致。仅计时阶段可重复单样本，权重作废。\n')
gen=(ROOT/'.build/author-observation-control-20261008/downloaded_generation_sources/GenerateMeasurementDataTrain.py').read_text(encoding='utf-8-sig')
pilot=OUT/'pilot';pilot.mkdir(exist_ok=True)
gen=gen.replace('destinationFolder = "data"','destinationFolder = '+repr(str(pilot)))
gen=gen.replace('if destinationFolder not in os.listdir(os.getcwd()):','if not os.path.isdir(destinationFolder):')
gen=gen.replace('numberOfCases = 5','numberOfCases = 1')
gen=gen.replace('F_ = Utilities.getSource(', 'F_ = compact_source(').replace('Nx_, Ny_, dx_, dy_, Lx, Ly, N_, dt_).to(device)','Nx_, Ny_, dx_, dy_, Lx, Ly, N_, dt_)')
a=gen.index('    Um = forwardSolver_.forwardNSteps(')
b=gen.index('    gamma = torch.ones',a)
gen=gen[:a]+'    Um = receiver_forward(forwardSolver_,u0_.clone(),u1_.clone(),gamma_,F_,N_,selx_,sely_)\n\n'+gen[b:]
print('PILOT_GENERATION_START',flush=True)
t=time.perf_counter()
with torch.no_grad():exec(compile(gen,'author_generate_pilot','exec'),{'__file__':str(AUTHOR/'GenerateMeasurementDataTrain.py'),'__name__':'__main__','compact_source':compact_source,'receiver_forward':receiver_forward})
torch.cuda.synchronize();tg=time.perf_counter()-t
print('PILOT_GENERATION_SECONDS',tg,flush=True)
train=(SUP/'Pretraining.py').read_text(encoding='utf-8-sig')
train=train.replace('destinationFolder = "data"','destinationFolder = '+repr(str(pilot)))
train=train.replace('numberOfSamples = 800','numberOfSamples = 200')
train=train.replace('random.sample(range(800), numberOfSamples)','[0] * numberOfSamples')
train=train[:train.index('# Save the pretrained model')]
print('PILOT_TRAINING_START 200 repeated samples; timing only',flush=True)
t=time.perf_counter()
ns={'__file__':str(AUTHOR/'Pretraining.py'),'__name__':'__main__'}
exec(compile(train,'author_train_timing','exec'),ns)
torch.cuda.synchronize();tt=time.perf_counter()-t
result=dict(generation_one_seconds=tg,training_one_epoch_200_with_load_seconds=tt,pilot_wall_seconds=time.time()-start,
            training_timing_note='Same generated sample repeated 200 times; discard timing weights; epoch=1',device=str(ns['device']))
(OUT/'pilot_times.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('PILOT_RESULT '+json.dumps(result),flush=True)
timer.cancel()
