"""New quality-guard experiment; import-only helpers, no legacy files changed."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MPLBACKEND','Agg')
import sys,time,json,gc
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LEGACY=ROOT/'Research_Direction/Author_Baseline_Singh_Zenodo14616677'
SUP=LEGACY/'singh_zenodo';AUTHOR=SUP/'singh_zenodo'
BUILD=ROOT/'.build/fwi-quality-guards-20261013'
sys.path[:0]=[str(SUP),str(AUTHOR),str(LEGACY)]
import Utilities as U
import NeuralNetwork as NN
import FiniteDifferencePyTorchConv as FD
import AdjointMethod as AM
torch.set_num_threads(4)
DEVICE=torch.device('cuda')
GMIN=1e-5

def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def arrays(nx,ny):
    return np.meshgrid(np.linspace(-.1/nx,.1+.1/nx,nx+3),np.linspace(-.05/ny,.05+.05/ny,ny+3),indexing='ij')

def raster(spec,nx,ny):
    x,y=arrays(nx,ny);labels=np.zeros(x.shape,np.int16)
    for i,p in enumerate(spec,1):
        a=(x-p['x'])*np.cos(p['angle'])+(y-p['y'])*np.sin(p['angle'])
        b=-(x-p['x'])*np.sin(p['angle'])+(y-p['y'])*np.cos(p['angle'])
        if p['kind']=='hole':inside=a*a+b*b <= (p['diameter']/2)**2
        elif p['kind']=='slit':inside=(abs(a)<=p['length']/2)&(abs(b)<=p['width']/2)
        else:inside=(a/p['a'])**2+(b/p['b'])**2<=1
        assert not np.any(inside & (labels>0))
        labels[inside]=i
    material=np.ones(labels.shape,np.float32);material[labels>0]=GMIN
    return material,labels

def setup(nx=255,ny=127,nt=2000,dt=30e-9):
    dx=.1/nx;dy=.05/ny
    assert 6000*dt*np.sqrt(dx**-2+dy**-2)<1
    sx,sy=U.getSourceLocations(255,127,36,4)
    rx,ry=U.getSensorLocations(255,127,6,sx)
    source_xy=np.stack([(np.array(sx)-1)*(.1/255),(np.array(sy)-1)*(.05/127)],axis=1)
    receiver_xy=np.stack([(np.array(rx)-1)*(.1/255),(np.array(ry)-1)*(.05/127)],axis=1)
    # Non-nested 511/255 fine grid: linear source deposition and receiver sampling
    # retain the same physical acquisition coordinates and integrated source strength.
    spatial=torch.zeros((4,1,nx+3,ny+3),device=DEVICE)
    def interpolation(xy):
        coords=xy/np.array([dx,dy])+1
        base=np.floor(coords+1e-10).astype(int);frac=coords-base
        frac[np.abs(frac)<1e-9]=0
        return [(base[:,0]+i,base[:,1]+j,((1-frac[:,0]) if i==0 else frac[:,0])*((1-frac[:,1]) if j==0 else frac[:,1])) for i,j in [(0,0),(1,0),(0,1),(1,1)]]
    for ix,iy,w in interpolation(source_xy):
        spatial[torch.arange(4),0,torch.as_tensor(ix),torch.as_tensor(iy)]+=torch.as_tensor(w,device=DEVICE,dtype=torch.float32)
    wave=U.generateSineBurst(500000,2,1e12)(np.arange(nt)*dt)/(dx*dy)
    wave=torch.as_tensor(wave,device=DEVICE,dtype=torch.float32)
    solver=FD.FiniteDifference(dt,dx,dy,6000.,2700.,device=DEVICE)
    return dict(nx=nx,ny=ny,nt=nt,dt=dt,dx=dx,dy=dy,solver=solver,spatial=spatial,wave=wave,interpolation=interpolation(receiver_xy),rx=rx,ry=ry)

def receiver_forward(s,gamma):
    u0=torch.zeros((4,1,s['nx']+3,s['ny']+3),device=DEVICE);u1=torch.zeros_like(u0)
    out=torch.empty((4,24,s['nt']),device=DEVICE)
    for t in range(s['nt']):
        u2=s['solver'].forward(u0.clone(),u1.clone(),gamma,s['spatial']*s['wave'][t])
        u0[:],u1[:]=u1,u2
        value=0
        for ix,iy,w in s['interpolation']:
            value=value+u2[:,0,ix,iy]*torch.as_tensor(w,device=DEVICE,dtype=torch.float32)
        out[:,:,t]=value
    return out

def dense_source(s):return s['spatial'][...,None]*s['wave']

def cost_grad(s,gamma,f,measurement):
    z=torch.zeros((4,1,s['nx']+3,s['ny']+3),device=DEVICE)
    with torch.no_grad():
        return AM.getAdjointGradient(s['solver'],z,z.clone(),6000.,2700.,gamma.detach(),f,s['nx'],s['dx'],s['ny'],s['dy'],s['nt'],s['dt'],4,measurement,s['rx'],s['ry'],DEVICE)

def cleanup():gc.collect();torch.cuda.empty_cache()

def data_path(case):
    path=BUILD/'geometry_data_overrides.json'
    mapping=json.loads(path.read_text()) if path.exists() else {}
    return BUILD/mapping.get(case,'data/'+case+'.npz')
