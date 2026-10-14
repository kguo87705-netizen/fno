"""Supplemental streaming physical timing; unchanged R2, four-source workload."""
import argparse,statistics
from common import *
from basic_fno import BasicFNO
OUT=ROOT/'.build/fwi-quality-guards-round2-20261013/t4'
def run(scale,kind):
 nx,ny,nt,dt=[(255,127,2000,30e-9),(511,255,4000,15e-9),(1023,511,8000,7.5e-9)][scale]
 row=dict(scale=scale,Nx=nx,Ny=ny,Nt=nt,dt=dt,kind=kind,sources=4,
  source_execution='physical simultaneous batch4' if kind=='physical_streaming' else 'four sequential single-source R2 evaluations',
  precision='float32/complex64',device=torch.cuda.get_device_name(),warmup=3,repetitions=10,
  output_frames=500 if kind.startswith('fno') else 'final state only; no history',
  applicability='timing only; different output storage/frame counts, no quality-qualified FWI acceleration claim')
 try:
  if kind=='physical_streaming':
   s=setup(nx,ny,nt,dt);gamma=torch.ones((1,1,nx+3,ny+3),device=DEVICE)
   z=torch.zeros((4,1,nx+3,ny+3),device=DEVICE);f0=s['spatial']*s['wave'][min(10,nt-1)]
   def single():
    with torch.no_grad():y=s['solver'].forward(z.clone(),z.clone(),gamma,f0)
    del y
   def operation():
    with torch.no_grad():
     u0=z.clone();u1=z.clone()
     for t in range(nt):
      u2=s['solver'].forward(u0.clone(),u1.clone(),gamma,s['spatial']*s['wave'][t])
      u0[:],u1[:]=u1,u2
     assert torch.isfinite(u1).all()
    del u0,u1,u2
   for _ in range(3):single()
   torch.cuda.synchronize();st=[]
   for _ in range(10):
    t=time.perf_counter();single();torch.cuda.synchronize();st.append(time.perf_counter()-t)
   row.update(single_step_seconds=st,single_step_median_seconds=statistics.median(st))
  else:
   torch.manual_seed(20261013)
   model=BasicFNO(width=8,modes=(16,8,64),layers=4,input_channels=8).to(DEVICE).eval().requires_grad_(False)
   row.update(modes=[16,8,64],width=8,layers=4,input_channels=8,output='full field500 fixed frames over60us; spatial grid refined, modes fixed',parameter_count=sum(p.numel() for p in model.parameters()))
   x=torch.randn((1,8,nx+1,ny+1,500),device=DEVICE,requires_grad='backward' in kind)
   def operation():
    for _ in range(4):
     if 'backward' in kind:
      x.grad=None;y=model(x);loss=y.square().mean();loss.backward();del y,loss
     else:
      with torch.no_grad():y=model(x)
      del y
  torch.cuda.reset_peak_memory_stats()
  for i in range(3):
   operation();print('WARMUP',scale,kind,i+1,flush=True)
  torch.cuda.synchronize();times=[]
  for i in range(10):
   t=time.perf_counter();operation();torch.cuda.synchronize();times.append(time.perf_counter()-t)
   print('TIMING',scale,kind,i+1,round(times[-1],4),flush=True)
  row.update(status='ok',seconds=times,median_seconds=statistics.median(times),peak_allocated_bytes=torch.cuda.max_memory_allocated())
 except torch.OutOfMemoryError as e:
  row.update(status='OOM',error=str(e).splitlines()[0])
 dump(OUT/f'scale{scale}_{kind}.json',row);print('T4_RESULT',json.dumps(row),flush=True)
 return row
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scale',type=int,required=True);p.add_argument('--kind',required=True);a=p.parse_args();run(a.scale,a.kind)
