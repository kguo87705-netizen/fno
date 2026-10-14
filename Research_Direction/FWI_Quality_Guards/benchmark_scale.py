"""Single isolated scale/mode timing; no training, no memory checkpointing."""
import argparse,statistics
from common import *
from basic_fno import BasicFNO

def run(scale,kind):
    nx,ny,nt,dt=[(255,127,2000,30e-9),(511,255,4000,15e-9),(1023,511,8000,7.5e-9)][scale]
    row=dict(scale=scale,Nx=nx,Ny=ny,Nt=nt,dt=dt,kind=kind,device=torch.cuda.get_device_name(),output_frames=500 if kind.startswith('fno') else nt,precision='float32/complex64',warmup=3,repetitions=10,source_batch=1 if kind.startswith('fno') else 4)
    try:
        if kind.startswith('physical'):
            s=setup(nx,ny,nt,dt);gamma=torch.ones((1,1,nx+3,ny+3),device=DEVICE)
            f=dense_source(s);z=torch.zeros((4,1,nx+3,ny+3),device=DEVICE)
            if kind=='physical_forward':
                def operation():
                    with torch.no_grad():out=s['solver'].forwardNSteps(z.clone(),z.clone(),gamma,f,nx,ny,nt,4,DEVICE)
                    del out
            else:
                # Receiver coordinates vary with scale; round physical positions only
                # for this isolated timing. It makes no quality claim.
                rx=np.rint((np.array(s['rx'])-1)*nx/255).astype(int)+1
                ry=[ny+1]*len(rx);meas=torch.zeros((4,24,nt),device=DEVICE)
                def operation():
                    with torch.no_grad():out=AM.getAdjointGradient(s['solver'],z,z.clone(),6000.,2700.,gamma,f,nx,s['dx'],ny,s['dy'],nt,dt,4,meas,rx,ry,DEVICE)
                    del out
        else:
            torch.manual_seed(20261013)
            model=BasicFNO(width=8,modes=(16,8,64),layers=4,input_channels=8).to(DEVICE).eval()
            model.requires_grad_(False)
            # Full R2 input includes material, coordinates, and source descriptors.
            # Random benchmark data change values only, not tensor dimensions.
            x=torch.randn((1,8,nx+1,ny+1,500),device=DEVICE)
            backward='backward' in kind;receivers='receiver' in kind
            x.requires_grad_(backward)
            def operation():
                if backward:
                    x.grad=None
                    y=model(x)
                    if receivers:y=y[:,:,::max(1,(nx+1)//24),ny//2,:]
                    loss=y.square().mean();loss.backward();del y,loss
                else:
                    with torch.no_grad():
                        y=model(x)
                        if receivers:y=y[:,:,::max(1,(nx+1)//24),ny//2,:].contiguous()
                    del y
        torch.cuda.reset_peak_memory_stats()
        for _ in range(3):operation()
        torch.cuda.synchronize();times=[]
        for _ in range(10):
            t=time.perf_counter();operation();torch.cuda.synchronize();times.append(time.perf_counter()-t)
        row.update(status='ok',seconds=times,median_seconds=statistics.median(times),peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except torch.OutOfMemoryError as e:
        row.update(status='OOM',error=str(e).splitlines()[0])
    dump(BUILD/'scale'/f'scale{scale}_{kind}.json',row)
    print('SCALE_RESULT',json.dumps(row),flush=True)
    return row

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scale',type=int,required=True);p.add_argument('--kind',required=True);a=p.parse_args();run(a.scale,a.kind)
