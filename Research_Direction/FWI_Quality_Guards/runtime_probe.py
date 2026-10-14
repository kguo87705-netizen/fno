"""35-step end-to-end timing and resource-limited concurrency probe."""
import subprocess,threading,shutil
from common import *

def invoke(destination):
    log=BUILD/f'{destination}.log'
    handle=log.open('w',encoding='utf-8')
    p=subprocess.Popen([sys.executable,str(HERE/'invert.py'),'--case','validation_000','--destination',destination],stdout=handle,stderr=subprocess.STDOUT)
    return p,handle

if __name__=='__main__':
    samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            q=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if q.returncode==0:
                u,m=map(float,q.stdout.strip().split(','));samples.append(dict(time=time.time(),gpu_percent=u,memory_mib=m))
            stop.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    start=time.time();p,h=invoke('serial_probe');rc=p.wait();h.close();stop.set();thread.join()
    dump(BUILD/'gpu_samples.json',samples)
    if rc:raise RuntimeError('Serial timing failed; inspect serial_probe.log')
    path=BUILD/'serial_probe/seed30_epoch0/validation_000'
    metrics=json.loads((path/'metrics.json').read_text())
    steady=samples[8:-4] or samples
    utilization=float(np.mean([s['gpu_percent'] for s in steady]));peak_used=max(s['memory_mib'] for s in samples)
    row=dict(serial=metrics,mean_gpu_percent=utilization,peak_device_memory_mib=peak_used,sample_count=len(samples))
    # Only test concurrency if compute is not saturated AND two processes can fit
    # the measured per-process peak. This resource check does not change results.
    total=torch.cuda.get_device_properties(0).total_memory
    can_fit=2*metrics['peak_allocated_bytes']<.92*total
    if utilization>=90:
        row['concurrency_status']='not needed: mean GPU utilization >=90%'
    elif not can_fit:
        row['concurrency_status']='not safe: measured two-process peak exceeds device budget'
    else:
        runs=[];t=time.time()
        for i in range(2):runs.append(invoke(f'concurrent2_{i}'))
        failures=0
        for p,h in runs:
            failures+=int(p.wait()!=0);h.close()
        row['concurrent2_wall_seconds']=time.time()-t;row['concurrent2_failures']=failures
        if failures:
            row['concurrency_status']='concurrent2 failed; no concurrency increase'
            dump(BUILD/'runtime_probe.json',row)
            if failures>=2:raise RuntimeError('STOP: two concurrency failures')
        else:
            base=np.load(path/'fields.npz')['prediction'];compared=[]
            for i in range(2):
                other=np.load(BUILD/f'concurrent2_{i}/seed30_epoch0/validation_000/fields.npz')['prediction']
                compared.append(dict(bitwise=bool(np.array_equal(base,other)),within_tolerance=bool(np.allclose(base,other,atol=1e-7,rtol=1e-5)),max_abs=float(abs(base-other).max())))
            row['comparisons']=compared;row['throughput_speedup']=2*metrics['wall_seconds']/row['concurrent2_wall_seconds']
            row['concurrency_status']='concurrent2 valid' if all(x['within_tolerance'] for x in compared) else 'STOP: concurrent output differs'
            if not all(x['within_tolerance'] for x in compared):
                dump(BUILD/'runtime_probe.json',row);raise RuntimeError('STOP: concurrency equivalence failed')
            # Four workers only if measured memory leaves room; otherwise record.
            row['concurrent4_status']='not safe under four-times measured peak' if 4*metrics['peak_allocated_bytes']>=.92*total else 'not selected: two-worker test sufficient; larger launch requires measured spare compute'
    dump(BUILD/'runtime_probe.json',row);print('RUNTIME_PROBE',json.dumps(row),flush=True)
