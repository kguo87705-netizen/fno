"""Independent T4 budget: stop at two failures, then leave T2 unaffected."""
import subprocess
from common import *
OUT=ROOT/'.build/fwi-quality-guards-round2-20261013/t4'
OUT.mkdir(parents=True,exist_ok=True)
start=time.monotonic();failures=0;rows=[]
jobs=[(1,'physical_streaming'),(2,'physical_streaming'),(0,'fno_forward'),(0,'fno_backward'),(1,'fno_forward'),(1,'fno_backward'),(2,'fno_forward'),(2,'fno_backward')]
for scale,kind in jobs:
 remaining=3600-(time.monotonic()-start)
 if remaining<30:break
 path=OUT/f'scale{scale}_{kind}.json';assert not path.exists()
 with (OUT/f'scale{scale}_{kind}.log').open('w',encoding='utf-8') as f:
  try:r=subprocess.run([sys.executable,str(HERE/'benchmark_round2.py'),'--scale',str(scale),'--kind',kind],stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
  except subprocess.TimeoutExpired:
   rows.append(dict(scale=scale,kind=kind,status='budget timeout'));break
 if r.returncode or not path.exists():
  failures+=1;rows.append(dict(scale=scale,kind=kind,status='failed',exitcode=r.returncode))
 else:
  row=json.loads(path.read_text());rows.append(row)
  failures+=int(row['status']!='ok')
 print('T4_JOB',scale,kind,rows[-1]['status'],flush=True)
 dump(OUT/'status.json',dict(elapsed_seconds=time.monotonic()-start,failures=failures,rows=rows,status='running'))
 if failures>=2:break
dump(OUT/'status.json',dict(elapsed_seconds=time.monotonic()-start,failures=failures,rows=rows,status='stopped'))
print('T4_STOPPED',flush=True)
