"""Serial approved 255 inversions, isolated workers and seven-hour hard cap."""
import subprocess,sys,traceback
from common import *
ROUND=ROOT/'.build/fwi-quality-guards-round2-20261013'
cfg=json.loads((ROUND/'config.json').read_text());selection=json.loads((ROUND/'selection.json').read_text())
start=time.monotonic();deadline=start+cfg['runtime_limit_seconds'];completed=[];failures=[]
status_path=ROUND/'status.json'
def status(state,current=None):
 dump(status_path,dict(status=state,started_utc=started,elapsed_seconds=time.monotonic()-start,completed=len(completed),planned=255,current=current,failures=failures,finished=completed))
started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
status('running')
for case in selection['cases']:
 for seed in cfg['seeds']:
  for group,epoch,initialization in [('A',0,'expanded'),('B',100,'ellipse'),('C',100,'expanded')]:
   job=dict(case=case,seed=seed,group=group,epoch=epoch)
   dest='group_'+group;out=ROUND/dest/f'seed{seed}_epoch{epoch}'/case
   if (out/'metrics.json').exists():raise RuntimeError('Existing results: do not overwrite or silently resume')
   success=False
   for attempt in range(2):
    remaining=deadline-time.monotonic()
    if remaining<100:
     status('stopped: budget reserve insufficient',job);sys.exit(2)
    status('running',dict(**job,attempt=attempt+1))
    log=ROUND/'logs'/f'{case}_seed{seed}_{group}_attempt{attempt+1}.log';log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(HERE/'invert_round2.py'),'--case',case,'--seed',str(seed),'--epoch',str(epoch),'--destination',dest,'--initialization',initialization]
    try:
     with log.open('w',encoding='utf-8') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,remaining-2))
     if result.returncode:raise RuntimeError(f'worker exit {result.returncode}; see {log}')
     m=json.loads((out/'metrics.json').read_text());assert m['saved_material_steps']==[0,5,10,20,35]
     completed.append(dict(**job,wall_seconds=m['wall_seconds'],metrics=str(out/'metrics.json')))
     status('running');print('COMPLETED',len(completed),'/255',case,seed,group,round(m['wall_seconds'],2),flush=True);success=True;break
    except Exception as e:
     failures.append(dict(**job,attempt=attempt+1,error=str(e)))
     print('FAILURE',json.dumps(failures[-1]),flush=True)
     if time.monotonic()>=deadline:status('stopped: hard budget',job);sys.exit(2)
   if not success:
    status('stopped: same item failed twice',job);sys.exit(3)
status('completed; stopped');print('ROUND2_COMPLETE',flush=True)
