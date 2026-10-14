"""Run only the authorized B-group validation calibration, after A/C finish."""
import subprocess
from common import *

if __name__=='__main__':
    cfg=json.loads((BUILD/'config.json').read_text());started=time.time();events=[]
    while not (BUILD/'frozen_thresholds.json').exists():
        statuspath=BUILD/'round_status.json'
        if statuspath.exists() and json.loads(statuspath.read_text())['status']=='stopped':raise RuntimeError('STOP B: current A/C round stopped')
        if time.time()>cfg['deadline_unix']-3600:raise RuntimeError('STOP: insufficient budget to queue B calibration')
        time.sleep(2)
    while not (BUILD/'round_status.json').exists():time.sleep(.2)
    state=json.loads((BUILD/'round_status.json').read_text())
    if state['status']=='stopped':raise RuntimeError('STOP B: A/C stopped')
    state.update(status='running B validation calibration supplement');dump(BUILD/'round_status.json',state)
    for i in range(25):
        if time.time()>cfg['deadline_unix']-180:raise RuntimeError('STOP first-round eight-hour budget before B completion')
        case=f'validation_{i:03d}';t=time.time()
        log=BUILD/f'calibration_B_{case}.log'
        with log.open('w',encoding='utf-8') as h:
            result=subprocess.run([sys.executable,str(HERE/'invert.py'),'--case',case,'--seed','30','--epoch','100','--initialization','ellipse','--destination','calibration_B'],stdout=h,stderr=subprocess.STDOUT,timeout=min(900,cfg['deadline_unix']-time.time()-30))
        row=dict(name=f'calibration_B_{case}',seconds=time.time()-t,returncode=result.returncode);events.append(row);dump(BUILD/'B_events.json',events)
        print('B_JOB_FINISHED',json.dumps(row),flush=True)
        if result.returncode:raise RuntimeError('STOP B calibration failure; inspect log before retry')
    result=subprocess.run([sys.executable,str(HERE/'evaluate.py')])
    if result.returncode:raise RuntimeError('STOP B threshold freeze failed')
    state=json.loads((BUILD/'round_status.json').read_text());state.update(status='completed first-round scope including B supplement',B_calibration=True,seconds=time.time()-cfg['start_unix']);dump(BUILD/'round_status.json',state)
    print('B_CALIBRATION_COMPLETE',time.time()-started,flush=True)
