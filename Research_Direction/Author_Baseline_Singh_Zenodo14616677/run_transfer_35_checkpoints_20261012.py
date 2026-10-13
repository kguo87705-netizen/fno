"""Only training change: author TransferLearningFWI epochs 5 -> 35."""
import os
os.environ['MPLBACKEND']='Agg'
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
import sys,time,json,threading
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
SUPPLIED=BASE/'singh_zenodo'
AUTHOR=SUPPLIED/'singh_zenodo'
OUT=ROOT/'.build/singh-transfer35-author-20261012'
STEPS=[5,10,20,35]
ROWS=[]

def snapshot(ns):
    step=ns['epoch']+1
    if time.time()>=DEADLINE:raise TimeoutError('30 minute budget reached')
    if step not in STEPS:return
    torch.cuda.synchronize()
    fit_seconds=time.perf_counter()-ns['start0']
    model=ns['model']
    buffers={name:buf.detach().clone() for name,buf in model.named_buffers()}
    # Keep author's train-mode output, restoring buffers so observation does not
    # modify BatchNorm running statistics or the subsequent optimization path.
    with torch.no_grad():
        gamma=torch.ones_like(ns['gammaPred'])
        gamma[:,:,1:-1,1:-1]=model(ns['inputData'])
        for name,buf in model.named_buffers():buf.copy_(buffers[name])
    pred=gamma.cpu().numpy().squeeze()
    truth=ns['gamma'].cpu().numpy().squeeze()
    assert np.isfinite(pred).all()
    actual_steps=[int(a['step'].item()) for a in ns['optimizer'].state.values() if 'step' in a]
    assert actual_steps and min(actual_steps)==max(actual_steps)==step
    folder=OUT/f'step{step}'/f'case{int(ns["case"])}'
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'final_fields.npz',truth=truth,reconstruction=pred,
        author_preupdate=ns['gammaPred'].detach().cpu().numpy().squeeze())
    row=dict(case=int(ns['case']),step=step,material_mse=float(((pred[1:-1,1:-1].astype(float)-truth[1:-1,1:-1])**2).mean()),
        fit_seconds=fit_seconds,used_lr=ns['lr']*(1+.2*(step-1))**-.5,
        next_lr=ns['optimizer'].param_groups[0]['lr'])
    assert np.isclose(row['next_lr'],ns['lr']*(1+.2*step)**-.5)
    if step==5:
        prior=np.load(ROOT/f'.build/singh-pretrained-author-20261012/transfer/case{int(ns["case"])}/final_fields.npz')
        row['max_difference_from_prior_post5']=float(np.max(abs(pred-prior['post5_reconstruction'])))
        assert row['max_difference_from_prior_post5']<1e-5
    ROWS.append(row)
    (folder/'metrics.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    (OUT/'results.json').write_text(json.dumps(ROWS,indent=2),encoding='utf-8')
    print('CHECKPOINT '+json.dumps(row),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    sys.path[:0]=[str(SUPPLIED),str(AUTHOR)]
    OUT.mkdir(parents=True,exist_ok=True)
    source=(SUPPLIED/'TransferLearningFWI.py').read_text(encoding='utf-8-sig')
    assert source.count('        epochs = 5')==1
    source=source.replace('        epochs = 5','        epochs = 35')
    source=source.replace('cases = np.linspace(1, 4, 4, dtype=int)','cases = np.array([selected_case],dtype=int)')
    source=source.replace('destinationFolder = "data"','destinationFolder = outputFolder')
    source=source.replace('if destinationFolder not in os.listdir(os.getcwd()):','if not os.path.isdir(destinationFolder):')
    for stem in ['material','gradient','measurement','source.h5']:
        source=source.replace(f'destinationFolder + "/{stem}',f'inputFolder + "/{stem}')
    a=source.index('model.load_state_dict(torch.load(')
    b=source.index('map_location=device))',a)+len('map_location=device))')
    source=source[:a]+'model.load_state_dict(torch.load(pretrainedPath,map_location=device,weights_only=True))'+source[b:]
    assert source.count('            scheduler.step()')==1
    source=source.replace('            scheduler.step()','            scheduler.step()\n            snapshot(globals())')
    source=source[:source.index('        ##Plotting and saving figures over iterations')]
    START=time.time();DEADLINE=START+1800
    def stop():print('STOP 30 minute deadline',flush=True);os._exit(124)
    watchdog=threading.Timer(1800,stop);watchdog.daemon=True;watchdog.start()
    cases=[]
    for c in range(1,5):
        if DEADLINE-time.time()<220:raise TimeoutError('Insufficient remaining budget')
        folder=OUT/f'case{c}';folder.mkdir(parents=True,exist_ok=True)
        ns=dict(__file__=str(AUTHOR/'TransferLearningFWI.py'),__name__='__main__',selected_case=c,
            inputFolder=str(AUTHOR/'data_casestudy'),outputFolder=str(folder),
            pretrainedPath=str(SUPPLIED/'model_Unet_pretrained_800'),snapshot=snapshot)
        ts=time.perf_counter()
        print(f'RUN case={c} remaining={DEADLINE-time.time():.1f}s',flush=True)
        exec(compile(source,str(SUPPLIED/'TransferLearningFWI.py'),'exec'),ns)
        cases.append(dict(case=c,wall_seconds=time.perf_counter()-ts))
        (OUT/'case_times.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
        del ns;torch.cuda.empty_cache()
    print(f'TOTAL_COMPUTE_SECONDS {time.time()-START:.3f}',flush=True)
