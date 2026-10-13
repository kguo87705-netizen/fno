"""Run supplied author scripts with path/device adaptation, no retuning."""
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
OUT=ROOT/'.build/singh-pretrained-author-20261012'
WEIGHT=SUPPLIED/'model_Unet_pretrained_800'
RESULTS=[]

def before(ns):
    torch.cuda.synchronize()
    ns['initial_ready_seconds']=time.perf_counter()-CASE_START
    ns['initial_field']=ns['gammaPred'].detach().cpu().numpy().squeeze().copy()
    ns['initial_lr_actual']=ns['optimizer'].param_groups[0]['lr']
    assert ns['initial_lr_actual']==ns['lr']
    assert ns['epochs']==(35 if ns['method']=='initial_fwi' else 5)
    print('INITIAL '+json.dumps(dict(method=ns['method'],case=int(ns['case']),lr=ns['initial_lr_actual'],seconds=ns['initial_ready_seconds'])),flush=True)

def finish(ns):
    torch.cuda.synchronize()
    fit_seconds=time.perf_counter()-ns['start0']
    pred=ns['gammaPred'].detach().cpu().numpy().squeeze().copy()
    truth=ns['gamma'].cpu().numpy().squeeze()
    fields=dict(truth=truth,reconstruction=pred,initial=ns['initial_field'])
    post_seconds=0
    if ns['method']=='transfer':
        ts=time.perf_counter()
        # Supplementary only: keep author's saved pre-last-update output primary.
        with torch.no_grad():
            after=torch.ones_like(ns['gammaPred'])
            after[:,:,1:-1,1:-1]=ns['model'](ns['inputData'])
        torch.cuda.synchronize()
        fields['post5_reconstruction']=after.cpu().numpy().squeeze()
        post_seconds=time.perf_counter()-ts
    assert all(np.isfinite(a).all() for a in fields.values())
    steps=[int(s['step'].item()) for s in ns['optimizer'].state.values() if 'step' in s]
    assert steps and min(steps)==max(steps)==ns['epochs']
    row=dict(method=ns['method'],case=int(ns['case']),updates=ns['epochs'],actual_initial_lr=ns['initial_lr_actual'],
        material_mse=float(((pred[1:-1,1:-1].astype(float)-truth[1:-1,1:-1])**2).mean()),
        initial_mse=float(((ns['initial_field'][1:-1,1:-1].astype(float)-truth[1:-1,1:-1])**2).mean()),
        fit_seconds=fit_seconds,initial_ready_seconds=ns['initial_ready_seconds'],post5_forward_seconds=post_seconds)
    if ns['method']=='transfer':
        row['post5_mse']=float(((fields['post5_reconstruction'][1:-1,1:-1].astype(float)-truth[1:-1,1:-1])**2).mean())
        row['saved_field_note']='Author gammaPred precedes optimizer step 5 (four updates reflected)'
    np.savez_compressed(Path(ns['outputFolder'])/'final_fields.npz',**fields)
    ns['final_metrics']=row
    print('FINAL '+json.dumps(row),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    sys.path[:0]=[str(SUPPLIED),str(AUTHOR)]
    import NeuralNetwork as NN
    state=torch.load(WEIGHT,map_location='cpu',weights_only=True)
    model=NN.Unet([1,16,32,64,128],2,1e-5,bnorm=True)
    model.load_state_dict(state,strict=True)
    del state,model
    OUT.mkdir(parents=True,exist_ok=True)
    START=time.time();DEADLINE=START+1800
    def timeout():
        print('STOP 30-minute budget reached',flush=True);os._exit(124)
    watchdog=threading.Timer(1800,timeout);watchdog.daemon=True;watchdog.start()
    for method,filename in [('initial_fwi','conventionalFWIWithInitialGuess.py'),('transfer','TransferLearningFWI.py')]:
        source=(SUPPLIED/filename).read_text(encoding='utf-8-sig')
        source=source.replace('cases = np.linspace(1, 4, 4, dtype=int)','cases = np.array([selected_case],dtype=int)')
        for old in ['destinationFolder = "data_casestudy"','destinationFolder = "data"']:
            source=source.replace(old,'destinationFolder = outputFolder')
        source=source.replace('if destinationFolder not in os.listdir(os.getcwd()):','if not os.path.isdir(destinationFolder):')
        for stem in ['material','gradient','measurement','source.h5']:
            source=source.replace(f'destinationFolder + "/{stem}',f'inputFolder + "/{stem}')
        a=source.index('model.load_state_dict(torch.load(')
        b=source.index('map_location=device))',a)+len('map_location=device))')
        source=source[:a]+'model.load_state_dict(torch.load(pretrainedPath, map_location=device, weights_only=True))\n'+('    model.to(device)' if method=='initial_fwi' else '')+source[b:]
        indent='    ' if method=='initial_fwi' else '        '
        source=source.replace(indent+'start0 = time.perf_counter()',indent+'before(globals())\n'+indent+'start0 = time.perf_counter()')
        marker='    # Plotting and saving figures' if method=='initial_fwi' else '        ##Plotting and saving figures over iterations'
        source=source[:source.index(marker)]+indent+'finish(globals())\n'
        for c in range(1,5):
            if DEADLINE-time.time()<180:raise TimeoutError('Insufficient remaining budget')
            folder=OUT/method/f'case{c}';folder.mkdir(parents=True,exist_ok=True)
            CASE_START=time.perf_counter()
            ns=dict(__file__=str(AUTHOR/filename),__name__='__main__',selected_case=c,method=method,
                inputFolder=str(AUTHOR/'data_casestudy'),outputFolder=str(folder),pretrainedPath=str(WEIGHT),before=before,finish=finish)
            print(f'RUN {method} case={c} remaining={DEADLINE-time.time():.1f}',flush=True)
            exec(compile(source,str(SUPPLIED/filename),'exec'),ns)
            row=ns['final_metrics'];row['case_wall_seconds']=time.perf_counter()-CASE_START
            RESULTS.append(row)
            (folder/'metrics.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
            (OUT/'results.json').write_text(json.dumps(RESULTS,indent=2),encoding='utf-8')
            del ns;torch.cuda.empty_cache()
    print(f'TOTAL_COMPUTE_SECONDS {time.time()-START:.3f}',flush=True)
