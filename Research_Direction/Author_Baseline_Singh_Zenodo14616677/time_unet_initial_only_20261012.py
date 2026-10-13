"""Requested zero-iteration U-Net baseline, using supplied saved gradients."""
import os
os.environ['OMP_NUM_THREADS']='4'
import sys,time,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
SUPPLIED=BASE/'singh_zenodo'
AUTHOR=SUPPLIED/'singh_zenodo'
OUT=ROOT/'.build/singh-pretrained-author-20261012'
sys.path.insert(0,str(SUPPLIED))
import NeuralNetwork as NN
torch.set_num_threads(4)
device=torch.device('cuda')
rows=[]
for c in range(1,5):
    start=time.perf_counter()
    torch.manual_seed(99)
    model=NN.Unet([1,16,32,64,128],2,1e-5,bnorm=True)
    model.load_state_dict(torch.load(SUPPLIED/'model_Unet_pretrained_800',map_location=device,weights_only=True))
    model.to(device)
    gradient=torch.tensor(pd.read_hdf(AUTHOR/f'data_casestudy/gradient{c}.h5').values).reshape(1,1,256,128)
    x=(gradient-gradient.amin((2,3),keepdim=True))/(gradient.amax((2,3),keepdim=True)-gradient.amin((2,3),keepdim=True))*2-1
    x=x.to(device)
    torch.cuda.synchronize();infer_start=time.perf_counter()
    with torch.no_grad():y=model(x)
    torch.cuda.synchronize()
    infer_seconds=time.perf_counter()-infer_start
    pred=np.ones((258,130),dtype=np.float32);pred[1:-1,1:-1]=y.cpu().numpy().squeeze()
    total=time.perf_counter()-start
    existing=np.load(OUT/f'initial_fwi/case{c}/final_fields.npz')
    delta=float(np.max(abs(pred-existing['initial'])))
    assert delta<1e-6
    rows.append(dict(case=c,inference_seconds=infer_seconds,initial_online_seconds=total,max_difference=delta,
        excludes='saved initial adjoint-gradient generation and offline pretraining; no FWI iterations'))
(OUT/'initial_only_timing.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2))
