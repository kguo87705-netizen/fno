"""Frozen default/IEEE evaluation, all endpoints repeated; no optimization."""
import argparse
import json
import time
from pathlib import Path
import torch
from run_contrast_derivative_audit import cost, metrics, summarize, dump, ROOT, BASE, WIDE
from run_fno_local_contrast import samples
from basic_fno import BasicFNO
from time_feature_input import encode_time_features
from time_reconstruction import cubic_time_reconstruction
from run_reduced_material_scan import sha

OLD=Path('.build/fno-contrast-derivative-20261009')
OUT=Path('.build/fno-contrast-precision-control-20261009')
PLAN=OLD/'next_precision_protocol.json'

def flags():
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_precision=torch.get_float32_matmul_precision())

def report(raw):
    result={'arms':{},'center_policy_change':[],'elapsed_seconds':raw['elapsed_seconds']}
    for arm,states in raw['arms'].items():
        z={**raw['reference'],'methods':states,'elapsed_seconds':raw['elapsed_seconds']}
        r=summarize(z)
        for row in r['rows']:
            v=states[row['method']][str(row['state'])]
            row['repeat_max_absolute']=max(float((x-y).abs().max()) for x,y in zip(v['repeat1'],v['repeat2']))
            row['ad_forward_max_absolute']=float((v['center']-v['repeat1'][0]).abs().max())
            row['numerical_pass']=row['two_epsilon_relative_change']<=.01 and all(x['own_ad_fd_relative_error']<=.01 for x in row['responses'])
            row['science_screen_pass']=all(x['relative_l2']<=.2 and x['cosine']>=.9 for x in row['responses'])
        result['arms'][arm]=r
    for m,states in raw['arms']['default'].items():
        for k,v in states.items():
            q=raw['arms']['ieee'][m][k]['center'];delta=q-v['center']
            result['center_policy_change'].append({'method':m,'state':float(k),'absolute_l2':float(delta.norm()),
                'absolute_max':float(delta.abs().max()),'relative_full_l2':float(delta.norm()/v['center'].norm())})
    return result

def run():
    torch.set_num_threads(4)
    plan=json.loads(PLAN.read_text());prev=json.loads((OLD/'protocol.json').read_text())
    for p,h in {**plan['frozen_files_sha256'],**prev['plan']['frozen_files_sha256']}.items():assert sha(Path(p))==h,p
    for n,h in prev['dependency_sha256'].items():assert sha(ROOT/n)==h,n
    assert not OUT.exists();OUT.mkdir()
    dump(OUT/'protocol.json',{'plan':plan,'plan_sha256':sha(PLAN),'script_sha256':sha(Path(__file__)),
        'default_flags':flags(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'amendment_before_execution':'Arithmetic correction: 3 methods * 2 states * 2 arms * 5 points * 2 repeats =120 pure forward calls, plus12 AD calls, rather than60. Preserve all planned repeats and 10-minute wall-time cap; no training.',
        'prior_dependencies':prev['dependency_sha256']})
    start=time.perf_counter()
    def budget():
        if time.perf_counter()-start>600:raise TimeoutError('10-minute cap')
    reference=torch.load(OLD/'raw.pt',weights_only=True,map_location='cpu')
    manifest,groups=samples();s=groups['train'][0];scale=manifest['normalization']['wavefield_rms'];device='cuda'
    bg=torch.load('.build/operator-pilot-v3-20261008/homogeneous.pt',weights_only=True,map_location='cpu')
    support=(s['gamma']<.99).to(device);ids=s['receiver_indices'].to(device)
    background=bg['traces_full'][0:1].to(device).double();target=reference['target'].to(device)
    times=torch.arange(1,2001,device=device,dtype=torch.float64)*30e-9
    cpuids=ids.cpu()
    training=[(x['target'][0][cpuids[:,0],cpuids[:,1],:]-bg['wavefield_snapshots'][0][cpuids[:,0],cpuids[:,1],:]/scale).to(device) for x in groups['train']]
    raw={'reference':reference,'arms':{},'flags':{}}
    for arm in ['default','ieee']:
        if arm=='ieee':
            torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        raw['flags'][arm]=flags();methods={}
        for method,path in [('width4',BASE/'final.pt'),('width8',WIDE/'final.pt'),('piecewise_linear',None)]:
            model=None
            if path:
                ck=torch.load(path,weights_only=True,map_location='cpu');model=BasicFNO(width=ck['config']['width'],input_channels=8).to(device)
                model.load_state_dict(ck['state_dict']);model.eval();model.requires_grad_(False);del ck
            def proxy(a,grad=False):
                budget();av=torch.tensor(a,dtype=torch.float32,device=device,requires_grad=grad)
                with torch.set_grad_enabled(grad):
                    if model is not None:
                        prediction=model(encode_time_features(s,gamma=torch.where(support,av,1.),mode='zero')[None])[0,0]
                        sparse=prediction[ids[:,0],ids[:,1],:]
                    else:
                        lo,hi=(0,1) if a<=.5 else (1,2);w=(av-groups['train'][lo]['contrast'])/.2
                        sparse=(1-w)*training[lo]+w*training[hi]
                    pred=cubic_time_reconstruction(sparse*scale,s['time_seconds'].to(device),times)[None].double()+background
                    g=float(torch.autograd.grad(cost(pred,target),av)[0]) if grad else None
                return pred.detach().cpu(),g
            states={}
            for k,p in reference['physical'].items():
                values=[float(k)]+[a for pair in p['endpoints'] for a in pair]
                first=[proxy(a)[0] for a in values];second=[proxy(a)[0] for a in values]
                center,g=proxy(float(k),True)
                states[k]={'center':center,'gradient':g,'pairs':[(first[1],first[2]),(first[3],first[4])],
                    'endpoints':p['endpoints'],'repeat1':first,'repeat2':second,'trials':[]}
                print(arm,method,k,'AD',g,flush=True)
            methods[method]=states;del model;torch.cuda.empty_cache()
        raw['arms'][arm]=methods
    raw['elapsed_seconds']=time.perf_counter()-start
    torch.save(raw,OUT/'raw.pt');dump(OUT/'summary.json',report(raw))

def verify():
    torch.set_num_threads(4);p=json.loads((OUT/'protocol.json').read_text())
    assert sha(Path(__file__))==p['script_sha256'] and sha(PLAN)==p['plan_sha256']
    for n,h in p['plan']['frozen_files_sha256'].items():assert sha(Path(n))==h
    for n,h in p['prior_dependencies'].items():assert sha(ROOT/n)==h
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu');r=report(raw)
    assert r==json.loads((OUT/'summary.json').read_text())
    old=raw['reference']['methods'];parity=True
    for m,states in raw['arms']['default'].items():
        for k,v in states.items():
            parity &= torch.equal(v['center'],old[m][k]['center']) and v['gradient']==old[m][k]['gradient']
    dump(OUT/'verification.json',{'status':'passed','summary_recomputed':'exact with four CPU threads',
        'default_center_and_ad_old_run_parity':parity,'frozen_files_and_code':'SHA verified',
        'raw_sha256':sha(OUT/'raw.pt'),'scope':'Numerical consistency is separate from physical derivative screens'})
    print('verified; original default center/AD parity:',parity)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    try:verify() if a.verify else run()
    except (torch.cuda.OutOfMemoryError,TimeoutError) as e:
        if OUT.exists():dump(OUT/'failure.json',{'error':str(e)})
        raise
