"""OLS training residual only; cached recovery and two-epsilon response checks."""
import argparse,json,time
from pathlib import Path
import torch
from run_contrast_derivative_audit import cost,dump,metrics
from run_reduced_material_scan import sha
from run_training_bias_control import report as constant_report
OLD=Path('.build/fno-training-bias-control-20261009')
OUT=Path('.build/fno-training-affine-bias-control-20261009')
PLAN=OLD/'next_affine_bias_protocol.json'
CLARIFY=OLD/'next_affine_gradient_clarification.json'

def fit(c):
    a=torch.tensor([r['a'] for r in c['records']],dtype=torch.float64)
    e=torch.stack([r['residual'] for r in c['records']]);mean=a.mean();x=a-mean
    return {'mean_a':float(mean),'b0':e.mean(0),'b1':(x[:,None,None,None]*e).sum(0)/x.square().sum(),
            'actual_training_a':a,'training_only':True}
def offset(c,a):return c['b0']+(a-c['mean_a'])*c['b1']

def report(raw):
    result=constant_report(raw['constant_raw']);old=raw['constant_raw']['reference'];c=raw['affine']
    for key,grid in old['windows'].items():
        target=old['targets'][key];ps={str(a):old['predictions']['width8'][str(a)]-offset(c,a) for a in grid}
        values=[float(cost(ps[str(a)],target)) for a in grid]
        i=min(range(len(grid)),key=lambda i:(values[i],grid[i]));a=grid[i];err=ps[str(a)]-target
        physical=old['predictions']['physical'][str(a)];pe=physical-target;oe=ps[key]-target
        result['curves'][key]['affine_width8']=values
        result['rows'].append({'method':'affine_width8','target':float(key),'selected_a':a,'boundary_minimum':i in [0,len(grid)-1],
            'absolute_contrast_error':abs(a-float(key)),'material_half_mse':.5*(a-float(key))**2*old['support_fraction'],
            'selected_proxy_cost':values[i],'physical_cost':float(cost(physical,target)),
            'physical_cost_over_initial_a05':float(cost(physical,target)/cost(old['initial_physical'],target)),
            'selected_full_receiver_error':float(err.norm()/target.norm()),'selected_scatter_receiver_error':float(err.norm()/(target-old['background']).norm()),
            'physical_full_receiver_error':float(pe.norm()/target.norm()),'physical_scatter_receiver_error':float(pe.norm()/(target-old['background']).norm()),
            'true_input_full_receiver_error':float(oe.norm()/target.norm()),'true_input_scatter_receiver_error':float(oe.norm()/(target-old['background']).norm())})
    responses=[];precision=raw['precision']
    for key,v in precision['arms']['ieee']['width8'].items():
        phys=precision['reference']['physical'][key];js=[];checks=[]
        for idx,eps in enumerate([.001,.0005]):
            lo,hi=v['endpoints'][idx];pl=v['pairs'][idx][0]-offset(c,lo);ph=v['pairs'][idx][1]-offset(c,hi)
            jp=(ph-pl)/(hi-lo);jr=(phys['pairs'][idx][1]-phys['pairs'][idx][0])/(hi-lo);js.append(jp)
            analytic=(v['pairs'][idx][1]-v['pairs'][idx][0])/(hi-lo)-c['b1']
            checks.append({'epsilon':eps,**metrics(jp,jr),'slope_identity_max_absolute':float((jp-analytic).abs().max()),
                'own_loss_central_difference':float((cost(ph,precision['reference']['target'])-cost(pl,precision['reference']['target']))/(hi-lo))})
        responses.append({'state':float(key),'checks':checks,'two_epsilon_relative_change':metrics(js[0],js[1])['relative_l2'],
                          'science_pass':all(x['relative_l2']<=.2 and x['cosine']>=.9 for x in checks)})
    result.update(responses=responses,elapsed_seconds=raw['elapsed_seconds'],new_network_calls=0,new_physical_calls=0,
                  b1_norm=float(c['b1'].norm()),scope='Training-only affine receiver calibration; development-only known-support recovery. No exact objective AD computed.')
    return result

def run():
    torch.set_num_threads(4);plan=json.loads(PLAN.read_text());assert not OUT.exists()
    prior=json.loads((OLD/'protocol.json').read_text());frozen={**prior['frozen_sha256'],**plan['frozen_sha256']}
    for p,h in frozen.items():assert sha(Path(p))==h,p
    OUT.mkdir();dump(OUT/'protocol.json',{'status':'fixed_before_fit','plan':plan,'plan_sha256':sha(PLAN),
        'clarification':json.loads(CLARIFY.read_text(encoding='utf-8-sig')),'clarification_sha256':sha(CLARIFY),
        'script_sha256':sha(Path(__file__)),'constant_report_script_sha256':sha(Path(__file__).with_name('run_training_bias_control.py')),
        'frozen_sha256':frozen,'execution':'Cached float64 tensors, exact OLS with actual training a, four CPU threads, coefficient1'})
    start=time.perf_counter();c=torch.load(OLD/'calibration.pt',weights_only=True,map_location='cpu');affine=fit(c)
    torch.save(affine,OUT/'affine.pt');dump(OUT/'affine_commit.json',{'sha256':sha(OUT/'affine.pt'),'saved_before_development_load':True})
    raw={'affine':affine,'constant_raw':torch.load(OLD/'raw.pt',weights_only=True,map_location='cpu'),
         'precision':torch.load('.build/fno-contrast-precision-control-20261009/raw.pt',weights_only=True,map_location='cpu')}
    raw['elapsed_seconds']=time.perf_counter()-start
    torch.save(raw,OUT/'raw.pt');dump(OUT/'summary.json',report(raw))
def verify():
    torch.set_num_threads(4);p=json.loads((OUT/'protocol.json').read_text())
    assert sha(PLAN)==p['plan_sha256'] and sha(CLARIFY)==p['clarification_sha256'] and sha(Path(__file__))==p['script_sha256']
    assert sha(Path(__file__).with_name('run_training_bias_control.py'))==p['constant_report_script_sha256']
    for n,h in p['frozen_sha256'].items():assert sha(Path(n))==h
    c=torch.load(OLD/'calibration.pt',weights_only=True,map_location='cpu');expected=fit(c)
    raw=torch.load(OUT/'raw.pt',weights_only=True,map_location='cpu')
    assert all(torch.equal(expected[k],raw['affine'][k]) for k in ['b0','b1','actual_training_a'])
    assert sha(OUT/'affine.pt')==json.loads((OUT/'affine_commit.json').read_text())['sha256']
    assert report(raw)==json.loads((OUT/'summary.json').read_text())
    for key,grid in raw['constant_raw']['reference']['windows'].items():
        for a in grid:
            assert torch.equal(raw['affine']['b0']+(a-raw['affine']['mean_a'])*torch.zeros_like(raw['affine']['b1']),c['bias'])
    dump(OUT/'verification.json',{'status':'passed','OLS_and_metrics':'exactly recomputed','training_commit':'SHA passed before development load',
        'zero_slope':'exact constant bias reproduction','frozen_files':'SHA unchanged','raw_sha256':sha(OUT/'raw.pt'),
        'limitation':'No exact objective AD available; response and objective derivatives are finite differences only'})
    print('Verification passed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args();verify() if a.verify else run()
