"""Independent retrospective audit; immutable previous artifacts; four frozen forwards."""
import json,time,hashlib,inspect
from pathlib import Path
import torch
from run_paired_depth_coverage import policy,data_setup,predict,BasicFNO,state_hash,tensor_hash
ROOT=Path('D:/下载/FNO_Research_Workspace');OUT=ROOT/'.build/route-evidence-audit-20261011'
CODE=ROOT/'Research_Direction/Author_Baseline_Singh_Zenodo14616677'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def load(p):return torch.load(p,weights_only=True,map_location='cpu')
def spectral(x,dt):
    x=x.double();n=x.shape[-1];z=torch.fft.rfft(x,dim=-1,norm='ortho');w=torch.full((z.shape[-1],),2.,dtype=torch.float64);w[0]=1
    if n%2==0:w[-1]=1
    e=(z.abs().square()*w).sum(tuple(range(x.ndim-1)));total=float(x.square().sum());assert abs(float(e.sum())-total)<=1e-10*max(total,1)
    f=torch.fft.rfftfreq(n,dt);edges=[0,416666.6666667,1666666.6666667,float('inf')]
    return {'total':total,'bands':[float(e[(f>=lo)&(f<hi)].sum()) for lo,hi in zip(edges[:-1],edges[1:])],'n':n,'dt':dt}
def run():
    policy();assert not OUT.exists();OUT.mkdir()
    folders=['fno-matched-material-replay-20261009','fno-loss-ablation-20261010','fno-adam-state-control-20261010','fno-lowfreq-field-control-20261011','fno-lowfreq-field-control-r2-20261011','fno-coverage-diagnostic-20261011','fno-adam-direction-audit-20261010']
    files=[p for n in folders for p in (ROOT/'.build'/n).glob('*') if p.is_file()]+list(CODE.glob('run_*control*.py'))+[CODE/'run_coverage_diagnostic.py',CODE/'run_adam_direction_audit.py',Path(__file__)]
    frozen={str(p):sha(p) for p in files}
    dump(OUT/'protocol.json',{'status':'frozen_before_recomputation','budget':{'training':0,'physical':0,'forward':4,'seconds_cap':120},'purpose':'Independent provenance/checkpoint/saved-metric audit,plus same-Hz field and receiver spectra on two final R2 models at two used development depths. No threshold tuning.','bands_hz':[0,416666.6666667,1666666.6666667,'Nyquist'],'frozen_sha256':frozen})
    start=time.perf_counter();records={};checkpoint_audit={}
    for n in folders:
        d=ROOT/'.build'/n
        checkpoint_audit[n]={}
        for p in d.glob('*update*.pt'):
            ck=load(p)
            if 'optimizer_state_dict' in ck:
                checkpoint_audit[n][p.name]={'claimed_update':ck.get('update'),'actual_adam_steps':sorted({float(s['step']) for s in ck['optimizer_state_dict']['state'].values()}),'weight_hash':ck.get('state_hash')}
        if (d/'raw.pt').exists() and 'arms' in (r:=load(d/'raw.pt')):
            records[n]={a:{'field':[v['history'][-1]['evaluation']['records'][k]['scatter_field_relative_l2'] for k in ['depth-1','depth1']],'trace':[v['history'][-1]['evaluation']['records'][k]['scatter_trace_relative_l2'] for k in ['depth-1','depth1']],'geometry':v['history'][-1]['evaluation']['response'],'material':[q['relative_l2'] for q in v['material_response'][-1]['rows']],'loss_count':len(v['losses']),'slots':v['slot_exposures']} for a,v in r['arms'].items()}
    comparisons={}
    for n,c,t in [(folders[1],'loss_original','loss_field_only'),(folders[2],'loss_stateful','loss_reset'),(folders[4],'loss_original','loss_lowfreq')]:
        a,b=records[n][c],records[n][t]
        comparisons[n]={'field_relative_improvement':[1-y/x for x,y in zip(a['field'],b['field'])],'field_twenty_percent_pass':all(y<=.8*x for x,y in zip(a['field'],b['field'])),'geometry_pass':b['geometry']['relative_l2']<=.2 and b['geometry']['cosine']>=.9,'material_ten_percent_guard':b['material'][-1]<=1.1*a['material'][-1],'trace_ten_percent_guard':all(y<=1.1*x for x,y in zip(a['trace'],b['trace']))}
    device,nom,bg,scale,sr,samples=data_setup();dt=float((nom['time_seconds_snapshots'][1:]-nom['time_seconds_snapshots'][:-1]).double().mean());assert abs(dt-120e-9)<1e-10
    spectra={};d=ROOT/'.build'/folders[4]
    for name in ['loss_original','loss_lowfreq']:
        ck=load(d/f'{name}_update400.pt');model=BasicFNO(width=8,input_channels=8).to(device);model.load_state_dict(ck['state_dict']);model.eval();before=state_hash(model);saved=load(d/f'{name}_results.pt')['history'][-1]['evaluation'];spectra[name]={}
        for key in ['depth-1','depth1']:
            assert time.perf_counter()-start<120
            p=predict(model,{**nom,'gamma':samples[key]['gamma']},device,scale,bg);ref=samples[key]['physical'];assert tensor_hash(p['field'])==saved['field_sha256'][key] and torch.equal(p['trace'],saved['traces'][key]);out={}
            for space,spacing,background in [('field',dt,bg['wavefield_snapshots'][0]),('trace',30e-9,bg['traces_full'][0:1])]:
                e=p[space].double()-ref[space].double();target=ref[space].double()-background.double();ee=spectral(e,spacing);tt=spectral(target,spacing)
                out[space]={'error':ee,'target_scatter':tt,'relative_l2_bands':[(x/y)**.5 if y>0 else None for x,y in zip(ee['bands'],tt['bands'])]}
                if space=='field':
                    assert list(e.shape)==[256,128,500]
                    out['regions']={}
                    for axis,label in [(0,'x'),(1,'y')]:
                        out['regions'][label]=[]
                        for sl in [slice(0,2),slice(2,-2),slice(-2,None)]:
                            index=[slice(None)]*3;index[axis]=sl;q=e[tuple(index)];out['regions'][label].append({'error_energy':float(q.square().sum()),'count':q.numel(),'mean_square':float(q.square().mean())})
                        assert abs(sum(v['error_energy'] for v in out['regions'][label])-ee['total'])<=ee['total']*1e-10
            spectra[name][key]=out;del p
        assert state_hash(model)==before;del model,ck;torch.cuda.empty_cache()
    # Old time bands used fraction of different Nyquist rates; record exact represented bins.
    actual={'trace_old_low_last_hz':99/(2000*30e-9),'field_loss_low_last_hz':24/(500*dt),'source_hz':500000,'field_shape_XYT':[256,128,500]}
    for f,h in frozen.items():assert sha(Path(f))==h
    result={'status':'completed','checkpoint_audit':checkpoint_audit,'saved_metrics':records,'corrected_comparisons':comparisons,'same_hz_spectra':spectra,'frequency_axis_correction':actual,'elapsed_seconds':time.perf_counter()-start,'new_calls':{'forward':4,'optimizer':0,'physical':0},'cost_correction':'Recent material replay/loss ablation/Adam reset/lowfreq R1/R2 =1000 training updates; R1 excluded scientifically but still costs200. Failed preflight and diagnosis costs not fully logged.'}
    dump(OUT/'results.json',result);dump(OUT/'verification.json',{'status':'passed','checks':['4 checkpoint predictions reproduce saved full-field hashes and trace bits','both spatial-axis partitions conserve error energy','one-sided FFT Parseval energy matches time-domain energy','all captured prior file hashes unchanged','comparisons recomputed from cached raw.pt,not previous decision flags'],'limitations':['No training or new physical labels','Does not retroactively prove unrecorded prior checks','Old historical hashes and missing failure records not recoverable by this audit']});print(json.dumps({'comparisons':comparisons,'axis':actual,'elapsed':result['elapsed_seconds']},indent=2))
if __name__=='__main__':run()
