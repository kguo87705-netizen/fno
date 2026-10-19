"""Source equation diagnostic. Calibration uses homogeneous single-shot only."""
import json
import math
from pathlib import Path
import sys
import time
import traceback

import h5py
import numpy as np
import torch
import deepwave
from deepwave_compare import dump, sha, stats, metrics, shifted, propagate, Monitor, reciprocity


def read_mat(path):
    with h5py.File(path, "r") as f:
        return np.asarray(f["fmc_internal"]).transpose(2, 1, 0).copy(), np.asarray(f["txids"]).reshape(-1).astype(int)


def waves(cfg, r):
    t = np.arange(len(r), dtype=np.float64)*cfg["dt_s"]
    a = np.pi*cfg["source_frequency_hz"]*(t-cfg["source_shift_s"])
    derivative = np.pi*cfg["source_frequency_hz"]*(4*a**3-6*a)*np.exp(-a*a)
    # Form the difference in float64 to avoid unnecessary cancellation in input float32.
    forward = np.diff(r.astype(np.float64), append=0)/cfg["dt_s"]
    factor = -2/(cfg["max_vel_m_s"]*cfg["dx_m"])
    return {"raw_ricker":r.astype(np.float32),
            "analytic_derivative":(factor*derivative).astype(np.float32),
            "forward_difference":(factor*forward).astype(np.float32)}


def simulate(cfg, c, wave, txids):
    ac=cfg["array"]
    loc=torch.tensor([[ac["depth_index"],ac["first_lateral_index"]+j*ac["pitch_cells"]] for j in range(ac["count"])], dtype=torch.long,device="cuda")
    v=torch.tensor(c,dtype=torch.float32,device="cuda")
    torch.cuda.reset_peak_memory_stats()
    monitor=Monitor(cfg["memory_poll_s"])
    monitor.worker.start()
    started=time.perf_counter()
    attempts=[]
    batch=cfg["batch_shots"]
    try:
        while True:
            try:
                parts=[]
                torch.cuda.synchronize()
                tick=time.perf_counter()
                for j in range(0,len(txids),batch):
                    ids=torch.tensor(np.asarray(txids[j:j+batch])-1,dtype=torch.long,device="cuda")
                    w=torch.tensor(wave,device="cuda")[None,None,:].repeat(len(ids),1,1)
                    y=propagate(cfg,v,w,loc[ids,None,:].contiguous(),loc[None,:,:].repeat(len(ids),1,1))
                    parts.append(y.cpu().numpy().copy())
                    del y
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-tick
                break
            except torch.cuda.OutOfMemoryError as exc:
                attempts.append(str(exc))
                if batch==cfg["oom_fallback_shots"]: raise
                parts=[]
                torch.cuda.empty_cache()
                batch=cfg["oom_fallback_shots"]
        arr=np.concatenate(parts)
        assert np.isfinite(arr).all() and np.max(np.abs(arr))>0
        detail={"propagation_copy_s":elapsed,"with_prepare_s":time.perf_counter()-started,
                "batch":batch,"shots":len(txids),"oom":attempts,
                "allocated_peak_bytes":torch.cuda.max_memory_allocated(),
                "reserved_peak_bytes":torch.cuda.max_memory_reserved()}
    finally:
        monitor.stop.set(); monitor.worker.join(timeout=6)
    detail["gpu_samples"]=monitor.rows
    detail["whole_card_peak_mib"]=max((x[1] for x in monitor.rows),default=None)
    return arr,detail


def calibrate(cfg, ref, x):
    ac=cfg["array"]
    tx=cfg["calibration"]["tx_one_based"]
    dist=np.abs(np.arange(ac["count"])-(tx-1))*ac["pitch_cells"]*cfg["dx_m"]
    rc=(dist>=cfg["calibration"]["min_distance_m"]-1e-12)&(dist<=cfg["calibration"]["max_distance_m"]+1e-12)
    times=np.arange(cfg["nt"])*cfg["dt_s"]
    arrival=cfg["source_shift_s"]+dist/cfg["max_vel_m_s"]
    mask=rc[:,None]&(np.abs(times[None,:]-arrival[:,None])<=cfg["calibration"]["half_window_s"])
    a=ref.astype(np.float64)
    x=x.astype(np.float64)
    best=None
    for lag in range(-cfg["calibration"]["max_lag_steps"],cfg["calibration"]["max_lag_steps"]+1):
        b=shifted(x,lag)
        gain=np.sum(a[mask]*b[mask])/np.sum(b[mask]**2)
        error=np.linalg.norm(a[mask]-gain*b[mask])/np.linalg.norm(a[mask])
        if best is None or error<best[0]: best=(error,lag,gain)
    error,lag,gain=best
    raw=np.linalg.norm(a[mask]-x[mask])/np.linalg.norm(a[mask])
    return {"gain":float(gain),"lag_internal_steps":int(lag),"lag_s":lag*cfg["dt_s"],
            "calibration_window_raw_l2":float(raw),"calibration_window_aligned_l2":float(error),
            "receiver_ids":(np.flatnonzero(rc)+1).tolist(),"samples":int(mask.sum()),
            "lag_at_search_edge":abs(lag)==cfg["calibration"]["max_lag_steps"]}


def apply_cal(x, cal):
    return shifted(x.astype(np.float64),cal["lag_internal_steps"])*cal["gain"]


def measure(ref,x):
    summary, trace_l2, trace_corr=metrics(ref,x)
    a=ref.astype(np.float64).ravel(); b=x.astype(np.float64).ravel()
    a-=a.mean(); b-=b.mean()
    summary["global_pearson"]=float(np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b)))
    if ref.ndim==3 and ref.shape[0]==ref.shape[1]:
        mask=~np.eye(ref.shape[0],dtype=bool)
        summary["off_diagonal_global_l2"]=float(np.linalg.norm((x-ref)[mask])/np.linalg.norm(ref[mask]))
    return summary


def dispersion_diagnostic(cfg,scattering):
    # 8th-order centered second derivative coefficients, as in regular_grid.py.
    co=[-205/72,8/5,-1/5,8/315,-1/560]
    rows=[]
    for c in cfg['dispersion']['speeds_m_s']:
        for f in cfg['dispersion']['frequencies_hz']:
            for deg in cfg['dispersion']['angles_deg']:
                theta=math.radians(deg)
                qs=[2*math.pi*f/c*cfg['dx_m']*math.cos(theta),2*math.pi*f/c*cfg['dx_m']*math.sin(theta)]
                eig=sum(co[0]+2*sum(co[j]*math.cos(j*q) for j in range(1,5)) for q in qs)/cfg['dx_m']**2
                omega=2/cfg['dt_s']*math.asin(c*cfg['dt_s']/2*math.sqrt(max(0,-eig)))
                rows.append({'speed_m_s':c,'frequency_hz':f,'angle_deg':deg,
                             'points_per_wavelength':c/f/cfg['dx_m'],
                             'phase_velocity_relative_error':omega/(2*math.pi*f)-1})
    spectrum=np.fft.rfft(scattering,axis=-1)
    power=np.sum(np.abs(spectrum)**2,axis=(0,1))
    power[1:-1]*=2
    freq=np.fft.rfftfreq(scattering.shape[-1],cfg['dt_s'])
    tails={str(f):float(power[freq>f].sum()/power.sum()) for f in cfg['dispersion']['frequencies_hz']}
    return {'homogeneous_infinite_grid_fd_symbol':rows,
            'kwave_scattering_rectangular_record_energy_above_hz':tails,
            'note':'Analytical dispersion relative to continuum; not an attribution of measured interface or PML errors.'}


def plot_single(cfg,run,ref,outputs,cals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    receivers=cfg["plot_receivers_one_based"]
    times=np.arange(cfg["nt"])*cfg["dt_s"]
    tx=cfg["calibration"]["tx_one_based"]
    fig,axes=plt.subplots(len(receivers),2,figsize=(11,9))
    for row,rx in enumerate(receivers):
        center=cfg["source_shift_s"]+abs(rx-tx)*cfg["array"]["pitch_cells"]*cfg["dx_m"]/cfg["max_vel_m_s"]
        mask=abs(times-center)<=cfg["plot_half_window_s"]
        for col in range(2):
            ax=axes[row,col]
            ax.plot(times[mask]*1e6,ref[rx-1,mask],label='k-Wave',lw=1.2)
            for name in ['analytic_derivative','forward_difference']:
                y=outputs[name][0] if col==0 else apply_cal(outputs[name][0],cals[name])
                ax.plot(times[mask]*1e6,y[rx-1,mask],label=name,lw=.8,alpha=.85)
            ax.set_ylabel(f'Tx {tx}, Rx {rx}\npressure')
            ax.set_xlabel('Acquisition time (us)')
    axes[0,0].set_title('Derived amplitude and source: no fitted calibration')
    axes[0,1].set_title('Frozen homogeneous direct-wave gain + lag')
    axes[0,1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(run/'source_match_uniform.png',dpi=cfg['figure_dpi']);plt.close(fig)


def plot_full(cfg,run,kdef,ksc,ddef,dsc,cal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times=np.arange(cfg["nt"])*cfg["dt_s"]
    pairs=cfg["diagnostic_pairs_one_based"]
    fig,axes=plt.subplots(len(pairs),2,figsize=(12,10))
    da=apply_cal(ddef,cal);sa=apply_cal(dsc,cal)
    for row,(tx,rx) in enumerate(pairs):
        for col,(a,b,title) in enumerate([(kdef,da,'Full field'),(ksc,sa,'Defect minus homogeneous')]):
            ax=axes[row,col]
            ax.plot(times*1e6,a[tx-1,rx-1],label='k-Wave',lw=.8)
            ax.plot(times*1e6,b[tx-1,rx-1],label='Matched Deepwave (frozen calibration)',lw=.8,alpha=.8)
            ax.set_ylabel(f'Tx {tx}, Rx {rx}')
            ax.set_xlabel('Time (us)')
            if row==0: ax.set_title(title);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(run/'source_match_scattering.png',dpi=cfg['figure_dpi']);plt.close(fig)
    tx=cfg['calibration']['tx_one_based']-1
    stride=cfg['output_stride']; a=ksc[tx,:,::stride];b=sa[tx,:,::stride]
    limit=max(float(abs(a).max()),float(abs(b).max()))
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,data,title in zip(axes,[a,b,b-a],['k-Wave scattering','Matched Deepwave scattering','Deepwave minus k-Wave']):
        im=ax.imshow(data,aspect='auto',extent=[0,times[::stride][-1]*1e6,64,1],cmap='seismic',vmin=-limit,vmax=limit)
        ax.set_title(title);ax.set_xlabel('Time (us)');ax.set_ylabel('Receiver')
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('Same pressure scale; Tx 32; fixed homogeneous calibration')
    fig.tight_layout();fig.savefig(run/'source_match_scattering_bscan.png',dpi=cfg['figure_dpi']);plt.close(fig)


def main(path,stage,run_path):
    diag=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    cfg=json.loads(Path(diag['base_config']).read_text(encoding='utf-8-sig'));cfg.update(diag)
    run=Path(run_path)
    dump(run/'config.json',cfg)
    dump(run/'provenance.json',{'code_sha256':sha(__file__),'config_sha256':sha(path),'reference_sha256':sha(cfg['reference'])})
    assert sha(cfg['reference'])==cfg['reference_sha256']
    assert torch.cuda.is_available()
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    with h5py.File(cfg['reference'],'r') as f:
        c=np.asarray(f['c']).T.copy()
        r=np.asarray(f['source_wavelet']).reshape(-1)
    uniform=np.full_like(c,cfg['max_vel_m_s'])
    wavelets=waves(cfg,r)
    timings={}
    if stage=='single':
        ref,tx=read_mat(cfg['kwave_uniform_single'])
        assert tx.tolist()==[cfg['calibration']['tx_one_based']]
        out={};cals={};summary={}
        for name,w in wavelets.items():
            out[name],timings[name]=simulate(cfg,uniform,w,tx)
            cals[name]=calibrate(cfg,ref[0],out[name][0])
            summary[name]={'raw':measure(ref[...,::cfg['output_stride']],out[name][...,::cfg['output_stride']]),
                           'aligned':measure(ref[...,::cfg['output_stride']],apply_cal(out[name],cals[name])[...,::cfg['output_stride']]),
                           'calibration':cals[name]}
        with h5py.File(run/'single.h5','x') as f:
            f.create_dataset('kwave',data=ref)
            for name,a in out.items():f.create_dataset(name,data=a);f.create_dataset(name+'_source',data=wavelets[name])
        dump(run/'calibration.json',cals);dump(run/'single_metrics.json',summary)
        plot_single(cfg,run,ref[0],out,cals)
        print(json.dumps(summary,indent=2),flush=True)
    else:
        cals=json.loads(Path(cfg['frozen_calibration']).read_text())
        dump(run/'frozen_calibration.json',cals)
        kdef,_=read_mat(cfg['reference'])
        khom=np.zeros_like(kdef);seen=[]
        for p in [cfg['kwave_uniform_single'],cfg['kwave_uniform_remaining']]:
            with h5py.File(p,'r') as f:
                assert np.all(np.asarray(f['c'])==cfg['max_vel_m_s'])
                assert np.array_equal(np.asarray(f['source_wavelet']).reshape(-1),r)
            a,ids=read_mat(p);khom[ids-1]=a;seen.extend(ids.tolist())
        assert sorted(seen)==list(range(1,65))
        ksc=kdef-khom
        summary={};all_data={}
        ids=list(range(1,cfg['array']['count']+1))
        for name,w in wavelets.items():
            hom,detail=simulate(cfg,uniform,w,ids);timings[name+'_uniform']=detail
            if name=='raw_ricker':
                with h5py.File(cfg['previous_deepwave_raw'],'r') as f: defect=f['fmc_internal'][:]
                timings[name+'_defect']={'reused_previous':cfg['previous_deepwave_raw']}
            else:
                defect,detail=simulate(cfg,c,w,ids);timings[name+'_defect']=detail
            scattering=defect-hom
            stride=cfg['output_stride']
            result={}
            for tag,k,d in [('uniform',khom,hom),('defect',kdef,defect),('scattering',ksc,scattering)]:
                result[tag]={'raw':measure(k[...,::stride],d[...,::stride]),
                             'aligned_frozen':measure(k[...,::stride],apply_cal(d,cals[name])[...,::stride])}
            result['reciprocity']=reciprocity(defect[...,::stride])
            summary[name]=result;all_data[name]=(hom,defect)
            print(name,json.dumps(result),flush=True)
        name=cfg['primary_variant'];primary=summary[name]
        passed=primary['defect']['aligned_frozen']['trace_pearson_zero_lag']['mean']>cfg['decision']['min_mean_correlation'] and primary['scattering']['aligned_frozen']['global_relative_l2']<cfg['decision']['max_scattering_l2']
        summary['decision']={'variant':name,'passed':passed,'calibration_source':cfg['frozen_calibration'],'mean_correlation':primary['defect']['aligned_frozen']['trace_pearson_zero_lag']['mean'],'scattering_l2':primary['scattering']['aligned_frozen']['global_relative_l2']}
        norm_ratio=float(np.linalg.norm(ksc.astype(np.float64))/np.linalg.norm(kdef.astype(np.float64)))
        summary['kwave_scattering_to_full_norm_ratio']=norm_ratio
        summary['kwave_scattering_to_full_energy_ratio']=norm_ratio**2
        summary['dispersion_diagnostic']=dispersion_diagnostic(cfg,ksc)
        with h5py.File(run/'source_matched_fmc.h5','x') as f:
            f.create_dataset('kwave_uniform',data=khom)
            f.create_dataset('kwave_scattering',data=ksc)
            for name,(hom,defect) in all_data.items():
                f.create_dataset(name+'/uniform',data=hom)
                f.create_dataset(name+'/defect',data=defect)
                f.create_dataset(name+'/source',data=wavelets[name])
        dump(run/'full_metrics.json',summary)
        name=cfg['primary_variant'];hom,defect=all_data[name]
        plot_full(cfg,run,kdef,ksc,defect,defect-hom,cals[name])
        print('DECISION',json.dumps(summary['decision']),flush=True)
    dump(run/'timings.json',timings)
    dump(run/'success.json',{'stage':stage,'success':True})


if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception:
        Path(sys.argv[3],'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
