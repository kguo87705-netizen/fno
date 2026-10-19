"""Refinement gate; optional geometric source normalization, never fitted."""
import json
from pathlib import Path
import sys
import time
import traceback
import h5py
import numpy as np
import torch
from deepwave_compare import dump,sha,metrics
from source_match import waves,simulate


def score(a,b,mask):
    x=a.astype(np.float64);y=b.astype(np.float64)
    xv=x[mask];yv=y[mask]
    xc=xv-xv.mean();yc=yv-yv.mean()
    correlations=[]
    for i in range(x.shape[0]):
        if not mask[i].any():continue
        u=x[i,mask[i]];v=y[i,mask[i]]
        u=u-u.mean();v=v-v.mean()
        correlations.append(float(np.dot(u,v)/np.sqrt(np.dot(u,u)*np.dot(v,v))))
    return {'relative_l2_fine_minus_coarse_over_coarse':float(np.linalg.norm(yv-xv)/np.linalg.norm(xv)),
            'fine_over_coarse_norm_ratio':float(np.linalg.norm(yv)/np.linalg.norm(xv)),
            'global_pearson':float(np.dot(xc,yc)/np.sqrt(np.dot(xc,xc)*np.dot(yc,yc))),
            'mean_trace_pearson':float(np.mean(correlations)),'samples':int(mask.sum())}


def main(config_path,run_path):
    started=time.perf_counter();run=Path(run_path)
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    old=json.loads(Path(cfg['previous_config']).read_text(encoding='utf-8-sig'))
    coarse_cfg=json.loads(Path(old['base_config']).read_text(encoding='utf-8-sig'));coarse_cfg.update(old)
    frozen=json.loads(Path(cfg['frozen_calibration']).read_text())['forward_difference']
    assert frozen['lag_s']==0, 'Frozen nonzero lag needs explicit resampling handling'
    assert sha(cfg['coarse_data'])==cfg['coarse_data_sha256']
    dump(run/'config.json',cfg)
    provenance={'script_sha256':sha(__file__),'configuration_sha256':sha(config_path),
                'coarse_data_sha256':sha(cfg['coarse_data']),'frozen_calibration_sha256':sha(cfg['frozen_calibration'])}
    dump(run/'provenance.json',provenance)
    with h5py.File(coarse_cfg['reference'],'r') as f:
        label=np.asarray(f['label']).T.copy()
        c=np.asarray(f['c']).T.copy()
        old_wave=np.asarray(f['source_wavelet']).reshape(-1)
    assert np.array_equal(c,np.repeat(np.repeat(label,2,axis=0),2,axis=1))
    fine_model=np.repeat(np.repeat(label,cfg['fine_label_repeat'],axis=0),cfg['fine_label_repeat'],axis=1)
    assert fine_model.shape==tuple(cfg['fine_shape'])
    # Build the requested label grid but do not propagate any defect model before the gate passes.
    uniform=np.full_like(fine_model,coarse_cfg['max_vel_m_s'])
    fine_cfg=dict(coarse_cfg)
    fine_cfg.update({'dx_m':cfg['fine_dx_m'],'dt_s':cfg['fine_dt_s'],'nt':cfg['fine_nt'],
                     'pml_width':[cfg['fine_pml_cells']]*4,'accuracy':cfg['accuracy'],
                     'batch_shots':cfg['batch_shots'],'oom_fallback_shots':cfg['oom_fallback_shots']})
    ratio=coarse_cfg['dx_m']/cfg['fine_dx_m'];assert ratio==2
    ac=dict(coarse_cfg['array'])
    for key in ['depth_index','first_lateral_index','pitch_cells']:ac[key]=int(ac[key]*ratio)
    fine_cfg['array']=ac
    t=np.arange(cfg['fine_nt'],dtype=np.float64)*cfg['fine_dt_s']
    arg=np.pi*coarse_cfg['source_frequency_hz']*(t-coarse_cfg['source_shift_s'])
    r=(1-2*arg**2)*np.exp(-arg**2)
    assert np.allclose(r[::2],old_wave,rtol=1e-11,atol=1e-13)
    wave=waves(fine_cfg,r)['forward_difference']
    compensation=1.0
    if cfg.get('source_normalization')=='fixed_2d_integrated_strength':
        assert cfg['source_reference_dx_m']==coarse_cfg['dx_m']
        compensation=cfg['source_reference_dx_m']/cfg['fine_dx_m']
        wave=(wave*compensation).astype(np.float32)
    tx=cfg['gate_transmitter_one_based']
    torch.manual_seed(coarse_cfg['seed']);np.random.seed(coarse_cfg['seed'])
    fine,detail=simulate(fine_cfg,uniform,wave,[tx])
    with h5py.File(cfg['coarse_data'],'r') as f:
        coarse=f['forward_difference/uniform'][tx-1]
        coarse_source=f['forward_difference/source'][:]
    fs=int(round(cfg['output_dt_s']/cfg['fine_dt_s']))
    cs=int(round(cfg['output_dt_s']/coarse_cfg['dt_s']))
    a=coarse[:,::cs].astype(np.float64)*frozen['gain']
    b=fine[0,:,::fs].astype(np.float64)*frozen['gain']
    assert a.shape==b.shape==(ac['count'],cfg['output_nt'])
    times=np.arange(cfg['output_nt'])*cfg['output_dt_s']
    dist=np.abs(np.arange(ac['count'])-(tx-1))*ac['pitch_cells']*cfg['fine_dx_m']
    arrivals=coarse_cfg['source_shift_s']+dist/coarse_cfg['max_vel_m_s']
    mask=np.abs(times[None,:]-arrivals[:,None])<=cfg['direct_half_window_s']
    nonself=mask.copy();nonself[tx-1,:]=False
    rc=(dist>=old['calibration']['min_distance_m']-1e-12)&(dist<=old['calibration']['max_distance_m']+1e-12)
    report={'direct_all_64':score(a,b,mask),'direct_excluding_self':score(a,b,nonself),
            'previous_direct_subset':score(a,b,mask&rc[:,None]),'full_record':score(a,b,np.ones_like(mask)),
            'frozen_gain':frozen['gain'],'frozen_lag_s':frozen['lag_s'],'runtime':detail,
            'source_scaling_audit':{
                'fine_over_coarse_h':cfg['fine_dx_m']/coarse_cfg['dx_m'],
                'geometric_compensation':compensation,
                'predicted_2d_integrated_source_ratio':compensation*cfg['fine_dx_m']/coarse_cfg['dx_m'],
                'fine_spacetime_source_l1':float(np.sum(np.abs(wave.astype(np.float64)))*cfg['fine_dt_s']*cfg['fine_dx_m']**2),
                'coarse_spacetime_source_l1':float(np.sum(np.abs(coarse_source.astype(np.float64)))*coarse_cfg['dt_s']*coarse_cfg['dx_m']**2),
                'ricker_same_continuous_definition':True,'no_area_compensation':compensation==1,'no_fitting':True}}
    report['passed']=report['direct_all_64']['relative_l2_fine_minus_coarse_over_coarse']<cfg['direct_gate_l2_strict_upper']
    report['gate_threshold_strict_upper']=cfg['direct_gate_l2_strict_upper']
    report['downstream_executed']=False
    report['code_success']=True
    with h5py.File(run/'uniform_refinement_gate.h5','x') as f:
        f.create_dataset('fine_internal',data=fine)
        f.create_dataset('fine_output',data=b)
        f.create_dataset('coarse_output',data=a)
        f.create_dataset('fine_source',data=wave)
        f.create_dataset('t_output',data=times)
        f.create_dataset('direct_mask',data=mask)
        f.attrs['source_normalization']=cfg.get('source_normalization','literal_original_formula')
    report['compute_analysis_save_s']=time.perf_counter()-started
    dump(run/'gate_result.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(cfg['diagnostic_receivers_one_based']),1,figsize=(9,9))
    for ax,rx in zip(axes,cfg['diagnostic_receivers_one_based']):
        m=mask[rx-1]
        ax.plot(times[m]*1e6,a[rx-1,m],label='Deepwave h=0.10 mm, frozen calibration')
        ax.plot(times[m]*1e6,b[rx-1,m],label=f'Deepwave h=0.05 mm, geometric factor {compensation:g}')
        ax.set_ylabel(f'Tx {tx}, Rx {rx}\npressure');ax.set_xlabel('Acquisition time (us)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Uniform-medium gate: same axes; no new amplitude or time alignment')
    fig.tight_layout();fig.savefig(run/'refinement_uniform_gate.png',dpi=cfg['figure_dpi']);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    try:main(*sys.argv[1:])
    except Exception:
        Path(sys.argv[2],'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
