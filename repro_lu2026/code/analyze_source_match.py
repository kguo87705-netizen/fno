"""Read-only saved-data audit; no propagation, fitting, or threshold changes."""
import json
import hashlib
import math
from pathlib import Path
import sys
import h5py
import numpy as np


def main(config_path):
    cfg=json.loads(Path(config_path).read_text())
    run=Path(cfg['full_run']);single=Path(cfg['single_run'])
    protocol=json.loads((run/'config.json').read_text())
    cals=json.loads((run/'frozen_calibration.json').read_text())
    assert cals==json.loads((single/'calibration.json').read_text())
    assert cals['forward_difference']['lag_internal_steps']==0
    gain=cals['forward_difference']['gain']; stride=protocol['output_stride']
    audit={'frozen_calibration_identical':True,'posthoc_not_used_for_decision':True}
    with h5py.File(run/'source_matched_fmc.h5','r') as f,h5py.File(single/'single.h5','r') as one,h5py.File(protocol['reference'],'r') as ref:
        k=f['kwave_scattering'][:,:,::stride].astype(np.float64)
        d=(f['forward_difference/defect'][:,:,::stride].astype(np.float64)-f['forward_difference/uniform'][:,:,::stride].astype(np.float64))*gain
        expected=ref['fmc_internal'][:].transpose(2,1,0)-f['kwave_uniform'][:]
        audit['kwave_subtraction_exact']=bool(np.array_equal(expected,f['kwave_scattering'][:]))
        assert audit['kwave_subtraction_exact']
        audit['same_grid_model_representation']=bool(np.array_equal(ref['c'][:].T,np.repeat(np.repeat(ref['label'][:].T,2,axis=0),2,axis=1)))
        audit['single_vs_full_uniform_relative_l2']={}
        tx=protocol['calibration']['tx_one_based']-1
        for variant in ['raw_ricker','analytic_derivative','forward_difference']:
            a=f[variant+'/uniform'][tx];b=one[variant][0]
            audit['single_vs_full_uniform_relative_l2'][variant]=float(np.linalg.norm(a-b)/np.linalg.norm(b))
    error=d-k;power=float(np.sum(k*k))
    times=np.arange(k.shape[-1])*protocol['dt_s']*stride*1e6
    records=[]
    for lo,hi in zip(cfg['time_edges_us'][:-1],cfg['time_edges_us'][1:]):
        mask=(times>=lo)&(times<hi)
        ep=float(np.sum(k[...,mask]**2))
        records.append({'start_us':lo,'end_us':hi,'reference_energy_fraction':ep/power,
                        'relative_l2':float(np.linalg.norm(error[...,mask])/np.sqrt(ep))})
    audit['time_bands']=records
    ks=np.fft.rfft(k,axis=-1);es=np.fft.rfft(error,axis=-1)
    kp=np.sum(abs(ks)**2,axis=(0,1));ep=np.sum(abs(es)**2,axis=(0,1));kp[1:-1]*=2;ep[1:-1]*=2
    freq=np.fft.rfftfreq(k.shape[-1],protocol['dt_s']*stride)/1e6
    records=[]
    for lo,hi in zip(cfg['frequency_edges_mhz'][:-1],cfg['frequency_edges_mhz'][1:]):
        mask=(freq>=lo)&(freq<hi)
        records.append({'start_mhz':lo,'end_mhz':hi,'reference_energy_fraction':float(kp[mask].sum()/kp.sum()),
                        'relative_l2':float(np.sqrt(ep[mask].sum()/kp[mask].sum()))})
    audit['frequency_bands_rectangular_record']=records
    rows=[]
    cref=protocol['max_vel_m_s'];dt=protocol['dt_s']
    for c in protocol['dispersion']['speeds_m_s']:
        for f0 in protocol['dispersion']['frequencies_hz']:
            kval=2*math.pi*f0/c
            omega=2/dt*math.asin(c/cref*math.sin(cref*kval*dt/2))
            rows.append({'local_c_m_s':c,'frequency_hz':f0,'kwave_cref_m_s':cref,
                         'homogeneous_fixed_cref_phase_velocity_relative_error':omega/(2*math.pi*f0)-1})
    audit['kwave_fixed_cref_symbol']=rows
    audit['source_matched_data_sha256']=hashlib.sha256((run/'source_matched_fmc.h5').read_bytes()).hexdigest()
    (run/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,records,labels,title in [
        (axes[0],audit['time_bands'],[f"{r['start_us']}-{r['end_us']}" for r in audit['time_bands']],'Time windows (us)'),
        (axes[1],audit['frequency_bands_rectangular_record'],[f"{r['start_mhz']}-{r['end_mhz']}" for r in audit['frequency_bands_rectangular_record']],'FFT bands (MHz)')]:
        ax.bar(labels,[r['relative_l2'] for r in records],color='tab:blue')
        ax.axhline(.2,ls='--',color='tab:red',label='Full-scattering acceptance threshold')
        ax.set_title(title);ax.set_ylabel('Scattering relative L2')
        for i,r in enumerate(records):ax.text(i,r['relative_l2']+.015,f"energy {r['reference_energy_fraction']:.1%}",ha='center',fontsize=8)
        ax.set_ylim(0,max(r['relative_l2'] for r in records)*1.25)
    fig.suptitle('Posthoc diagnosis only; frozen homogeneous gain, no extra alignment')
    fig.tight_layout();fig.savefig(run/'source_match_residual_diagnostics.png',dpi=protocol['figure_dpi']);plt.close(fig)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main(sys.argv[1])
