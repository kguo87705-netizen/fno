"""Receiver amplitude and weighted wrapped spectral-phase audit of saved train shot."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def audit(prediction, reference, dt):
    p=np.fft.rfft(prediction,axis=-1);r=np.fft.rfft(reference,axis=-1)
    weight=np.abs(r)**2;weight[:,0]=0
    mask=np.abs(r)>=.01*np.abs(r).max()
    mask[:,0]=False
    phase=np.angle(p*np.conj(r));valid_weight=weight*mask
    phase_rms=np.sqrt(np.sum(valid_weight*phase**2)/np.sum(valid_weight))
    amplitude_error=np.sqrt(np.sum((np.abs(p)-np.abs(r))**2)/np.sum(np.abs(r)**2))
    freq=np.fft.rfftfreq(reference.shape[-1],dt)
    energy=weight.sum(0)
    return {'spectral_amplitude_relative_l2':float(amplitude_error),
            'reference_energy_weighted_wrapped_phase_rms_degrees':float(phase_rms*180/np.pi),
            'phase_reference_amplitude_threshold_fraction':.01,
            'phase_included_reference_energy_fraction':float(valid_weight.sum()/weight.sum()),
            'dominant_reference_frequency_MHz':float(freq[np.argmax(energy)]/1e6),
            'reference_energy_below_1p05MHz_fraction':float(energy[freq<=1.05e6].sum()/energy.sum()),
            'prediction_norm_ratio':float(np.linalg.norm(prediction)/np.linalg.norm(reference)),
            'caution':'Wrapped full-record spectral phase is not travel-time error; weak predicted amplitude makes phase less informative. No time shift fitted.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    d=torch.load(args.run/'example.pt',weights_only=True,map_location='cpu')
    report=audit(d['prediction_traces'].numpy(),d['reference_traces'].numpy(),30e-9)
    (args.run/'receiver_spectrum.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
