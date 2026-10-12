"""Constructive geometry alias certificates; zero forward solves."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from run_tx_design_value_20261012 import mask_for, sha, dump

CODE=Path(__file__).resolve().parent
ROOT=CODE.parents[1]
OUT=ROOT/'.build/geometry-rasterization-audit-20261012-r2'


def main():
    started=time.perf_counter()
    if OUT.exists(): raise FileExistsError('Old results are preserved')
    OUT.mkdir(parents=True)
    cfg=pd.read_csv(CODE/'settings2D.csv').iloc[0]
    nx,ny=int(cfg['Nx']),int(cfg['Ny']); lx,ly=float(cfg['Lx']),float(cfg['Ly'])
    dx,dy=lx/nx,ly/ny
    files=[Path(__file__),CODE/'run_tx_design_value_20261012.py',CODE/'settings2D.csv',CODE/'FiniteDifferencePyTorchConv.py']
    protocol=dict(status='frozen_before_audit',question='Can distinct continuous ellipse coordinates produce exactly the same discrete material and therefore identical forward distributions?',
        motivation='exploratory center mask preflight already found x+.01 cell and y+.1 cell aliasing; not held-out discovery',
        centers=[.038,.05,.062],center_y=.028,ellipse_axes=[.006,.003],gamma=.5,
        axes=['x','y'],grid=[nx,ny],spacing_m=[dx,dy],dtype='original torch float32 binary mask',
        scan_positive_fractions=[.0001,.001,.01,.1,.5,1.],scan_negative_fractions=[-.0001,-.001,-.01,-.1,-.5,-1.],
        plateau_search='nearest ideal float64 ellipse/grid crossing used only to bracket; test original float32 mask; 60 bisections; endpoints classified by actual original mask',
        certificate='axis offsets at 80 percent of positive/negative first-change bisection lower bounds; same mask and same full padded gamma tensor',
        no_continuous_resolution_claim='plateau is representation aliasing, not acoustic resolution; no CRLB or physical identifiability bound inferred',
        physical_calls=0,network_forwards=0,training_updates=0,wall_seconds_cap=60,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    dump(OUT/'protocol.json',protocol)
    torch.set_num_threads(4)
    records=[]; arrays={}; verify=[]
    xgrid=torch.linspace(0,lx,nx+1).numpy().astype(float)
    ygrid=torch.linspace(0,ly,ny+1).numpy().astype(float)
    for h,xc in enumerate(protocol['centers']):
        g=dict(x=xc,y=.028,a=.006,b=.003)
        base=mask_for(g,nx,ny,lx,ly)
        arrays[f'mask_{h}']=base.numpy()
        record=dict(center=g,base_area_cells=int(base.sum()),axis_results={})
        for axis,step in [('x',dx),('y',dy)]:
            def moved(delta):
                t=dict(g);t[axis]+=delta
                return mask_for(t,nx,ny,lx,ly)
            other=(ygrid-g['y'])/g['b'] if axis=='x' else (xgrid-g['x'])/g['a']
            active=other[np.abs(other)<=1]
            radii=(g['a'] if axis=='x' else g['b'])*np.sqrt(1-active**2)
            grid=xgrid if axis=='x' else ygrid
            roots=np.concatenate([(grid[:,None]+radii[None,:]).ravel(),(grid[:,None]-radii[None,:]).ravel()])-g[axis]
            plateau={}
            for sign in [-1,1]:
                candidates=sign*roots[sign*roots>0]
                ideal=float(candidates.min())
                hi=ideal*2+step*1e-5
                while torch.equal(moved(sign*hi),base):
                    hi*=2
                    if hi>step: raise ValueError('No unambiguous local bracket')
                lo=0.
                for _ in range(60):
                    mid=(lo+hi)/2
                    if torch.equal(moved(sign*mid),base): lo=mid
                    else: hi=mid
                assert torch.equal(moved(sign*lo),base) and not torch.equal(moved(sign*hi),base)
                certificate_offset=sign*.8*lo
                cm=moved(certificate_offset)
                assert certificate_offset!=0 and torch.equal(cm,base)
                basegamma=torch.ones(1,1,nx+3,ny+3)
                basegamma[0,0,1:-1,1:-1]=torch.where(base,torch.tensor(.5),torch.tensor(1.))
                certgamma=torch.ones_like(basegamma)
                certgamma[0,0,1:-1,1:-1]=torch.where(cm,torch.tensor(.5),torch.tensor(1.))
                assert torch.equal(basegamma,certgamma)
                arrays[f'certificate_mask_{h}_{axis}_{sign}']=cm.numpy()
                plateau[str(sign)]=dict(ideal_crossing_m=ideal,same_mask_lower_m=lo,changed_mask_upper_m=hi,
                    certificate_offset_m=certificate_offset,certificate_same_mask=True,certificate_same_padded_material=True)
                verify.append(True)
            scan=[]
            for frac in protocol['scan_negative_fractions']+protocol['scan_positive_fractions']:
                m=moved(frac*step)
                scan.append(dict(fraction=frac,offset_m=frac*step,changed_cells=int((m!=base).sum()),area_cells=int(m.sum())))
            span=plateau['-1']['same_mask_lower_m']+plateau['1']['same_mask_lower_m']
            record['axis_results'][axis]=dict(plateau=plateau,same_mask_axis_span_m=span,scan=scan)
        records.append(record)
    np.savez_compressed(OUT/'mask_certificates.npz',**arrays)
    result=dict(status='completed',records=records,distinct_coordinate_same_input_certificates=len(verify),
        conclusion='binary rasterization is locally constant in ellipse position; aliased coordinates have exactly identical forward distribution even with known material and no noise',
        implication='do not interpret the current representation as continuous position identifiability; first audit representation and grid convergence',
        scope='axis-wise local neighborhoods at three declared centers, existing float32 solver inputs; not a joint rectangular plateau or hardware resolution result')
    dump(OUT/'summary.json',result)
    saved=np.load(OUT/'mask_certificates.npz')
    for h in range(3):
        for axis in ['x','y']:
            for sign in [-1,1]: assert np.array_equal(saved[f'mask_{h}'],saved[f'certificate_mask_{h}_{axis}_{sign}'])
    checks=dict(status='passed',certificates=len(verify),saved_masks_equal=True,
        original_mask_same_and_changed_bisection_endpoints=True,input_hashes=all(sha(ROOT/p)==v for p,v in protocol['input_sha256'].items()),
        forward_calls_not_needed='identical padded material, source, grid and initial state give identical deterministic solver input; no forward pass executed')
    assert checks['input_hashes']
    dump(OUT/'verification.json',checks)
    dump(OUT/'ledger.json',dict(status='completed',new_physical_calls=0,network_forwards=0,training_updates=0,wall_seconds=time.perf_counter()-started))
    print(json.dumps(dict(plateau_spans_m=[dict(center_x=r['center']['x'],**{axis:r['axis_results'][axis]['same_mask_axis_span_m'] for axis in ['x','y']}) for r in records],verification=checks),indent=2))


if __name__=='__main__': main()
