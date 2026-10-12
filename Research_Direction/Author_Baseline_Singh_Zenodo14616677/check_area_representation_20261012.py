"""Frozen geometry-only area representation checks; zero physical solves."""
from pathlib import Path
import json
import time
import traceback
import numpy as np
import torch
from scipy.integrate import quad
from ellipse_area_fraction import ellipse_fraction,node_cell_edges,aggregate_fine_area
from run_tx_design_value_20261012 import mask_for,sha,dump

CODE=Path(__file__).resolve().parent;ROOT=CODE.parents[1]
UP=ROOT/'.build/geometry-rasterization-audit-20261012-r2'
OUT=ROOT/'.build/area-representation-20261012'


def main():
    start=time.perf_counter()
    if OUT.exists(): raise FileExistsError('Never overwrite earlier output')
    OUT.mkdir(parents=True)
    inputs=[Path(__file__),CODE/'ellipse_area_fraction.py',CODE/'run_tx_design_value_20261012.py',
        CODE/'settings2D.csv',CODE/'FiniteDifferencePyTorchConv.py',UP/'summary.json',UP/'protocol.json']
    protocol=dict(status='frozen_before_evaluation',question='Does analytic cell area representation remove certified geometry aliases while conserving area and agreeing under nested refinement?',
        candidate_geometry='three prior centers and all 12 certified axis aliases, development only',grid=[255,127],domain_m=[.1,.05],
        representation='exact branch-split ellipse/rectangle integration in node-centered Voronoi control cells, clipped at domain',
        refinements=[1,3,9],reason_for_odd_refinement='nested control-cell boundaries allow exact area aggregation',
        checks=dict(relative_area_error_max=1e-10,nested_aggregate_fraction_max_abs=1e-8,
            independent_quad_fraction_max_abs=1e-8,all_12_aliases_change_fraction_and_float32_material=True),
        independent_reference='scipy adaptive numerical integration, up to 20 boundary cells per base geometry, preselected evenly by index',
        independent_quad=dict(epsabs=1e-18,epsrel=1e-11,limit=200),
        material_probe='gamma32 = 1+(0.5-1)*area_fraction32; diagnostic only, no forward solver use',
        scientific_limit='representation convergence only; no wave-grid convergence, acoustic effective-medium validation, continuous inverse precision or active acquisition benefit',
        physical_calls=0,network_forwards=0,training_updates=0,wall_seconds_cap=180,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs})
    dump(OUT/'protocol.json',protocol)
    ledger=dict(status='running',physical_calls=0,network_forwards=0,training_updates=0)
    dump(OUT/'ledger.json',ledger)
    try:
        torch.set_num_threads(4)
        upstream=json.loads((UP/'summary.json').read_text(encoding='utf-8'))
        records=[];saved={};max_quad=0.;aliases_changed=[];all_area=[];all_nested=[]
        for h,entry in enumerate(upstream['records']):
            base=entry['center']; cases=[('base',dict(base))]
            for axis in ['x','y']:
                for sign in [-1,1]:
                    g=dict(base);g[axis]+=entry['axis_results'][axis]['plateau'][str(sign)]['certificate_offset_m']
                    cases.append((f'{axis}_{sign}',g))
            coarse0,cellarea=ellipse_fraction(base)
            hard0=mask_for(base,255,127,.1,.05).numpy()
            # Independent quadrature is deliberately not the analytic primitive.
            ex=node_cell_edges(.1,255);ey=node_cell_edges(.05,127)
            boundary=np.argwhere((coarse0>1e-12)&(coarse0<1-1e-12))
            for k in np.linspace(0,len(boundary)-1,min(20,len(boundary)),dtype=int):
                i,j=boundary[k];xl,xr=max(ex[i],base['x']-base['a']),min(ex[i+1],base['x']+base['a'])
                def overlap(x):
                    rad=base['b']*np.sqrt(max(0.,1-((x-base['x'])/base['a'])**2))
                    return max(0.,min(ey[j+1],base['y']+rad)-max(ey[j],base['y']-rad))
                # Branch-crossing points help the adaptive reference integrate kinks.
                points=[]
                for height in [ey[j],ey[j+1]]:
                    v=(height-base['y'])/base['b']
                    if abs(v)<1:
                        radius=base['a']*np.sqrt(1-v*v)
                        points.extend(t for t in [base['x']-radius,base['x']+radius] if xl<t<xr)
                integral,uncertainty=quad(overlap,xl,xr,points=sorted(set(points)),epsabs=1e-18,epsrel=1e-11,limit=200)
                max_quad=max(max_quad,abs(integral/cellarea[i,j]-coarse0[i,j]))
            for name,g in cases:
                if time.perf_counter()-start>180: raise TimeoutError('Frozen time cap exceeded')
                coarse,area=ellipse_fraction(g);target=np.pi*g['a']*g['b']
                rows=[]
                for factor in [1,3,9]:
                    f,a=(coarse,area) if factor==1 else ellipse_fraction(g,255*factor,127*factor)
                    error=abs(float(np.sum(f*a))-target)/target
                    agg=aggregate_fine_area(f,a,factor,255,127)
                    nesting=float(np.max(np.abs(agg/area-coarse)))
                    all_area.append(error);all_nested.append(nesting)
                    rows.append(dict(factor=factor,area_relative_error=error,nested_fraction_max_abs=nesting))
                hard=mask_for(g,255,127,.1,.05).numpy()
                diff=float(np.max(np.abs(coarse-coarse0)))
                material=1+(.5-1)*coarse.astype(np.float32)
                base_material=1+(.5-1)*coarse0.astype(np.float32)
                material_changes=int(np.count_nonzero(material!=base_material))
                if name!='base':
                    assert np.array_equal(hard,hard0)
                    aliases_changed.append(diff>0 and material_changes>0)
                saved[f'fraction_{h}_{name}']=coarse
                record=dict(center_index=h,label=name,geometry=g,refinement=rows,
                    fraction_max_change_vs_base=diff,float32_material_changed_cells=material_changes,
                    binary_changed_cells=int(np.count_nonzero(hard!=hard0)))
                records.append(record)
            print(f'geometry center {h}: five positions and three refinements checked',flush=True)
        passed=(max(all_area)<=1e-10 and max(all_nested)<=1e-8 and max_quad<=1e-8 and all(aliases_changed))
        summary=dict(status='completed',geometry_gate_pass=bool(passed),max_area_relative_error=max(all_area),
            max_nested_fraction_abs=max(all_nested),max_independent_quadrature_fraction_abs=max_quad,
            aliases_changed=sum(bool(x) for x in aliases_changed),aliases_checked=len(aliases_changed),records=records,
            conclusion='geometry representation checks only; solver-grid wave convergence remains untested')
        dump(OUT/'summary.json',summary)
        np.savez_compressed(OUT/'coarse_fractions.npz',**saved)
        z=np.load(OUT/'coarse_fractions.npz')
        assert len(z.files)==15 and all(np.isfinite(z[k]).all() and z[k].min()>=0 and z[k].max()<=1 for k in z.files)
        verification=dict(status='passed' if passed else 'geometry_gate_failed',
            input_hashes=all(sha(ROOT/p)==v for p,v in protocol['input_sha256'].items()),
            arrays_finite_and_bounded=True,saved_coarse_fractions=15,independent_quad_cells=60,
            geometry_gate_pass=bool(passed),wave_grid_convergence_checked=False,
            fraction_archive_sha256=sha(OUT/'coarse_fractions.npz'))
        assert verification['input_hashes']
        dump(OUT/'verification.json',verification)
        ledger.update(status='completed',wall_seconds=time.perf_counter()-start)
        dump(OUT/'ledger.json',ledger)
        print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))
    except Exception as e:
        ledger.update(status='failed',error=repr(e),wall_seconds=time.perf_counter()-start)
        dump(OUT/'ledger.json',ledger)
        dump(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),no_automatic_retries=True))
        raise


if __name__=='__main__': main()
