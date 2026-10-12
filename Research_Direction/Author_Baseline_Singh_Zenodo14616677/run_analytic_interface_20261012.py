"""Frozen independent 1D interface-wave benchmark; no inverse fitting."""
from pathlib import Path
import json
import time
import traceback
import numpy as np
import torch
import FiniteDifferencePyTorchConv as FD
from ellipse_area_fraction import node_cell_edges
from run_tx_design_value_20261012 import sha,dump

CODE=Path(__file__).resolve().parent;ROOT=CODE.parents[1]
PLAN=ROOT/'.build/evidence-route-review-20261012/selected_minimal_experiment.json'
OUT=ROOT/'.build/analytic-interface-20261012'
METHODS=['binary','arithmetic','reciprocal']


def gaussian(x):
    return np.exp(-.5*((x-.015)/.0015)**2)


def exact_field(x,t,xi,homogeneous=False):
    incident=gaussian(x-6000*t)
    if homogeneous: return incident
    return np.where(x<xi,incident+gaussian(2*xi-x-6000*t)/3,incident*4/3)


def update(u0,u1,gamma,faces,coef):
    u2=np.empty_like(u1)
    u2[1:-1]=-u0[1:-1]+2*u1[1:-1]+coef*(faces[1:]*(u1[2:]-u1[1:-1])-faces[:-1]*(u1[1:-1]-u1[:-2]))/gamma[1:-1]
    u2[0]=u2[2];u2[-1]=u2[-3]
    return u2


def build_gamma(nx,xi,method):
    nodes=np.linspace(0,.1,nx+1)
    edges=node_cell_edges(.1,nx)
    frac=np.clip((edges[1:]-np.maximum(edges[:-1],xi))/np.diff(edges),0,1)
    material=(np.where(nodes>=xi,.5,1.) if method=='binary' else
              1-.5*frac if method=='arithmetic' else 1/(1+frac))
    return np.r_[1.,material,1.]


def solve(nx,dt,n,xi,method,homogeneous=False):
    dx=.1/nx; x=(np.arange(nx+3)-1)*dx
    gamma=np.ones(nx+3) if homogeneous else build_gamma(nx,xi,method)
    faces=2/(1/gamma[:-1]+1/gamma[1:])
    u0=exact_field(x,-dt,xi,homogeneous);u1=exact_field(x,0.,xi,homogeneous)
    j=int(np.floor(.025/dx))+1;w=.025/dx-np.floor(.025/dx)
    trace=np.empty(n)
    for step in range(n):
        u2=update(u0,u1,gamma,faces,(6000*dt/dx)**2)
        trace[step]=(1-w)*u2[j]+w*u2[j+1]
        u0,u1=u1,u2
    assert np.isfinite(trace).all()
    return trace,gamma


def main():
    started=time.perf_counter()
    if OUT.exists(): raise FileExistsError('Old results are preserved')
    OUT.mkdir(parents=True)
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    files=[Path(__file__),PLAN,CODE/'FiniteDifferencePyTorchConv.py',CODE/'ellipse_area_fraction.py',CODE/'run_tx_design_value_20261012.py']
    protocol=dict(plan=plan,status='frozen_before_any_physical_task',input_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        implementation=dict(main_dtype='numpy float64',operator_check_dtype='torch and numpy float32',
             float32_operator_max_abs_tolerance=2e-6,homogeneous_reference_finest_relative_l2_max=.05,
             binary_interface_convention='node x >= xi belongs to right material',gamma_ghosts='ones, matching original padded material convention',
             sample='output at (step+1)*dt, downsample native fine traces at factor-1::factor',
             reflection_window='8e-6 <= common output time <= 12e-6',
             initial_state='piecewise analytic sharp-interface field for all methods; residual initial difference between alias phases reported',
             performance_scope='1D float64 benchmark; no GPU speed or 2D accuracy claim'))
    dump(OUT/'protocol.json',protocol)
    ledger=dict(status='running',entries=[],physical_tasks_attempted=0,network_forwards=0,training_updates=0)
    dump(OUT/'ledger.json',ledger)

    def task(label,role,func,**metadata):
        if time.perf_counter()-started>180: raise TimeoutError('Frozen time cap exceeded')
        assert ledger['physical_tasks_attempted']<22
        row=dict(label=label,role=role,status='started',**metadata)
        ledger['entries'].append(row);ledger['physical_tasks_attempted']+=1
        dump(OUT/'ledger.json',ledger); t0=time.perf_counter()
        try:
            value=func();row.update(status='completed',seconds=time.perf_counter()-t0)
            return value
        except Exception as e:
            row.update(status='failed',seconds=time.perf_counter()-t0,error=repr(e));raise
        finally: dump(OUT/'ledger.json',ledger)

    try:
        torch.set_num_threads(4)
        def operator_check():
            nx,ny=31,7;dt=3e-8;dx=.1/nx;dy=.05/ny
            rng=np.random.default_rng(2026101231)
            g=rng.uniform(.5,1,nx+3).astype(np.float32)
            u0=rng.normal(0,.01,nx+3).astype(np.float32);u1=rng.normal(0,.01,nx+3).astype(np.float32)
            faces=2/(1/g[:-1]+1/g[1:])
            one=update(u0,u1,g,faces,np.float32((6000*dt/dx)**2))
            expand=lambda a:torch.from_numpy(a).reshape(1,1,-1,1).expand(1,1,nx+3,ny+3).clone()
            solver=FD.FiniteDifference(dt,dx,dy,6000.,2700.,torch.device('cpu'))
            with torch.no_grad(): actual=solver(expand(u0),expand(u1),expand(g),torch.zeros(1,1,nx+3,ny+3))[0,0,:,ny//2+1].numpy()
            error=float(np.max(np.abs(actual-one)))
            assert error<=2e-6
            return dict(max_abs=error,status='passed')
        op=task('one_step_y_invariant','operator_validation',operator_check,steps=1)
        xis=plan['interface_positions_m'];times=(np.arange(534)+1)*3e-8
        window=(times>=8e-6)&(times<=12e-6)
        exact=np.stack([exact_field(.025,times,xi) for xi in xis])
        # Earliest reflected pulse returning from either boundary is after this window.
        left_return=(2*min(xis)-.015+.025)/6000
        right_return=(.1-.015+.1-.025)/6000
        assert min(left_return,right_return)>12e-6
        traces=np.empty((3,3,2,534));hom=np.empty((3,534));native={};materials={};refs=[]
        for k,grid in enumerate(plan['grids']):
            nx,dt,n,factor=grid['Nx'],grid['dt_s'],grid['N'],grid['factor']
            trace,g=task(f'homogeneous_r{factor}','homogeneous_reference',
                lambda:solve(nx,dt,n,xis[0],'binary',True),Nx=nx,steps=n)
            common=trace[factor-1::factor];hom[k]=common
            refs.append(float(np.linalg.norm(common-gaussian(.025-6000*times))/np.linalg.norm(gaussian(.025-6000*times))))
            native[f'homogeneous_r{factor}']=trace
            for j,method in enumerate(METHODS):
                for h,xi in enumerate(xis):
                    trace,g=task(f'{method}_r{factor}_phase{h}','interface_wave',
                        lambda:solve(nx,dt,n,xi,method),Nx=nx,steps=n,method=method,phase=h)
                    traces[k,j,h]=trace[factor-1::factor]
                    native[f'{method}_r{factor}_phase{h}']=trace
                    materials[f'{method}_r{factor}_phase{h}']=g
            print(f'grid factor {factor}: reference and six interface solves completed',flush=True)
        assert refs[-1]<.05
        reflection=np.linalg.norm(traces[:,:,:,window]-exact[None,None,:,window],axis=-1)/np.linalg.norm(exact[:,window],axis=-1)[None,None,:]
        true_difference=exact[1]-exact[0]
        predicted_difference=traces[:,:,1]-traces[:,:,0]
        difference=np.linalg.norm(predicted_difference[:,:,window]-true_difference[None,None,window],axis=-1)/np.linalg.norm(true_difference[window])
        base_improvement=1-float(reflection[0,1].max()/reflection[0,0].max())
        gates=dict(finest_area_reflection=bool(reflection[-1,1].max()<=.05),
            finest_area_difference=bool(difference[-1,1]<=.1),coarse_area_difference=bool(difference[0,1]<=.2),
            coarse_area_vs_binary=bool(base_improvement>=.2))
        metrics=[]
        for k,factor in enumerate([1,2,4]):
            metrics.append(dict(factor=factor,methods={method:dict(reflection_relative_l2_by_phase=reflection[k,j].tolist(),
                shift_difference_relative_l2=float(difference[k,j])) for j,method in enumerate(METHODS)}))
        initial_x=(np.arange(258)-1)*(.1/255)
        initial_delta=float(np.linalg.norm(exact_field(initial_x,0,xis[1])-exact_field(initial_x,0,xis[0])))
        result=dict(status='completed',scientific_gate_pass=bool(all(gates.values())),gates=gates,
            metrics=metrics,coarse_worst_phase_area_reflection_improvement=base_improvement,
            homogeneous_reference_relative_l2=refs,operator_check=op,
            exact_shift_difference_l2=float(np.linalg.norm(true_difference[window])),
            exact_base_reflection_l2=float(np.linalg.norm(exact[0,window])),
            sharp_initial_state_alias_phase_difference_l2=initial_delta,
            earliest_boundary_return_s=dict(left=left_return,right=right_return),
            interpretation='independent 1D sharp-interface displacement benchmark, fixed phases and window; no 2D or inverse accuracy claim')
        dump(OUT/'summary.json',result)
        np.savez_compressed(OUT/'traces.npz',times=times,window=window,exact=exact,traces=traces,homogeneous=hom,
            exact_shift_difference=true_difference,reflection_errors=reflection,difference_errors=difference)
        np.savez_compressed(OUT/'native_traces.npz',**native)
        np.savez_compressed(OUT/'materials.npz',**materials)
        z=np.load(OUT/'traces.npz');assert np.array_equal(z['traces'],traces)
        checks=dict(status='passed',input_hashes=all(sha(ROOT/p)==v for p,v in protocol['input_sha256'].items()),
            operator_check=op,homogeneous_reference_pass=bool(refs[-1]<.05),
            boundary_window_pass=True,saved_arrays_match=True,physical_tasks=ledger['physical_tasks_attempted'],
            failed_physical_tasks=sum(row['status']=='failed' for row in ledger['entries']),
            artifact_sha256={f:sha(OUT/f) for f in ['traces.npz','native_traces.npz','materials.npz','summary.json']})
        assert checks['input_hashes'] and ledger['physical_tasks_attempted']==22
        dump(OUT/'verification.json',checks)
        ledger.update(status='completed',wall_seconds=time.perf_counter()-started,
            physical_compute_seconds=sum(row['seconds'] for row in ledger['entries']),total_grid_time_steps=sum(row['steps'] for row in ledger['entries']))
        dump(OUT/'ledger.json',ledger)
        print(json.dumps(result,indent=2))
    except Exception as e:
        ledger.update(status='failed',error=repr(e),wall_seconds=time.perf_counter()-started)
        dump(OUT/'ledger.json',ledger)
        dump(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),no_automatic_retries=True))
        raise


if __name__=='__main__':main()
