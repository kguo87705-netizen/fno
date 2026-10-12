"""Small grouped synthetic wavefield/trace dataset for operator-interface QA.

This is an I/O pilot, not sufficient evidence for neural-operator performance.
Author cases remain outside its splits. No neural model is trained here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import FiniteDifferencePyTorchConv as FiniteDifference
import Utilities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()
    assert args.count >= 8 and args.stride >= 1
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / "manifest.json").exists(), "Choose a new output path to preserve this dataset version"
    root = Path(__file__).resolve().parent
    cfg = pd.read_csv(root / "settings2D.csv").iloc[0]
    nx, ny, steps, shots = (int(cfg[k]) for k in ("Nx", "Ny", "N", "numberOfSources"))
    dx, dy, dt = float(cfg["Lx"] / nx), float(cfg["Ly"] / ny), float(cfg["dt"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = FiniteDifference.FiniteDifference(dt, dx, dy, float(cfg["c"]), float(cfg["rho"]), device)
    sx, sy = Utilities.getSourceLocations(nx, ny, int(cfg["distanceBetweenSources"]), shots)
    rx, ry = Utilities.getSensorLocations(nx, ny, int(cfg["distanceBetweenSensors"]), sx)
    force = torch.tensor(pd.read_hdf(root / "data_casestudy/source.h5").values).reshape(shots, 1, nx+3, ny+3, steps).to(device)
    zeros = torch.zeros((shots, 1, nx+3, ny+3), device=device)
    x = torch.arange(nx+1, dtype=torch.float32)*dx
    y = torch.arange(ny+1, dtype=torch.float32)*dy
    times = torch.arange(1, steps+1, dtype=torch.float64)*dt
    indices = torch.arange(args.stride-1, steps, args.stride)
    if indices[-1] != steps-1:
        indices = torch.cat([indices, torch.tensor([steps-1])])
    rng = np.random.default_rng(20261008)
    records = []
    trace_scale_square_sum, trace_scale_count = 0.0, 0
    wave_scale_square_sum, wave_scale_count = 0.0, 0
    with torch.no_grad():
        for i in range(args.count):
            # Stable across dataset expansion: existing holdouts never move to train.
            split = "train" if i % 8 < 4 else "validation" if i % 8 < 6 else "test"
            number = 1 if i % 2 == 0 else 2
            gamma = torch.ones((1,1,nx+3,ny+3),device=device)
            material = torch.ones((nx+1,ny+1))
            defects = []
            for j in range(number):
                # Disjoint left/right ranges for two defects, avoiding overlap families.
                xc = rng.uniform(.024,.043) if number==2 and j==0 else rng.uniform(.057,.076) if number==2 else rng.uniform(.025,.075)
                yc, ax, ay = rng.uniform(.013,.037), rng.uniform(.003,.007), rng.uniform(.002,.005)
                value = rng.uniform(.3,.7)
                region = ((x[:,None]-xc)/ax).square()+((y[None,:]-yc)/ay).square() <= 1
                material[region] = value
                defects.append({"x_m": float(xc), "y_m": float(yc), "axis_x_m": float(ax), "axis_y_m": float(ay), "gamma": float(value)})
            gamma[0,0,1:-1,1:-1] = material.to(device)
            wave = solver.forwardNSteps(zeros.clone(),zeros.clone(),gamma,force,nx,ny,steps,shots,device)
            traces = wave[:,rx,ry,1:].cpu().clone()
            snapshots = wave[:,1:-1,1:-1,:].index_select(-1,(indices+1).to(device)).cpu()
            del wave
            assert torch.isfinite(traces).all() and torch.isfinite(snapshots).all()
            selected = snapshots[:,np.array(rx)-1,np.array(ry)-1,:]
            assert torch.equal(selected,traces.index_select(-1,indices)), "Field-to-receiver sampling mismatch"
            # Linear interpolation diagnostic. This does not certify bandwidth-safe compression.
            coarse_times = np.concatenate(([0.0],times[indices].numpy()))
            interpolated = np.stack([np.interp(times.numpy(),coarse_times,np.concatenate(([0.0],trace[indices].numpy()))) for trace in traces.reshape(-1,steps)]).reshape(shots,len(rx),steps)
            interpolation_error = float(np.linalg.norm(interpolated-traces.numpy())/np.linalg.norm(traces.numpy()))
            group = f"synthetic_workpiece_{i:03d}"
            file = f"{group}.pt"
            torch.save({"gamma": material,"traces_full":traces,"wavefield_snapshots":snapshots,"snapshot_trace_indices":indices,"time_seconds_full":times,"time_seconds_snapshots":times[indices],"x_m":x,"y_m":y,"source_xy_m":torch.tensor([[(a-1)*dx,(b-1)*dy] for a,b in zip(sx,sy)]),"receiver_physical_indices":torch.tensor([[a-1,b-1] for a,b in zip(rx,ry)]),"receiver_xy_m":torch.tensor([[(a-1)*dx,(b-1)*dy] for a,b in zip(rx,ry)])},args.out/file)
            entry={"group_id":group,"split":split,"file":file,"defects":defects,"gamma_sha256":hashlib.sha256(material.numpy().tobytes()).hexdigest(),"traces_shape":list(traces.shape),"wavefield_shape":list(snapshots.shape),"linear_temporal_interpolation_relative_error":interpolation_error}
            records.append(entry)
            if split=="train":
                trace_scale_square_sum += float(traces.double().square().sum()); trace_scale_count += traces.numel()
                wave_scale_square_sum += float(snapshots.double().square().sum()); wave_scale_count += snapshots.numel()
            print(f"Saved {group}, {split}, interpolation error={interpolation_error:.3%}",flush=True)
    manifest={"schema_version":1,"scope":"Synthetic I/O pilot on the released scalar density-scaling solver. Four author cases excluded; all shots and future versions of a workpiece must share its split. No FNO trained.","seed":20261008,"grid_physical":[nx+1,ny+1],"grid_padded":[nx+3,ny+3],"steps":steps,"dt_seconds":dt,"snapshot_stride":args.stride,"source_file_sha256":hashlib.sha256((root/'data_casestudy/source.h5').read_bytes()).hexdigest(),"solver_sha256":hashlib.sha256((root/'FiniteDifferencePyTorchConv.py').read_bytes()).hexdigest(),"material_quantity":"dimensionless density scaling gamma","wave_quantity":"author scalar wave variable; absolute experimental calibration not established","records":records,"normalization":{"fit_split":"train","trace_mean":0,"wavefield_mean":0,"trace_rms":float(np.sqrt(trace_scale_square_sum/trace_scale_count)),"wavefield_rms":float(np.sqrt(wave_scale_square_sum/wave_scale_count))},"compression_warning":"Stored sparse snapshots are for interface testing. Full 2000-step traces retained; time compression must pass waveform, phase and gradient checks before FWI use."}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(2,4,figsize=(12,4.8),constrained_layout=True)
    for ax,record in zip(axes.flat,records[:8]):
        data=torch.load(args.out/record['file'],weights_only=True)
        ax.imshow(data['gamma'].T,origin='lower',extent=(0,100,0,50),vmin=.3,vmax=1,cmap='viridis')
        ax.set(title=f"{record['group_id'][-3:]}: {record['split']}",xlabel='x (mm)',ylabel='y (mm)')
    fig.savefig(args.out/'pilot_geometries.png',dpi=150);plt.close(fig)
    print(json.dumps({"count":len(records),"normalization":manifest['normalization']},indent=2))


if __name__=='__main__':
    main()
