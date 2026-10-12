"""Validate a discrete AD reference and compare it with the released adjoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import AdjointMethod
import FiniteDifferencePyTorchConv as FiniteDifference
import Utilities
from discrete_reference import DiscreteReference, time_cost, variable_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--case", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--updates", type=int, default=0, help="Selected case saved update index, 0..35")
    parser.add_argument("--chunk", type=int, default=50)
    parser.add_argument("--adjoint-batch", type=int, default=0, help="0: joint author adjoint; 1: average independent shots")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = pd.read_csv(root / "settings2D.csv").iloc[0]
    nx, ny, full_steps, shots = (int(cfg[k]) for k in ("Nx", "Ny", "N", "numberOfSources"))
    assert 1 < args.steps <= full_steps and 0 <= args.updates <= 35
    dx, dy, dt, c, rho = (float(cfg[k]) for k in ("Lx", "Ly", "dt", "c", "rho"))
    dx, dy = dx / nx, dy / ny
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2)
    sx, sy = Utilities.getSourceLocations(nx, ny, int(cfg["distanceBetweenSources"]), shots)
    rx, ry = Utilities.getSensorLocations(nx, ny, int(cfg["distanceBetweenSensors"]), sx)
    # The released wavefield buffer uses the default dtype. Keep both coherent.
    torch.set_default_dtype(dtype)
    solver = FiniteDifference.FiniteDifference(dt, dx, dy, c, rho, device).to(dtype=dtype)
    full_force = torch.tensor(pd.read_hdf(root / "data_casestudy/source.h5").values).reshape(shots, 1, nx + 3, ny + 3, full_steps)
    force = full_force[..., :args.steps].to(device=device, dtype=dtype)
    del full_force
    observed = torch.tensor(pd.read_hdf(root / f"data_casestudy/measurement{args.case}.h5").values).reshape(1, shots, len(rx), full_steps)[..., :args.steps].to(device=device, dtype=dtype)
    gamma = torch.ones((1, 1, nx + 3, ny + 3), device=device, dtype=dtype)
    history_path = Path(f".build/author-singh-case{args.case}-35epochs-20261008/GammaHistory/conventionalFWI_{args.case}.h5")
    if args.updates:
        history = pd.read_hdf(history_path).values.reshape(36, nx + 3, ny + 3)
        gamma = torch.tensor(history[args.updates], device=device, dtype=dtype)[None, None]
    mask = variable_mask(gamma, rx, ry, sx, sy)
    zeros = gamma.new_zeros((shots, 1, nx + 3, ny + 3))
    reference = DiscreteReference(solver, rx, ry, args.chunk)
    started = time.perf_counter()
    with torch.no_grad():
        wave = solver.forwardNSteps(zeros.clone(), zeros.clone(), gamma, force, nx, ny, args.steps, shots, device)
        released = wave[:, rx, ry, 1:].clone()
        del wave
        ref_data = reference.traces(gamma, force, checkpointed=False)
        difference = float((ref_data - released).norm() / released.norm())
        max_difference = float((ref_data - released).abs().max())
        assert difference < 1e-6, f"Forward parity failed: {difference}"
        initial_cost = float(time_cost(ref_data, observed, dx, dy, dt))
        if args.adjoint_batch == 1:
            continuous = gamma.new_zeros((nx + 1, ny + 1))
            for shot in range(shots):
                _, one = AdjointMethod.getAdjointGradient(solver, zeros[shot:shot+1].clone(), zeros[shot:shot+1].clone(), c, rho, gamma, force[shot:shot+1], nx, dx, ny, dy, args.steps, dt, 1, observed[:, shot:shot+1], rx, ry, device)
                continuous += one / shots
        else:
            _, continuous = AdjointMethod.getAdjointGradient(solver, zeros, zeros.clone(), c, rho, gamma, force, nx, dx, ny, dy, args.steps, dt, shots, observed, rx, ry, device)
        adjoint = torch.zeros_like(gamma)
        adjoint[0, 0, 1:-1, 1:-1] = continuous
        adjoint *= mask
    print(f"Forward parity passed: relative={difference:.3g}; starting discrete AD", flush=True)
    variable = gamma.detach().clone().requires_grad_(True)
    cost = time_cost(reference.traces(variable, force), observed, dx, dy, dt)
    (cost * 1e12).backward()
    ad = variable.grad.detach() / 1e12 * mask
    assert torch.isfinite(ad).all() and ad.norm() > 0
    cosine = float((ad * adjoint).sum() / (ad.norm() * adjoint.norm()))
    l2 = float((ad - adjoint).norm() / ad.norm())
    best_scale = float((ad * adjoint).sum() / adjoint.square().sum())
    scaled_l2 = float((ad - best_scale * adjoint).norm() / ad.norm())
    print(f"AD completed; cosine={cosine:.6f}; relative L2={l2:.6f}", flush=True)
    directions = {"author_gradient": adjoint / adjoint.norm(), "discrete_gradient": ad / ad.norm()}
    x = (torch.arange(nx + 3, device=device) - 1) * dx
    y = (torch.arange(ny + 3, device=device) - 1) * dy
    for name, xc, yc in (("lower_left_bump", .038, .010), ("upper_right_bump", .070, .035)):
        bump = torch.exp(-.5 * ((x[:, None] - xc) / .006).square() - .5 * ((y[None, :] - yc) / .004).square())[None, None] * mask
        directions[name] = bump / bump.norm()
    checks = {}
    epsilons = (.05, .02) if dtype == torch.float32 else (.02, .005)
    with torch.no_grad():
        for name, direction in directions.items():
            ad_projected = float((ad * direction).sum())
            author_projected = float((adjoint * direction).sum())
            values = {}
            for epsilon in epsilons:
                plus = float(time_cost(reference.traces(gamma + epsilon * direction, force, False), observed, dx, dy, dt))
                minus = float(time_cost(reference.traces(gamma - epsilon * direction, force, False), observed, dx, dy, dt))
                fd = (plus - minus) / (2 * epsilon)
                values[str(epsilon)] = {
                    "fd": fd, "fd_over_ad": fd / ad_projected,
                    "ad_relative_error": abs(fd - ad_projected) / max(abs(ad_projected), 1e-30),
                    "author_relative_error": abs(fd - author_projected) / max(abs(author_projected), 1e-30),
                    "plus_cost": plus, "minus_cost": minus,
                }
            checks[name] = {"ad_projection": ad_projected, "author_projection": author_projected, "checks": values}
            print(f"Direction {name} checked", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"gamma": gamma.cpu(), "mask": mask.cpu(), "discrete_gradient": ad.cpu(), "author_gradient": adjoint.cpu()}, args.out / "fields.pt")
    report = {
        "scope": f"Case {args.case}, same released spatial stencil, sources, receivers and time step; checkpointed discrete AD with actual time integration.",
        "case": args.case,
        "steps": args.steps, "updates": args.updates, "dtype": args.dtype,
        "solver_constants": "Released dt*c/dx constants initially constructed as float32, then promoted if float64; unchanged stencil coefficients.",
        "chunk_size": args.chunk, "initial_cost": initial_cost, "ad_cost": float(cost.detach()),
        "author_adjoint_shot_batch": 1 if args.adjoint_batch == 1 else shots,
        "forward_relative_error": difference, "forward_max_abs_error": max_difference,
        "gradient_cosine": cosine, "gradient_angle_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
        "author_vs_ad_relative_l2": l2, "best_author_scale": best_scale, "scaled_author_relative_l2": scaled_l2,
        "directions": checks,
        "discrete_ad_passes_2_percent": all(item["ad_relative_error"] < .02 for direction in checks.values() for item in direction["checks"].values()),
        "torch_version": torch.__version__, "device": str(device), "elapsed_seconds": time.perf_counter() - started,
        "original_solver_sha256": hashlib.sha256((root / "FiniteDifferencePyTorchConv.py").read_bytes()).hexdigest(),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
