"""Case-1 controlled FWI using the validated discrete gradient, not author edits."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import FiniteDifferencePyTorchConv as FiniteDifference
import Utilities
from discrete_reference import DiscreteReference, time_cost, variable_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=.08)
    parser.add_argument("--init", choices=["uniform", "truth", "near_truth"], default="uniform")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = pd.read_csv(root / "settings2D.csv").iloc[0]
    nx, ny, steps, shots = (int(cfg[k]) for k in ("Nx", "Ny", "N", "numberOfSources"))
    dx, dy, dt = float(cfg["Lx"] / nx), float(cfg["Ly"] / ny), float(cfg["dt"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2)
    solver = FiniteDifference.FiniteDifference(dt, dx, dy, float(cfg["c"]), float(cfg["rho"]), device)
    sx, sy = Utilities.getSourceLocations(nx, ny, int(cfg["distanceBetweenSources"]), shots)
    rx, ry = Utilities.getSensorLocations(nx, ny, int(cfg["distanceBetweenSensors"]), sx)
    reference = DiscreteReference(solver, rx, ry)
    force = torch.tensor(pd.read_hdf(root / "data_casestudy/source.h5").values).reshape(shots, 1, nx + 3, ny + 3, steps).to(device)
    observed = torch.tensor(pd.read_hdf(root / "data_casestudy/measurement1.h5").values).reshape(shots, len(rx), steps).to(device)
    truth = torch.ones((1, 1, nx + 3, ny + 3), device=device)
    truth[0, 0, 1:-1, 1:-1] = torch.tensor(pd.read_hdf(root / "data_casestudy/material1.h5").values, device=device)
    if args.init == "uniform":
        initial = torch.ones_like(truth)
    elif args.init == "truth":
        initial = truth.detach().clone()
    else:
        initial = 0.95 * truth.detach() + 0.05
    gamma = torch.nn.Parameter(initial)
    mask = variable_mask(gamma, rx, ry, sx, sy)
    optimizer = torch.optim.Adam([gamma], lr=args.lr)
    costs, errors, history = [], [], []
    started = time.perf_counter()
    for update in range(args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        if update == args.epochs:
            with torch.no_grad():
                loss = time_cost(reference.traces(gamma, force, False), observed, dx, dy, dt)
        else:
            loss = time_cost(reference.traces(gamma, force), observed, dx, dy, dt)
        assert torch.isfinite(loss)
        costs.append(float(loss.detach()))
        errors.append(float((.5 * (gamma.detach() - truth).square().mean()).item()))
        history.append(gamma.detach().cpu().clone())
        if update < args.epochs:
            (loss * 1e12).backward()
            gamma.grad *= mask
            assert torch.isfinite(gamma.grad).all()
            torch.nn.utils.clip_grad_norm_([gamma], 1e-5)
            optimizer.step()
            with torch.no_grad():
                gamma.clamp_(float(cfg["gamma0"]), 1)
        if update % 5 == 0:
            print(f"Update {update}: time cost={costs[-1]:.8g}, material half-MSE={errors[-1]:.8g}, elapsed={time.perf_counter()-started:.1f}s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"truth": truth.cpu(), "reconstruction": gamma.detach().cpu(), "history": torch.cat(history, dim=0), "costs": torch.tensor(costs), "material_mse_half": torch.tensor(errors)}, args.out / "fields.pt")
    # Evaluate every saved author model with the same correct time objective.
    author_path = Path(".build/author-singh-case1-35epochs-20261008")
    author_history = pd.read_hdf(author_path / "GammaHistory/conventionalFWI_1.h5").values.reshape(36, nx + 3, ny + 3)
    author_costs, author_errors = [], []
    with torch.no_grad():
        for field in author_history:
            material = torch.tensor(field, dtype=gamma.dtype, device=device)[None, None]
            author_costs.append(float(time_cost(reference.traces(material, force, False), observed, dx, dy, dt)))
            author_errors.append(float((.5 * (material - truth).square().mean()).item()))
    author_final = torch.tensor(author_history[-1], dtype=gamma.dtype)[None, None]
    report = {
        "case": 1, "epochs": args.epochs, "steps": steps, "dtype": str(gamma.dtype),
        "scope": "Same released case, stencil, acquisition, initialization, bounds, learning rate and clipping; correct time cost and exact discrete AD replace continuous gradient. Original files and results preserved.",
        "learning_rate": args.lr, "clip_gradient_norm": 1e-5, "cost_scaling": 1e12, "initialization": args.init,
        "gradient_mask": "Ghost cells and source/receiver material fixed; no added frozen physical boundary layer.",
        "costs": costs, "material_mse_half": errors,
        "final_cost_over_initial": costs[-1] / costs[0] if costs[0] else None,
        "final_material_mse_over_initial": errors[-1] / errors[0] if errors[0] else None,
        "author_time_costs": author_costs, "author_material_mse_half": author_errors,
        "author_final_cost_over_initial": author_costs[-1] / author_costs[0],
        "author_final_material_mse_over_initial": author_errors[-1] / author_errors[0],
        "elapsed_seconds_including_comparison": time.perf_counter() - started,
        "torch_version": torch.__version__,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for ax, material, title in zip(axes[0], (truth.cpu(), author_final, gamma.detach().cpu()), ("Truth", "Released adjoint, 35 updates", f"Discrete AD, lr={args.lr}, {args.epochs} updates")):
        img = ax.imshow(material[0, 0, 1:-1, 1:-1].T, origin="lower", extent=(0, 100, 0, 50), vmin=0, vmax=1, cmap="viridis")
        ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    axes[1, 0].plot(np.array(author_costs) / author_costs[0], label="Released adjoint")
    axes[1, 0].plot(np.array(costs) / costs[0], label="Discrete AD")
    axes[1, 0].set(title="Correct time objective", xlabel="Updates", ylabel="Cost / initial")
    axes[1, 1].plot(np.array(author_errors) / author_errors[0], label="Released adjoint")
    axes[1, 1].plot(np.array(errors) / errors[0], label="Discrete AD")
    axes[1, 1].set(title="Material reconstruction", xlabel="Updates", ylabel="Half-MSE / initial")
    for ax in axes[1, :2]: ax.legend(fontsize=8)
    axes[1, 2].axis("off")
    axes[1, 2].text(0, .8, f"Case 1, four sources, 24 receivers\n2000 steps, 30 ns\nAdam lr={args.lr}, clipped norm=1e-5\nReleased adjoint lr=0.08\nSame source/receiver mask and bounds\nNo extra frozen boundary layer\nCurves evaluated with time integration", fontsize=10, va="top")
    fig.savefig(args.out / "discrete_fwi_comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ("epochs", "final_cost_over_initial", "final_material_mse_over_initial", "author_final_cost_over_initial", "author_final_material_mse_over_initial")}, indent=2))


if __name__ == "__main__":
    main()
