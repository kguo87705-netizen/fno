"""Compare the released continuous-adjoint direction with discrete cost changes.

This audits the code's gradient; it does not modify the published solver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import AdjointMethod
import FiniteDifferencePyTorchConv as FiniteDifference
import Utilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = pd.read_csv(root / "settings2D.csv").iloc[0]
    nx, ny, steps, shots = (int(cfg[k]) for k in ("Nx", "Ny", "N", "numberOfSources"))
    dx, dy, dt, c, rho = float(cfg["Lx"] / nx), float(cfg["Ly"] / ny), float(cfg["dt"]), float(cfg["c"]), float(cfg["rho"])
    sx, sy = Utilities.getSourceLocations(nx, ny, int(cfg.distanceBetweenSources), shots)
    rx, ry = Utilities.getSensorLocations(nx, ny, int(cfg.distanceBetweenSensors), sx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gamma = torch.ones((1, 1, nx + 3, ny + 3), device=device)
    u0 = torch.zeros((shots, 1, nx + 3, ny + 3), device=device)
    u1 = torch.zeros_like(u0)
    forcing = torch.tensor(pd.read_hdf(root / "data_casestudy/source.h5").values).reshape(shots, 1, nx + 3, ny + 3, steps).to(device)
    observed = torch.tensor(pd.read_hdf(root / "data_casestudy/measurement1.h5").values).reshape(1, shots, len(rx), steps).to(device)
    solver = FiniteDifference.FiniteDifference(dt, dx, dy, c, rho, device)
    with torch.no_grad():
        cost, gradient = AdjointMethod.getAdjointGradient(solver, u0, u1, c, rho, gamma, forcing, nx, dx, ny, dy, steps, dt, shots, observed, rx, ry, device)
        direction = torch.zeros_like(gamma)
        direction[0, 0, 1:-1, 1:-1] = gradient
        direction[0, 0, rx, ry] = 0
        direction[0, 0, sx, sy] = 0
        direction /= direction.norm()
        projected = float((gradient * direction[0, 0, 1:-1, 1:-1]).sum().item())

        def objective(material: torch.Tensor) -> tuple[float, float]:
            wavefield = solver.forwardNSteps(u0.clone(), u1.clone(), material, forcing, nx, ny, steps, shots, device)
            residual = wavefield[:, rx, ry, 1:] - observed
            published = 0.5 * torch.sum(dx * dy * torch.trapezoid(residual.square(), dx=dt, dim=2)) / shots
            time_axis = 0.5 * torch.sum(dx * dy * torch.trapezoid(residual.square(), dx=dt, dim=-1)) / shots
            return float(published.item()), float(time_axis.item())

        initial_published, initial_time_axis = objective(gamma)
        checks = {}
        for epsilon in (0.1, 0.05, 0.02):
            (plus, plus_time), (minus, minus_time) = objective(gamma + epsilon * direction), objective(gamma - epsilon * direction)
            fd = (plus - minus) / (2 * epsilon)
            fd_time = (plus_time - minus_time) / (2 * epsilon)
            checks[str(epsilon)] = {
                "finite_difference": fd, "fd_over_adjoint": fd / projected,
                "relative_error": abs(fd - projected) / max(abs(projected), 1e-30),
                "cost_minus_direction": minus, "cost_plus_direction": plus,
                "time_axis_finite_difference": fd_time,
                "time_axis_fd_over_adjoint": fd_time / projected,
                "time_axis_relative_error": abs(fd_time - projected) / max(abs(projected), 1e-30),
            }
        additional = {}
        x = (torch.arange(nx + 3, device=device) - 1) * dx
        y = (torch.arange(ny + 3, device=device) - 1) * dy
        for name, xc, yc in (("lower_left_bump", 0.038, 0.010), ("upper_right_bump", 0.070, 0.035)):
            bump = torch.exp(-0.5 * ((x[:, None] - xc) / 0.006).square() - 0.5 * ((y[None, :] - yc) / 0.004).square())
            trial = torch.zeros_like(gamma)
            trial[0, 0, 1:-1, 1:-1] = bump[1:-1, 1:-1]
            trial[0, 0, rx, ry] = 0
            trial[0, 0, sx, sy] = 0
            trial /= trial.norm()
            adjoint = float((gradient * trial[0, 0, 1:-1, 1:-1]).sum().item())
            additional[name] = {"author_adjoint_projection": adjoint, "checks": {}}
            for epsilon in (0.05, 0.02):
                (plus, plus_time), (minus, minus_time) = objective(gamma + epsilon * trial), objective(gamma - epsilon * trial)
                fd = (plus - minus) / (2 * epsilon)
                fd_time = (plus_time - minus_time) / (2 * epsilon)
                additional[name]["checks"][str(epsilon)] = {
                    "finite_difference": fd, "fd_over_adjoint": fd / adjoint,
                    "relative_error": abs(fd - adjoint) / max(abs(adjoint), 1e-30),
                    "time_axis_finite_difference": fd_time,
                    "time_axis_fd_over_adjoint": fd_time / adjoint,
                    "time_axis_relative_error": abs(fd_time - adjoint) / max(abs(adjoint), 1e-30),
                }
    report = {
        "description": "Case1 at homogeneous initial material; direction is the normalized author adjoint gradient after source/receiver masking. Original grid/time and float32 retained.",
        "initial_cost": float(cost.item()), "author_adjoint_projection": projected,
        "residual_shape": [1, shots, len(rx), steps],
        "published_trapezoid_axis": 2, "actual_time_axis": 3,
        "initial_published_axis_cost": initial_published,
        "initial_last_axis_time_cost": initial_time_axis,
        "checks": checks,
        "additional_directions": additional,
        "passes_2_percent_discrete_direction_check": all(v["relative_error"] < 0.02 for v in checks.values()),
        "minus_direction_lowers_cost": all(v["cost_minus_direction"] < float(cost.item()) for v in checks.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
