"""Re-evaluate unchanged author reconstructions with last-axis time integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

import FiniteDifferencePyTorchConv as FiniteDifference
import Utilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path(".build"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = pd.read_csv(root / "settings2D.csv").iloc[0]
    nx, ny, steps, shots = (int(cfg[k]) for k in ("Nx", "Ny", "N", "numberOfSources"))
    dx, dy, dt = float(cfg["Lx"] / nx), float(cfg["Ly"] / ny), float(cfg["dt"])
    sx, sy = Utilities.getSourceLocations(nx, ny, int(cfg["distanceBetweenSources"]), shots)
    rx, ry = Utilities.getSensorLocations(nx, ny, int(cfg["distanceBetweenSensors"]), sx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = FiniteDifference.FiniteDifference(dt, dx, dy, float(cfg["c"]), float(cfg["rho"]), device)
    force = torch.tensor(pd.read_hdf(root / "data_casestudy/source.h5").values).reshape(shots, 1, nx + 3, ny + 3, steps).to(device)
    zeros = torch.zeros((shots, 1, nx + 3, ny + 3), device=device)

    def simulate(material: torch.Tensor) -> torch.Tensor:
        wavefield = solver.forwardNSteps(zeros.clone(), zeros.clone(), material, force, nx, ny, steps, shots, device)
        return wavefield[:, rx, ry, 1:]

    def costs(prediction: torch.Tensor, observation: torch.Tensor) -> dict:
        residual = prediction - observation
        assert tuple(residual.shape) == (1, shots, len(rx), steps)
        return {name: float((0.5 * torch.sum(dx * dy * torch.trapezoid(residual.square(), dx=dt, dim=axis)) / shots).item()) for name, axis in (("released_axis2", 2), ("last_axis_time", -1))}

    summary = {}
    with torch.no_grad():
        initial = simulate(torch.ones((1, 1, nx + 3, ny + 3), device=device))
        for case in (1, 2, 3, 4):
            path = args.build / f"author-singh-case{case}-35epochs-20261008"
            data = torch.load(path / "fields.pt", weights_only=True, map_location="cpu")
            report = json.loads((path / "run_report.json").read_text())
            observation = torch.tensor(pd.read_hdf(root / f"data_casestudy/measurement{case}.h5").values).reshape(1, shots, len(rx), steps).to(device)
            first, final = costs(initial, observation), costs(simulate(data["reconstruction"].to(device)), observation)
            assert abs(first["released_axis2"] / report["initial_cost"] - 1) < 1e-5
            assert abs(final["released_axis2"] / report["post_update_cost"] - 1) < 1e-5
            summary[str(case)] = {
                "initial": first, "final": final,
                "corrected_time_cost_final_over_initial": final["last_axis_time"] / first["last_axis_time"],
                "model_updates_changed": False,
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
