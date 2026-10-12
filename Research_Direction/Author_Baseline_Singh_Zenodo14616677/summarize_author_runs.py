"""Summarize four published-code conventional FWI runs without changing them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path(".build"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(4, 3, figsize=(11, 10), constrained_layout=True)
    summary = {}
    for row, case in enumerate((1, 2, 3, 4)):
        path = args.build / f"author-singh-case{case}-35epochs-20261008"
        report = json.loads((path / "run_report.json").read_text(encoding="utf-8"))
        assert report["status"] == "selected_case_completed" and report["epochs"] == 35
        data = torch.load(path / "fields.pt", weights_only=True, map_location="cpu")
        truth, reconstruction = data["truth"][0, 0], data["reconstruction"][0, 0]
        initial_mse = float((0.5 * (truth - 1).square().mean()).item())
        final_mse = float((0.5 * (truth - reconstruction).square().mean()).item())
        cost_history = np.loadtxt(path / f"CostHistory/conventionalFWI_{case}.txt")
        mse_history = np.loadtxt(path / f"MSEHistory/conventionalFWI_{case}.txt")
        costs = np.append(cost_history, report["post_update_cost"])
        errors = np.insert(mse_history, 0, initial_mse)
        assert np.isfinite(costs).all() and np.isfinite(errors).all()
        summary[str(case)] = {
            "initial_cost": report["initial_cost"], "post_update_cost": report["post_update_cost"],
            "final_cost_over_initial": report["post_update_cost"] / report["initial_cost"],
            "initial_material_mse_half": initial_mse, "final_material_mse_half": final_mse,
            "final_material_mse_over_initial": final_mse / initial_mse,
            "elapsed_seconds": report["elapsed_seconds"], "epochs": 35,
        }
        for ax, material, label in zip(axes[row, :2], (truth, reconstruction), ("Truth", "Conventional FWI, 35 updates")):
            ax.imshow(material[1:-1, 1:-1].T, origin="lower", extent=(0, 100, 0, 50), vmin=0, vmax=1, cmap="viridis")
            ax.set(title=f"Case {case}: {label}", xlabel="x (mm)", ylabel="y (mm)")
        axes[row, 2].plot(costs / costs[0], label="Released cost / initial")
        axes[row, 2].plot(errors / errors[0], label="Material MSE / initial")
        axes[row, 2].set(xlabel="Update", ylabel="Ratio", title="Cost and reconstruction differ")
        axes[row, 2].legend(fontsize=8)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "author_conventional_fwi_cases.png", dpi=160)
    plt.close(fig)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
