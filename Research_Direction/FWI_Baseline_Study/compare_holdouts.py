"""Summarize the preselected small/double defect checks from saved outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path(".build"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    modes = [("indexed", "Indexed point"), ("1p5", "Gaussian 1.5 mm"), ("3", "Gaussian 3 mm")]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    summary = {}
    for row, case in enumerate(("small", "double")):
        summary[case] = {}
        first = args.build / f"fwi-holdout-{case}-indexed"
        truth = torch.load(first / "fields.pt", weights_only=True, map_location="cpu")["truth"]
        axes[row, 0].imshow(truth.T, origin="lower", extent=(0, 100, 0, 50), vmin=0.3, vmax=1)
        axes[row, 0].set_title(f"{case}: truth")
        for col, (key, label) in enumerate(modes, start=1):
            path = args.build / f"fwi-holdout-{case}-{key}"
            report = json.loads((path / "report.json").read_text(encoding="utf-8"))
            data = torch.load(path / "fields.pt", weights_only=True, map_location="cpu")
            image = data["reconstruction"]
            axes[row, col].imshow(image.T, origin="lower", extent=(0, 100, 0, 50), vmin=0.3, vmax=1)
            axes[row, col].set_title(f"{label}\nMAE {report['material_mae']:.3f}; Dice {report['exploratory_dice_at_gamma_0p65']:.3f}")
            summary[case][key] = {
                "material_mae": report["material_mae"],
                "dice_gamma_below_0p65": report["exploratory_dice_at_gamma_0p65"],
                "inside_mean": report["mean_reconstructed_inside"],
                "outside_mean": report["mean_reconstructed_outside"],
                "predicted_defect_cells": report["predicted_defect_cells_at_gamma_0p65"],
                "true_defect_cells": report["true_defect_cells"],
                "final_waveform_loss": report["final_loss"],
            }
        for ax in axes[row]:
            ax.set(xlabel="x (mm)", ylabel="y (mm)")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "holdout_comparison.png", dpi=160)
    plt.close(fig)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
