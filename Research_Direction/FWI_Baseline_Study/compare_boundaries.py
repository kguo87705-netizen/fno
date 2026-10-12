"""Compare reflecting-specimen and expanded-domain inversion diagnostics."""

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
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), constrained_layout=True)
    summary = {}
    for row, case in enumerate(("shallow", "original", "deep")):
        reflecting = args.build / ("fwi-aperture-inversion-3" if case == "original" else f"fwi-depth-{case}-3")
        expanded = args.build / f"fwi-boundary-expanded-{case}-20261008"
        summary[case] = {}
        for col, (mode, path) in enumerate((("reflecting", reflecting), ("expanded", expanded)), start=1):
            data = torch.load(path / "fields.pt", weights_only=True, map_location="cpu")
            report = json.loads((path / "report.json").read_text())
            summary[case][mode] = {key: report[key] for key in ("material_mae", "exploratory_dice_at_gamma_0p65", "mean_reconstructed_inside", "mean_reconstructed_outside", "final_loss")}
            axes[row, col].imshow(data["reconstruction"].T, origin="lower", extent=(0, 100, 0, 50), vmin=0.3, vmax=1)
            axes[row, col].set_title(f"{mode}: Dice {report['exploratory_dice_at_gamma_0p65']:.3f}")
            if col == 1:
                axes[row, 0].imshow(data["truth"].T, origin="lower", extent=(0, 100, 0, 50), vmin=0.3, vmax=1)
                axes[row, 0].set_title(f"{case}: truth")
        for ax in axes[row]:
            ax.set(xlabel="x (mm)", ylabel="y (mm)")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "boundary_inversion_comparison.png", dpi=160)
    plt.close(fig)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
