"""Compare equal-size defects at three depths using saved inversion fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def paths(build: Path, depth: str, mode: str) -> Path:
    if depth == "middle":
        return build / {
            "indexed": "fwi-baseline-refined-80",
            "1p5": "fwi-aperture-inversion-1p5",
            "3": "fwi-aperture-inversion-3",
        }[mode]
    return build / f"fwi-depth-{depth}-{mode}"


def fields(path: Path, original_baseline: bool) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path / "fields.pt", weights_only=True, map_location="cpu")
    if original_baseline:
        return data["true_gamma"], data["reconstructed_gamma"]
    return data["truth"], data["reconstruction"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path(".build"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), constrained_layout=True)
    summary = {}
    for row, depth in enumerate(("shallow", "middle", "deep")):
        summary[depth] = {}
        for col, (mode, label) in enumerate((("indexed", "Indexed"), ("1p5", "1.5 mm"), ("3", "3 mm")), start=1):
            path = paths(args.build, depth, mode)
            truth, model = fields(path, depth == "middle" and mode == "indexed")
            true_mask, predicted_mask = truth < 0.5, model < 0.65
            dice = float((2 * (true_mask & predicted_mask).sum() / (true_mask.sum() + predicted_mask.sum()).clamp_min(1)).item())
            mae = float((truth - model).abs().mean().item())
            report = json.loads((path / "report.json").read_text(encoding="utf-8"))
            summary[depth][mode] = {
                "material_mae": mae,
                "dice_gamma_below_0p65": dice,
                "inside_mean": float(model[true_mask].mean().item()),
                "outside_mean": float(model[~true_mask].mean().item()),
                "predicted_defect_cells": int(predicted_mask.sum().item()),
                "true_defect_cells": int(true_mask.sum().item()),
                "final_waveform_loss": report["final_loss"],
            }
            axes[row, col].imshow(model.T, origin="lower", extent=(0, 100, 0, 50), vmin=0.3, vmax=1)
            axes[row, col].set_title(f"{label}: MAE {mae:.3f}, Dice {dice:.3f}")
            if col == 1:
                axes[row, 0].imshow(truth.T, origin="lower", extent=(0, 100, 0, 50), vmin=0.3, vmax=1)
                axes[row, 0].set_title(f"{depth}: truth")
        for ax in axes[row]:
            ax.set(xlabel="x (mm)", ylabel="y (mm)")
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "depth_comparison.png", dpi=160)
    plt.close(fig)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
