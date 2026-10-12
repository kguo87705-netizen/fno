"""Localize continuous/discrete gradient differences in saved audit fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = {}
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.2), constrained_layout=True)
    for row, (label, suffix) in enumerate((("Initial", "initial"), ("Author update 5", "update5"))):
        base = Path(f".build/author-discrete-full64-{suffix}-20261008")
        data = torch.load(base / "fields.pt", weights_only=True)
        report = json.loads((base / "report.json").read_text())
        a = data["discrete_gradient"][0, 0, 1:-1, 1:-1].double()
        b = data["author_gradient"][0, 0, 1:-1, 1:-1].double()
        error = (a - b).square()
        statistics = {}
        for margin in (0, 1, 2, 4, 8):
            aa = a[margin:-margin, margin:-margin] if margin else a
            bb = b[margin:-margin, margin:-margin] if margin else b
            statistics[str(margin)] = {
                "cosine": float((aa * bb).sum() / aa.norm() / bb.norm()),
                "relative_l2": float((aa - bb).norm() / aa.norm()),
                "ad_gradient_squared_norm_fraction_retained": float(aa.square().sum() / a.square().sum()),
                "squared_difference_fraction_removed": float(1 - (aa - bb).square().sum() / error.sum()),
            }
        max_fd_error = max(x["ad_relative_error"] for d in report["directions"].values() for x in d["checks"].values())
        strips = {"array_side_last_two_y_rows": float(error[:, -2:].sum() / error.sum()), "opposite_side_first_two_y_rows": float(error[:, :2].sum() / error.sum())}
        summary[suffix] = {"gradient_angle_degrees": report["gradient_angle_degrees"], "max_ad_direction_relative_error": max_fd_error, "margins": statistics, "error_strips": strips}
        scale = max(float(a.abs().max()), float(b.abs().max()))
        for ax, field, title in zip(axes[row, :3], (a, b, a - b), ("Discrete AD", "Released adjoint", "AD minus adjoint")):
            im = ax.imshow((field / scale).T.numpy(), origin="lower", extent=(0, 100, 0, 50), cmap="RdBu_r", norm=SymLogNorm(linthresh=.001, vmin=-1, vmax=1))
            ax.set(title=f"{label}: {title}", xlabel="x (mm)", ylabel="y (mm)")
        profile = error.sum(0).numpy()
        axes[row, 3].semilogy(np.arange(len(profile)) * 50 / (len(profile)-1), np.maximum(profile / profile.sum(), 1e-15))
        axes[row, 3].set(title="Squared difference by y row", xlabel="y (mm), array at 50 mm", ylabel="Fraction per row", ylim=(1e-10, 1))
        axes[row, 3].grid(alpha=.25)
    fig.colorbar(im, ax=axes[:, :3], label="Gradient / shared maximum (symmetric log)", shrink=.7)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "discrete_gradient_localization.png", dpi=170)
    plt.close(fig)
    (args.out / "localization.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
