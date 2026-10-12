"""Compare fixed 35-update controls on the same correctly evaluated objective."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    outputs = [Path(".build/author-discrete-fwi-case1-35epochs-20261008"), Path(".build/author-discrete-fwi-case1-lr002-35epochs-20261008")]
    reports = [json.loads((p / "report.json").read_text()) for p in outputs]
    fields = [torch.load(p / "fields.pt", weights_only=True) for p in outputs]
    author = torch.load(".build/author-singh-case1-35epochs-20261008/fields.pt", weights_only=True)
    assert all(r["epochs"] == 35 for r in reports)
    assert np.allclose(reports[0]["author_time_costs"], reports[1]["author_time_costs"], rtol=1e-5, atol=0)
    labels = ["Released adjoint, lr=0.08", "Discrete AD, lr=0.08", "Discrete AD, lr=0.02"]
    costs = [reports[0]["author_time_costs"], reports[0]["costs"], reports[1]["costs"]]
    errors = [reports[0]["author_material_mse_half"], reports[0]["material_mse_half"], reports[1]["material_mse_half"]]
    summary = {}
    for label, cost, error in zip(labels, costs, errors):
        c, e = np.array(cost), np.array(error)
        assert np.isfinite(c).all() and np.isfinite(e).all()
        summary[label] = {"final_cost_over_initial": float(c[-1]/c[0]), "final_material_mse_over_initial": float(e[-1]/e[0]), "cost_increase_steps": int((np.diff(c)>0).sum()), "min_material_mse_update": int(e.argmin()), "min_material_mse_over_initial": float(e.min()/e[0])}
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, height_ratios=(1, 1.15))
    images = [author["truth"], author["reconstruction"], fields[0]["reconstruction"], fields[1]["reconstruction"]]
    image_axes = []
    for i, (field, title) in enumerate(zip(images, ["Truth", *labels])):
        ax = fig.add_subplot(grid[0, i]); image_axes.append(ax)
        im = ax.imshow(field[0,0,1:-1,1:-1].T, origin="lower", extent=(0,100,0,50), cmap="viridis", vmin=0, vmax=1)
        ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(im, ax=image_axes, label="Density scaling gamma", shrink=.65)
    for col, values, title, ylabel in ((0, costs, "Correct time-integrated waveform objective", "Cost / initial"), (2, errors, "Material reconstruction error", "Half-MSE / initial")):
        ax = fig.add_subplot(grid[1, col:col+2])
        for label, curve in zip(labels, values):
            ax.plot(np.array(curve)/curve[0], label=label)
        ax.set(title=title, xlabel="Updates", ylabel=ylabel)
        ax.grid(alpha=.2); ax.legend(fontsize=9)
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "discrete_fwi_controls.png", dpi=160)
    plt.close(fig)
    (args.out / "controls.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
