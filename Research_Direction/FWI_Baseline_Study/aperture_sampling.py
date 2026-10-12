"""Isolate point-grid alignment and finite-aperture effects in the scalar model.

All acquisition coordinates are fixed in physical space. The baseline's indexed
point source/receiver is compared with bilinear point sampling and Gaussian
sampling; no inversion is performed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from baseline import configuration, defect_model, forward
from grid_convergence import fit_gain_and_lag, rmse


def expanded_domain(cfg: dict) -> dict:
    """Zero-pad acquisition weights; move side/bottom boundaries by 50 mm.

    The original material cells are embedded unchanged, with the reflecting top
    retained. This changes the physical boundary problem and is diagnostic only.
    """
    pad_x, pad_y = cfg["nx"] // 2, cfg["ny"]
    updated = dict(cfg)
    updated["nx"] = cfg["nx"] + 2 * pad_x
    updated["ny"] = cfg["ny"] + pad_y
    updated["lx"] = updated["nx"] * cfg["dx"]
    updated["ly"] = updated["ny"] * cfg["dy"]
    updated["basis"] = torch.nn.functional.pad(cfg["basis"], (0, pad_y, pad_x, pad_x))
    if "receiver_weights" in cfg:
        updated["receiver_weights"] = torch.nn.functional.pad(cfg["receiver_weights"], (0, pad_y, pad_x, pad_x))
    updated["sources"] = cfg["sources"].clone()
    updated["sources"][:, 0] += pad_x
    updated["receiver_x"] = cfg["receiver_x"] + pad_x
    updated["padding"] = (0, pad_y, pad_x, pad_x)
    return updated


def bilinear_weights(cfg: dict, x_m: float, y_m: float) -> torch.Tensor:
    """Interpolate a cell-centered field at a fixed physical coordinate."""
    fx, fy = x_m / cfg["dx"] - 0.5, y_m / cfg["dy"] - 0.5
    ix, iy = int(fx // 1), int(fy // 1)
    assert 0 <= ix < cfg["nx"] - 1 and 0 <= iy < cfg["ny"] - 1
    wx, wy = fx - ix, fy - iy
    weight = torch.zeros((cfg["nx"], cfg["ny"]), dtype=torch.float64, device=cfg["device"])
    for ax, vx in ((ix, 1 - wx), (ix + 1, wx)):
        for ay, vy in ((iy, 1 - wy), (iy + 1, wy)):
            weight[ax, ay] = vx * vy
    return weight


def gaussian_weights(cfg: dict, x_m: float, y_m: float, sigma_x_m: float, sigma_y_m: float) -> torch.Tensor:
    x = (torch.arange(cfg["nx"], dtype=torch.float64, device=cfg["device"]) + 0.5) * cfg["dx"]
    y = (torch.arange(cfg["ny"], dtype=torch.float64, device=cfg["device"]) + 0.5) * cfg["dy"]
    weight = torch.exp(-0.5 * ((x[:, None] - x_m) / sigma_x_m) ** 2 - 0.5 * ((y[None, :] - y_m) / sigma_y_m) ** 2)
    return weight / weight.sum()


def acquisition(cfg: dict, source_mode: str, receiver_mode: str, sigma_mm: float = 0.0) -> dict:
    if source_mode == receiver_mode == "indexed":
        return cfg
    # Derived once from the original 64x32 indexed geometry; all grids now use
    # these exact physical centers rather than index*dx as a proxy for centers.
    base_dx = 0.1 / 64
    base_dy = 0.05 / 32
    source_x = [22 * base_dx, 42 * base_dx]
    receiver_x = [int(i) * base_dx for i in range(7, 58, 3)]
    y_m = base_dy
    def make_weight(mode: str, x: float) -> torch.Tensor:
        if mode == "bilinear":
            return bilinear_weights(cfg, x, y_m)
        if mode == "gaussian":
            return gaussian_weights(cfg, x, y_m, sigma_mm * 1e-3, 0.8e-3)
        raise ValueError(mode)
    updated = dict(cfg)
    if source_mode != "indexed":
        updated["basis"] = torch.stack([make_weight(source_mode, x) for x in source_x])
    if receiver_mode != "indexed":
        updated["receiver_weights"] = torch.stack([make_weight(receiver_mode, x) for x in receiver_x])
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    modes = [
        ("indexed", "indexed", "indexed", 0.0),
        ("bilinear", "bilinear", "bilinear", 0.0),
        ("gaussian_source_only_1p5mm", "gaussian", "indexed", 1.5),
        ("gaussian_receiver_only_1p5mm", "indexed", "gaussian", 1.5),
        ("gaussian_both_1p5mm", "gaussian", "gaussian", 1.5),
        ("gaussian_both_3mm", "gaussian", "gaussian", 3.0),
    ]
    report = {
        "description": "Forward-only sampling ablation. Each source profile sums to one and each receiver averages with weights summing to one; same scalar PDE, dt, domain and defect.",
        "fixed_centers": "Physical x=index*(0.1/64) m and y=0.05/32 m; bilinear and Gaussian schemes evaluate weights on cell-centered grids.",
        "gaussian_sigma_y_mm": 0.8,
        "cases": {},
    }
    with torch.no_grad():
        for name, source_mode, receiver_mode, sigma in modes:
            traces = {}
            for refinement in (1, 2, 4):
                cfg = acquisition(configuration(device, refinement), source_mode, receiver_mode, sigma)
                background = torch.ones((cfg["nx"], cfg["ny"]), dtype=torch.float64, device=device)
                homogeneous = forward(background, cfg).cpu()
                defect = forward(defect_model(cfg), cfg).cpu()
                traces[refinement] = {"homogeneous": homogeneous, "defect": defect}
            comparisons = {}
            for coarse_ref, fine_ref in ((1, 2), (2, 4)):
                coarse, fine = traces[coarse_ref], traces[fine_ref]
                fit = fit_gain_and_lag(coarse["homogeneous"], fine["homogeneous"])
                comparisons[f"{coarse_ref}x_vs_{fine_ref}x"] = {
                    "homogeneous_raw_rmse": fit["raw_rmse"],
                    "homogeneous_after_gain_and_lag_rmse": fit["gain_and_lag_rmse"],
                    "best_lag_samples": fit["best_lag_samples"],
                    "defect_raw_rmse": rmse(coarse["defect"], fine["defect"]),
                    "fine_defect_signal_rmse": rmse(fine["defect"], fine["homogeneous"]),
                }
            report["cases"][name] = comparisons
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    names = list(report["cases"])
    for ax, pair in zip(axes, ("1x_vs_2x", "2x_vs_4x")):
        values = [report["cases"][name][pair] for name in names]
        xx = list(range(len(names)))
        ax.bar([v - 0.2 for v in xx], [v["homogeneous_raw_rmse"] for v in values], 0.4, label="Background grid mismatch")
        ax.bar([v + 0.2 for v in xx], [v["fine_defect_signal_rmse"] for v in values], 0.4, label="Fine-grid defect signal")
        ax.set(title=pair, xticks=xx, xticklabels=names, ylabel="Trace RMSE")
        ax.tick_params(axis="x", rotation=35)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        ax.legend()
    fig.savefig(args.out / "aperture_sampling.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report["cases"], indent=2))


if __name__ == "__main__":
    main()
