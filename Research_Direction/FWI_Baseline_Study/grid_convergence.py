"""Diagnose spatial-grid mismatch in the independent scalar FWI smoke model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from baseline import configuration, defect_model, forward


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).square().mean().sqrt().item())


def shifted(trace: torch.Tensor, lag: int) -> torch.Tensor:
    """Positive lag delays a trace, padding with zeros rather than wrapping."""
    if lag == 0:
        return trace
    out = torch.zeros_like(trace)
    if lag > 0:
        out[..., lag:] = trace[..., :-lag]
    else:
        out[..., :lag] = trace[..., -lag:]
    return out


def fit_gain_and_lag(coarse: torch.Tensor, fine: torch.Tensor, max_lag: int = 12) -> dict:
    candidates = []
    for lag in range(-max_lag, max_lag + 1):
        x = shifted(coarse, lag)
        gain = (x * fine).sum() / x.square().sum()
        candidates.append((rmse(gain * x, fine), lag, float(gain.item())))
    error, lag, gain = min(candidates)
    gain_only = (coarse * fine).sum() / coarse.square().sum()
    return {
        "raw_rmse": rmse(coarse, fine),
        "gain_only": float(gain_only.item()),
        "gain_only_rmse": rmse(gain_only * coarse, fine),
        "best_lag_samples": lag,
        "best_lag_microseconds": lag * 0.04,
        "gain_with_lag": gain,
        "gain_and_lag_rmse": error,
    }


def halve_timestep(cfg: dict) -> dict:
    updated = dict(cfg)
    updated["dt"] = cfg["dt"] / 2
    updated["steps"] = cfg["steps"] * 2
    times = torch.arange(updated["steps"], dtype=torch.float64, device=cfg["device"]) * updated["dt"]
    omega = 2 * math.pi * cfg["frequency"]
    updated["pulse"] = torch.where(
        times <= 2 / cfg["frequency"],
        torch.sin(omega * times) * torch.sin(omega * times / 4) ** 2,
        torch.zeros_like(times),
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    traces = {}
    geometry = {}
    with torch.no_grad():
        for refinement in (1, 2, 4):
            cfg = configuration(device, refinement)
            background = torch.ones((cfg["nx"], cfg["ny"]), dtype=torch.float64, device=device)
            homogeneous = forward(background, cfg).cpu()
            defect = forward(defect_model(cfg), cfg).cpu()
            traces[refinement] = {"homogeneous": homogeneous, "defect": defect, "contrast": defect - homogeneous}
            geometry[refinement] = {
                "grid": [cfg["nx"], cfg["ny"]],
                "dx_mm": cfg["dx"] * 1e3,
                "source_x_mm": (cfg["sources"][:, 0] * cfg["dx"] * 1e3).cpu().tolist(),
                "receiver_x_mm": (cfg["receiver_x"] * cfg["dx"] * 1e3).cpu().tolist(),
                "source_receiver_y_mm": cfg["dy"] * refinement * 1e3,
                "homogeneous_peak": float(homogeneous.abs().max().item()),
                "defect_contrast_rmse": rmse(defect, homogeneous),
            }
    time_refinement = {}
    with torch.no_grad():
        for refinement in (1, 2):
            cfg = halve_timestep(configuration(device, refinement))
            background = torch.ones((cfg["nx"], cfg["ny"]), dtype=torch.float64, device=device)
            # Forward records u_(n+1), so indices 1,3,... align with coarse samples.
            homogeneous = forward(background, cfg)[..., 1::2].cpu()
            defect = forward(defect_model(cfg), cfg)[..., 1::2].cpu()
            time_refinement[refinement] = {
                "homogeneous_rmse": rmse(homogeneous, traces[refinement]["homogeneous"]),
                "defect_rmse": rmse(defect, traces[refinement]["defect"]),
            }
    comparisons = {}
    for coarse_ref, fine_ref in ((1, 2), (2, 4), (1, 4)):
        coarse, fine = traces[coarse_ref], traces[fine_ref]
        fit = fit_gain_and_lag(coarse["homogeneous"], fine["homogeneous"])
        lag, gain = fit["best_lag_samples"], fit["gain_with_lag"]
        gain_only = fit["gain_only"]
        comparisons[f"{coarse_ref}x_vs_{fine_ref}x"] = {
            "homogeneous": fit,
            "defect_raw_rmse": rmse(coarse["defect"], fine["defect"]),
            "defect_after_background_gain_rmse": rmse(gain_only * coarse["defect"], fine["defect"]),
            "defect_after_background_gain_and_lag_rmse": rmse(gain * shifted(coarse["defect"], lag), fine["defect"]),
            "contrast_raw_rmse": rmse(coarse["contrast"], fine["contrast"]),
            "fine_defect_contrast_rmse": rmse(fine["contrast"], torch.zeros_like(fine["contrast"])),
        }
    report = {
        "description": "Independent scalar finite-difference forward model; same dt, pulse, nominal source/receiver positions; not an author-code reproduction.",
        "device": str(device), "dt_seconds": 4e-8, "steps": 420,
        "geometry": geometry, "comparisons": comparisons,
        "halved_timestep_same_spatial_grid": time_refinement,
        "interpretation_limit": "A 4x grid is only a finer reference, not a converged continuum solution. Global gain/lag are diagnostic fits from homogeneous traces.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    t = (torch.arange(420) + 1) * 0.04
    shot, receiver = 0, 8
    for ref in (1, 2, 4):
        axes[0, 0].plot(t, traces[ref]["homogeneous"][shot, receiver], label=f"{ref}x")
        axes[0, 1].plot(t, traces[ref]["contrast"][shot, receiver], label=f"{ref}x")
    axes[0, 0].set(title="Homogeneous trace", xlabel="Time (us)", ylabel="Displacement")
    axes[0, 1].set(title="Defect minus homogeneous", xlabel="Time (us)", ylabel="Displacement")
    labels = list(comparisons)
    x = range(len(labels))
    axes[1, 0].bar([v - 0.2 for v in x], [comparisons[k]["homogeneous"]["raw_rmse"] for k in labels], width=0.2, label="Raw")
    axes[1, 0].bar(x, [comparisons[k]["homogeneous"]["gain_only_rmse"] for k in labels], width=0.2, label="Gain")
    axes[1, 0].bar([v + 0.2 for v in x], [comparisons[k]["homogeneous"]["gain_and_lag_rmse"] for k in labels], width=0.2, label="Gain + lag")
    axes[1, 0].set(title="Homogeneous RMSE", xticks=list(x), xticklabels=labels, ylabel="RMSE")
    axes[1, 0].legend()
    axes[1, 1].bar([v - 0.2 for v in x], [comparisons[k]["defect_raw_rmse"] for k in labels], width=0.4, label="Defect mismatch")
    axes[1, 1].bar([v + 0.2 for v in x], [comparisons[k]["fine_defect_contrast_rmse"] for k in labels], width=0.4, label="Defect signal")
    axes[1, 1].set(title="Mismatch versus defect signal", xticks=list(x), xticklabels=labels, ylabel="RMSE")
    axes[1, 1].legend()
    for ax in axes[0]:
        ax.legend()
    fig.savefig(args.out / "grid_convergence.png", dpi=160)
    plt.close(fig)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
