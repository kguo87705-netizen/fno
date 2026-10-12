"""Isolate side/bottom reflections without changing dx, source or receiver weights.

The expanded domain retains the reflecting top and embeds the original domain in
uniform material. It is a boundary-isolation diagnostic, not a physical model of
the same finite specimen or an independent numerical solver.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from aperture_inversion import case_model
from aperture_sampling import acquisition, expanded_domain
from baseline import configuration, forward


def rms(trace: torch.Tensor) -> float:
    return float(trace.square().mean().sqrt().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sigma-mm", type=float, default=3.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if args.sigma_mm <= 0:
        raise ValueError("sigma must be positive")
    traces = {}
    with torch.no_grad():
        for refinement in (1, 2):
            cfg = acquisition(configuration(device, refinement), "gaussian", "gaussian", args.sigma_mm)
            expanded = expanded_domain(cfg)
            traces[refinement] = {}
            for case in ("background", "shallow", "original", "deep"):
                gamma = torch.ones((cfg["nx"], cfg["ny"]), dtype=torch.float64, device=device) if case == "background" else case_model(cfg, case)
                padded_gamma = F.pad(gamma, expanded["padding"], value=1)
                original = forward(gamma, cfg).cpu()
                large = forward(padded_gamma, expanded).cpu()
                assert torch.isfinite(original).all() and torch.isfinite(large).all()
                traces[refinement][case] = {"reflecting": original, "expanded": large}
    windows = [("0_to_8us", 0, 200), ("8_to_12us", 200, 300), ("12_to_16p8us", 300, 420)]
    report = {
        "description": "Same-grid forward boundary diagnostic. Side boundaries move from x=0,100 mm to x=-50,150 mm; bottom from y=50 to100 mm. The y=0 reflecting top is retained. Original source and receiver weights are zero-padded without refitting.",
        "sigma_x_mm": args.sigma_mm, "sigma_y_mm": 0.8,
        "dt_seconds": 4e-8, "device": str(device),
        "windows": {}, "defect_energy_after_12us": {},
        "limit": "Expanded material changes the specimen boundary problem; it is not an absorber, an independent solver, or validation of real-array performance.",
    }
    for name, start, end in windows:
        window = slice(start, end)
        bg_fine = traces[2]["background"]
        bg_coarse = traces[1]["background"]
        row = {
            "background_boundary_difference_rmse": rms((bg_fine["reflecting"] - bg_fine["expanded"])[..., window]),
            "background_cross_grid_rmse_reflecting": rms((bg_fine["reflecting"] - bg_coarse["reflecting"])[..., window]),
            "background_cross_grid_rmse_expanded": rms((bg_fine["expanded"] - bg_coarse["expanded"])[..., window]),
            "defects": {},
        }
        for case in ("shallow", "original", "deep"):
            fine = traces[2][case]
            signals = {mode: fine[mode] - bg_fine[mode] for mode in ("reflecting", "expanded")}
            row["defects"][case] = {
                "signal_rmse_reflecting": rms(signals["reflecting"][..., window]),
                "signal_rmse_expanded": rms(signals["expanded"][..., window]),
                "signal_boundary_difference_rmse": rms((signals["reflecting"] - signals["expanded"])[..., window]),
            }
        report["windows"][name] = row
    for case in ("shallow", "original", "deep"):
        report["defect_energy_after_12us"][case] = {}
        for mode in ("reflecting", "expanded"):
            signal = traces[2][case][mode] - traces[2]["background"][mode]
            report["defect_energy_after_12us"][case][mode] = float((signal[..., 300:].square().sum() / signal.square().sum()).item())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(traces, args.out / "traces.pt")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    t = (torch.arange(420) + 1) * 0.04
    bg = traces[2]["background"]
    for ax, case in zip(axes.flat, ("background", "shallow", "original", "deep")):
        for mode in ("reflecting", "expanded"):
            signal = traces[2][case][mode]
            if case != "background":
                signal = signal - bg[mode]
            ax.plot(t, signal[0, 8], label=mode)
        ax.axvline(12, color="gray", linestyle="--", linewidth=1)
        ax.set(title=f"{case}: {'total waveform' if case == 'background' else 'defect minus background'}", xlabel="Time (us)", ylabel="Displacement")
        ax.legend()
    fig.savefig(args.out / "boundary_window_diagnostic.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"windows": report["windows"], "late_energy": report["defect_energy_after_12us"]}, indent=2))


if __name__ == "__main__":
    main()
