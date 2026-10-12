"""Small cross-grid inversion check with fixed physical Gaussian apertures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aperture_sampling import acquisition, expanded_domain
from baseline import configuration, defect_model, forward


def case_model(cfg: dict, case: str) -> torch.Tensor:
    if case == "original":
        return defect_model(cfg)
    x = (torch.arange(cfg["nx"], dtype=torch.float64, device=cfg["device"]) + 0.5) * cfg["dx"]
    y = (torch.arange(cfg["ny"], dtype=torch.float64, device=cfg["device"]) + 0.5) * cfg["dy"]
    if case == "small":
        ellipses = [(0.05, 0.025, 0.003, 0.0025)]
    elif case == "double":
        ellipses = [(0.04, 0.025, 0.003, 0.003), (0.06, 0.025, 0.003, 0.003)]
    elif case == "shallow":
        ellipses = [(0.05, 0.015, 0.007, 0.005)]
    elif case == "deep":
        ellipses = [(0.05, 0.035, 0.007, 0.005)]
    else:
        raise ValueError(case)
    mask = torch.zeros((cfg["nx"], cfg["ny"]), dtype=torch.bool, device=cfg["device"])
    for x0, y0, rx, ry in ellipses:
        mask |= ((x[:, None] - x0) / rx).square() + ((y[None, :] - y0) / ry).square() < 1
    return torch.where(mask, 0.35, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sigma-mm", type=float, default=1.5)
    parser.add_argument("--acquisition", choices=("indexed", "gaussian"), default="gaussian")
    parser.add_argument("--case", choices=("original", "small", "double", "shallow", "deep"), default="original")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--boundary-domain", choices=("reflecting", "expanded"), default="reflecting")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.sigma_mm <= 0 or args.iterations < 1:
        raise ValueError("sigma and iterations must be positive")
    torch.manual_seed(7)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if args.acquisition == "indexed":
        coarse, fine = configuration(device), configuration(device, 2)
    else:
        coarse = acquisition(configuration(device), "gaussian", "gaussian", args.sigma_mm)
        fine = acquisition(configuration(device, 2), "gaussian", "gaussian", args.sigma_mm)
    truth = case_model(coarse, args.case)
    fine_truth = case_model(fine, args.case)
    if args.boundary_domain == "expanded":
        coarse, fine = expanded_domain(coarse), expanded_domain(fine)

    def simulate(material: torch.Tensor, cfg: dict) -> torch.Tensor:
        if "padding" in cfg:
            material = torch.nn.functional.pad(material, cfg["padding"], value=1)
        return forward(material, cfg)

    with torch.no_grad():
        observed = simulate(fine_truth, fine)
        background_reference = forward(torch.ones((fine["nx"], fine["ny"]), dtype=torch.float64, device=device), fine)
        background_coarse = simulate(torch.ones_like(truth), coarse)
        background_rmse = float((background_reference - background_coarse).square().mean().sqrt().item())
        defect_signal_rmse = float((observed - background_reference).square().mean().sqrt().item())
    scale = observed.abs().max().detach()
    model = torch.nn.Parameter(torch.ones_like(truth))
    optimizer = torch.optim.Adam([model], lr=0.04)
    history = []
    for _ in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        predicted = simulate(model, coarse)
        loss = 0.5 * ((predicted - observed) / scale).square().mean()
        history.append(float(loss.item()))
        loss.backward()
        model.grad[:, :4] = 0
        model.grad[:4, :] = 0
        model.grad[-4:, :] = 0
        optimizer.step()
        with torch.no_grad():
            model.clamp_(0.3, 1.0)
    with torch.no_grad():
        predicted = simulate(model, coarse)
        history.append(float((0.5 * ((predicted - observed) / scale).square().mean()).item()))
        true_mask = truth < 0.5
        predicted_mask = model < 0.65
        dice = float((2 * (true_mask & predicted_mask).sum() / (true_mask.sum() + predicted_mask.sum()).clamp_min(1)).item())
    report = {
        "status": "synthetic_cross_grid_check_only",
        "case": args.case, "acquisition": args.acquisition,
        "boundary_domain": args.boundary_domain,
        "sigma_x_mm": args.sigma_mm if args.acquisition == "gaussian" else None,
        "sigma_y_mm": 0.8 if args.acquisition == "gaussian" else None,
        "coarse_grid": [64, 32], "reference_grid": [128, 64],
        "coarse_simulation_grid": [coarse["nx"], coarse["ny"]],
        "reference_simulation_grid": [fine["nx"], fine["ny"]],
        "iterations": args.iterations, "device": str(device),
        "background_cross_grid_rmse": background_rmse,
        "fine_defect_signal_rmse": defect_signal_rmse,
        "initial_loss": history[0], "final_loss": history[-1],
        "mean_reconstructed_inside": float(model.detach()[truth < 0.5].mean().item()),
        "mean_reconstructed_outside": float(model.detach()[truth >= 0.5].mean().item()),
        "material_mae": float((model.detach() - truth).abs().mean().item()),
        "exploratory_dice_at_gamma_0p65": dice,
        "predicted_defect_cells_at_gamma_0p65": int(predicted_mask.sum().item()),
        "true_defect_cells": int(true_mask.sum().item()),
        "history": history,
        "limit": "Same scalar PDE and difference scheme; expanded domain moves side/bottom specimen boundaries while retaining the reflecting top and original inversion cells. Gaussian widths are uncalibrated. The gamma<0.65 threshold is frozen for later cases before their runs.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"truth": truth.cpu(), "reconstruction": model.detach().cpu(), "observed": observed.cpu(), "predicted": predicted.cpu()}, args.out / "fields.pt")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
    for ax, field, title in zip(axes[:2], (truth, model.detach()), ("True density scale", "Reconstructed density scale")):
        im = ax.imshow(field.cpu().T, origin="lower", extent=(0, 100, 0, 50), vmin=0.3, vmax=1)
        ax.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
        fig.colorbar(im, ax=ax, shrink=0.7)
    axes[2].semilogy(history)
    axes[2].set(title="Normalized waveform loss", xlabel="Iteration", ylabel="MSE / 2")
    fig.savefig(args.out / "aperture_inversion.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
