"""Check acquisition preservation and the gradient through material padding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from aperture_sampling import acquisition, expanded_domain
from baseline import configuration, defect_model, forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(7)
    torch.set_num_threads(4)
    original = acquisition(configuration(torch.device("cpu")), "gaussian", "gaussian", 3.0)
    expanded = expanded_domain(original)
    x0 = original["nx"] // 2
    assert torch.equal(expanded["basis"][:, x0:x0 + original["nx"], :original["ny"]], original["basis"])
    assert torch.equal(expanded["receiver_weights"][:, x0:x0 + original["nx"], :original["ny"]], original["receiver_weights"])
    assert original["dx"] == expanded["dx"] and original["dy"] == expanded["dy"]
    with torch.no_grad():
        observed = forward(F.pad(defect_model(original), expanded["padding"], value=1), expanded)
    scale = observed.abs().max()
    model = torch.full((original["nx"], original["ny"]), 0.8, dtype=torch.float64, requires_grad=True)
    direction = torch.zeros_like(model)
    direction[4:-4, 4:] = torch.randn_like(direction[4:-4, 4:])
    direction /= direction.norm()

    def loss(material: torch.Tensor) -> torch.Tensor:
        predicted = forward(F.pad(material, expanded["padding"], value=1), expanded)
        return 0.5 * ((predicted - observed) / scale).square().mean()

    gradient = torch.autograd.grad(loss(model), model)[0]
    projected = float((gradient * direction).sum().item())
    errors = {}
    for epsilon in (0.01, 0.005):
        with torch.no_grad():
            fd = float(((loss(model + epsilon * direction) - loss(model - epsilon * direction)) / (2 * epsilon)).item())
        errors[str(epsilon)] = abs(fd - projected) / max(abs(projected), 1e-12)
    assert errors["0.005"] < 0.02, errors
    report = {"device": "cpu", "dtype": "float64", "padding_acquisition_checks": "passed", "autograd_projection": projected, "relative_directional_errors": errors}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
