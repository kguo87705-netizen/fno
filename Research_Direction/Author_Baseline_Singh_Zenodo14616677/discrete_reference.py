"""Differentiable time loop for the released 2-D finite-difference stencil.

The original files remain unchanged. Harmonic face coefficients and arithmetic
order match the author's step; ghost reflection uses functional indexing.
Checkpointing only recomputes time chunks, without truncating derivatives.
"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint


class DiscreteReference:
    def __init__(self, solver, rx, ry, chunk_size: int = 50):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.solver, self.rx, self.ry, self.chunk_size = solver, rx, ry, chunk_size

    def traces(self, gamma: torch.Tensor, force: torch.Tensor, checkpointed=True) -> torch.Tensor:
        inverse = 1.0 / gamma
        coefficients = tuple(1.0 / filt(inverse) for filt in (
            self.solver.filtergammax1, self.solver.filtergammax2,
            self.solver.filtergammay1, self.solver.filtergammay2,
        ))
        ix = torch.arange(gamma.shape[2], device=gamma.device)
        iy = torch.arange(gamma.shape[3], device=gamma.device)
        ix[0], ix[-1] = 2, gamma.shape[2] - 3
        iy[0], iy[-1] = 2, gamma.shape[3] - 3
        older = gamma.new_zeros(force.shape[:4])
        current = torch.zeros_like(older)

        def advance(u0, u1, material, a, b, c, d, forcing):
            observed = []
            for i in range(forcing.shape[-1]):
                u2 = (-u0 + 2 * u1 + (
                    a * self.solver.filterux1(u1) - b * self.solver.filterux2(u1)
                    + c * self.solver.filteruy1(u1) - d * self.solver.filteruy2(u1)
                    + self.solver.dt2rho * forcing[..., i]
                ) / material)
                u2 = u2.index_select(2, ix).index_select(3, iy)
                observed.append(u2[:, 0, self.rx, self.ry])
                u0, u1 = u1, u2
            return u0, u1, torch.stack(observed, dim=-1)

        chunks = []
        for start in range(0, force.shape[-1], self.chunk_size):
            arguments = (older, current, gamma, *coefficients, force[..., start:start + self.chunk_size])
            if checkpointed and torch.is_grad_enabled() and gamma.requires_grad:
                older, current, data = checkpoint(advance, *arguments, use_reentrant=False)
            else:
                older, current, data = advance(*arguments)
            chunks.append(data)
        return torch.cat(chunks, dim=-1)


def time_cost(prediction, observation, dx, dy, dt):
    """Author normalization, integrating along the actual final time axis."""
    if observation.ndim == 4:
        assert observation.shape[0] == 1
        observation = observation[0]
    assert prediction.shape == observation.shape
    return 0.5 * dx * dy * torch.trapezoid((prediction - observation).square(), dx=dt, dim=-1).sum() / prediction.shape[0]


def variable_mask(gamma, rx, ry, sx, sy):
    mask = torch.ones_like(gamma)
    mask[..., 0, :] = mask[..., -1, :] = 0
    mask[..., :, 0] = mask[..., :, -1] = 0
    mask[0, 0, rx, ry] = 0
    mask[0, 0, sx, sy] = 0
    return mask
