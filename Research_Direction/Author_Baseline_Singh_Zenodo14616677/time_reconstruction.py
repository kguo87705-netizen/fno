"""Differentiable fixed local cubic interpolation, with known zero initial data."""

import torch


def cubic_time_reconstruction(sparse, snapshot_times, full_times):
    assert snapshot_times.numel() == sparse.shape[-1] and snapshot_times.numel() >= 3
    assert snapshot_times[0] > 0 and full_times[0] > 0
    t = torch.cat([snapshot_times.new_zeros(1), snapshot_times])
    assert bool((torch.diff(t) > 0).all())
    assert full_times[-1] <= t[-1]
    # Use two points around the evaluation interval where possible, four at edges.
    first = (torch.searchsorted(t, full_times, right=True) - 2).clamp(0, t.numel()-4)
    idx = first[:,None] + torch.arange(4, device=t.device)[None,:]
    knots = t[idx]
    weights = torch.ones_like(knots)
    for j in range(4):
        for k in range(4):
            if j != k:
                weights[:,j] *= (full_times - knots[:,k]) / (knots[:,j]-knots[:,k])
    values = torch.cat([torch.zeros_like(sparse[...,:1]), sparse], dim=-1)
    chosen = values.index_select(-1,idx.flatten().to(sparse.device)).reshape(*sparse.shape[:-1],full_times.numel(),4)
    return (chosen * weights.to(sparse)).sum(-1)
