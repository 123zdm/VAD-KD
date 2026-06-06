"""Bures-Wasserstein and related distillation losses for AED-MAE Stage-2."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def diagonal_bw2(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Diagonal BW^2 between two patch sets x,y with shape [M, D]."""
    mu_x = x.mean(dim=0)
    mu_y = y.mean(dim=0)
    std_x = x.std(dim=0, unbiased=False).clamp(min=eps)
    std_y = y.std(dim=0, unbiased=False).clamp(min=eps)
    mean_term = ((mu_x - mu_y) ** 2).mean()
    cov_term = ((std_x - std_y) ** 2).mean()
    return mean_term + cov_term


def gram_patch_loss(x: torch.Tensor, y: torch.Tensor, max_patches: int = 128) -> torch.Tensor:
    """Frobenius loss between patch-patch cosine Gram matrices [M, M]."""
    m = x.shape[0]
    if m > max_patches:
        idx = torch.linspace(0, m - 1, max_patches, device=x.device).long()
        x = x.index_select(0, idx)
        y = y.index_select(0, idx)
        m = max_patches
    x_n = F.normalize(x, dim=-1, eps=1e-6)
    y_n = F.normalize(y, dim=-1, eps=1e-6)
    g_x = x_n @ x_n.T
    g_y = y_n @ y_n.T
    return ((g_x - g_y) ** 2).mean()


def lowrank_bw2(x: torch.Tensor, y: torch.Tensor, rank: int = 32, eps: float = 1e-4) -> torch.Tensor:
    """Full BW^2 in a low-rank subspace estimated from teacher patches."""
    m, d = x.shape
    r = min(rank, m - 1, d)
    if r < 1:
        return diagonal_bw2(x, y, eps=eps)

    yc = y - y.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(yc, full_matrices=False)
    basis = vh[:r].T
    x_r = x @ basis
    y_r = y @ basis

    mu_x = x_r.mean(dim=0)
    mu_y = y_r.mean(dim=0)
    cov_x = (x_r.T @ x_r) / max(m - 1, 1)
    cov_y = (y_r.T @ y_r) / max(m - 1, 1)
    eye = torch.eye(r, device=x.device, dtype=x.dtype)
    cov_x = cov_x + eps * eye
    cov_y = cov_y + eps * eye

    mean_term = ((mu_x - mu_y) ** 2).sum()
    sqrt_x = _matrix_sqrt(cov_x)
    middle = _matrix_sqrt(sqrt_x @ cov_y @ sqrt_x)
    cov_term = torch.trace(cov_x + cov_y - 2.0 * middle)
    return (mean_term + cov_term) / r


def _matrix_sqrt(mat: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix square root via eigh."""
    evals, evecs = torch.linalg.eigh(mat)
    evals = evals.clamp(min=0)
    return (evecs * evals.sqrt().unsqueeze(0)) @ evecs.T


def normalize_loss_terms(*terms: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Scale each per-sample loss term to batch mean ~= 1 (stop-grad denominators)."""
    out = []
    for term in terms:
        scale = term.detach().mean().clamp(min=1e-6)
        out.append(term / scale)
    return tuple(out)
