"""Clip-level Temporal Joint BW² distillation (separate from single-frame bw2_loss.py)."""

from __future__ import annotations

import torch


def _matrix_sqrt(mat: torch.Tensor) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(mat)
    evals = evals.clamp(min=0)
    return (evecs * evals.sqrt().unsqueeze(0)) @ evecs.T


def gaussian_bw2(
    mu_t: torch.Tensor,
    cov_t: torch.Tensor,
    mu_s: torch.Tensor,
    cov_s: torch.Tensor,
) -> torch.Tensor:
    """Full BW² between two Gaussians N(mu_t, cov_t) and N(mu_s, cov_s) in R^r."""
    r = mu_t.shape[0]
    mean_term = ((mu_t - mu_s) ** 2).sum()
    sqrt_t = _matrix_sqrt(cov_t)
    middle = _matrix_sqrt(sqrt_t @ cov_s @ sqrt_t)
    cov_term = torch.trace(cov_t + cov_s - 2.0 * middle)
    return (mean_term + cov_term) / max(r, 1)


def trajectory_gaussian_params(S: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate Gaussian params from K temporal frame-stats.

    S: [K, d] — each row is one frame's statistic vector.
    Returns mu [d], cov [d, d] with temporal covariance across K steps.
    """
    k, d = S.shape
    mu = S.mean(dim=0)
    if k < 2:
        eye = torch.eye(d, device=S.device, dtype=S.dtype)
        return mu, eps * eye

    centered = S - mu.unsqueeze(0)
    cov = (centered.T @ centered) / (k - 1)
    eye = torch.eye(d, device=S.device, dtype=S.dtype)
    return mu, cov + eps * eye


def temporal_joint_bw2(
    S_s: torch.Tensor,
    S_t: torch.Tensor,
    rank: int = 32,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Clip-level Temporal Joint BW² between teacher/student trajectories.

    S_s, S_t: [K, d] frame-level stats from the same clip.
    Projects to a low-rank subspace (from teacher SVD) before full BW².
    """
    k, d = S_t.shape
    r = min(rank, k - 1, d)
    if r < 1:
        return ((S_s.mean(0) - S_t.mean(0)) ** 2).mean()

    centered_t = S_t - S_t.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered_t, full_matrices=False)
    basis = vh[:r].T
    S_s_r = S_s @ basis
    S_t_r = S_t @ basis

    mu_s, cov_s = trajectory_gaussian_params(S_s_r, eps=eps)
    mu_t, cov_t = trajectory_gaussian_params(S_t_r, eps=eps)
    return gaussian_bw2(mu_t, cov_t, mu_s, cov_s)


def extract_frame_stats(
    preds: torch.Tensor,
    masks: torch.Tensor,
    stat: str = "mean",
) -> torch.Tensor:
    """
    Build per-clip temporal stat matrices from masked patch predictions.

    preds: [B, K, L, D]
    masks: [B, K, L], 0=keep 1=remove (same convention as MAE)
    Returns: [B, K, d]
    """
    batch_size, clip_len, _, feat_dim = preds.shape
    out = []
    for b in range(batch_size):
        frame_stats = []
        for k in range(clip_len):
            idx = masks[b, k].bool()
            if idx.sum() == 0:
                frame_stats.append(preds.new_zeros(feat_dim))
                continue
            patches = preds[b, k, idx]
            if stat == "mean_std":
                frame_stats.append(
                    torch.cat([patches.mean(0), patches.std(0, unbiased=False)], dim=0)
                )
            else:
                frame_stats.append(patches.mean(0))
        out.append(torch.stack(frame_stats, dim=0))
    return torch.stack(out, dim=0)


def per_clip_temporal_joint_bw2(
    S_s: torch.Tensor,
    S_t: torch.Tensor,
    rank: int = 32,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Batch of clips.

    S_s, S_t: [B, K, d] -> per-sample scalar [B]
    """
    losses = []
    for b in range(S_s.shape[0]):
        losses.append(temporal_joint_bw2(S_s[b], S_t[b], rank=rank, eps=eps))
    return torch.stack(losses)


def temporal_joint_distill_loss(
    preds_stud: torch.Tensor,
    preds_teacher: torch.Tensor,
    masks: torch.Tensor,
    per_frame_loss: torch.Tensor,
    *,
    loss_type: str,
    joint_lambda: float = 0.5,
    joint_rank: int = 32,
    joint_stat: str = "mean",
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Clip-level distillation loss (independent of single-frame forward_loss_TS).

    preds_stud / preds_teacher: [B, K, L, D]
    masks: [B, K, L]
    per_frame_loss: [B, K] per-frame scalar (MSE or BW2+MSE hybrid)
    """
    clip_frame = per_frame_loss.mean(dim=1)

    S_s = extract_frame_stats(preds_stud, masks, stat=joint_stat)
    S_t = extract_frame_stats(preds_teacher, masks, stat=joint_stat)
    clip_joint = per_clip_temporal_joint_bw2(
        S_s, S_t, rank=joint_rank, eps=eps
    )

    if loss_type == "temporal_joint":
        per_sample = clip_joint
    elif loss_type == "temporal_joint_mse":
        per_sample = clip_frame + joint_lambda * clip_joint
    elif loss_type in ("temporal_joint_bw2mse", "temporal_joint_bw2lr_mse"):
        per_sample = clip_frame + joint_lambda * clip_joint
    else:
        raise ValueError(
            f"temporal_joint_distill_loss expects temporal_joint* loss_type, got {loss_type}"
        )

    return per_sample.mean()
