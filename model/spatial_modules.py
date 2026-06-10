"""Spatial-aware modules for Route-A AED-MAE extensions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class PatchAttentionScoreHead(nn.Module):
    """Learnable patch attention readout for frame-level anomaly scoring."""

    def __init__(self, in_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.frame_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, patch_feats: torch.Tensor):
        """
        patch_feats: [B, L, D] per-patch score features (teacher, ts, map).
        Returns readout_score [B], frame_logit [B], attn [B, L].
        """
        attn_logits = self.attn_mlp(patch_feats).squeeze(-1)
        attn = F.softmax(attn_logits, dim=-1)
        pooled = torch.einsum("bl,bld->bd", attn, patch_feats)
        frame_logit = self.frame_head(pooled).squeeze(-1)
        readout = (attn * patch_feats[..., 0]).sum(dim=-1)
        return readout, frame_logit, attn


def pool_map_to_patches(tensor: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Average-pool a [B, C, H, W] map to [B, L] patch grid."""
    pooled = F.avg_pool2d(tensor, patch_size)
    if pooled.shape[1] == 1:
        pooled = pooled.squeeze(1)
        return rearrange(pooled, "b h w -> b (h w)")
    return rearrange(pooled, "b c h w -> b (h w)")


def normalize_patch_features(feats: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample min-max normalize each feature channel over patches."""
    lo = feats.amin(dim=1, keepdim=True)
    hi = feats.amax(dim=1, keepdim=True)
    return (feats - lo) / (hi - lo + eps)


def build_foreground_patch_mask(
    grad_mask: torch.Tensor,
    target: torch.Tensor,
    patch_size: int,
    grad_threshold: float = 0.35,
    map_threshold: float = 0.1,
) -> torch.Tensor:
    """
    Foreground mask on patch grid in [0, 1].
    Union of high-motion patches and pseudo-anomaly map patches.
    """
    grad = grad_mask.float()
    if grad.shape[1] > 1:
        grad = grad.mean(dim=1, keepdim=True)
    grad_p = pool_map_to_patches(grad, patch_size)
    grad_p = grad_p / (grad_p.amax(dim=1, keepdim=True) + 1e-6)
    grad_fg = (grad_p >= grad_threshold).float()

    gt_map = target[:, 3:4]
    gt_map = ((gt_map + 1.0) * 0.5).clamp(0.0, 1.0)
    map_p = pool_map_to_patches(gt_map, patch_size)
    map_fg = (map_p >= map_threshold).float()

    return torch.clamp(grad_fg + map_fg, 0.0, 1.0)
