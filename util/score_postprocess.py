"""Score post-processing for video-level anomaly detection."""

from __future__ import annotations

import numpy as np


def apply_temporal_peak_pooling(scores, window_size: int = 1) -> np.ndarray:
    """Per-video 1D max pooling: keep a peak if any neighbor in the window is high."""
    scores = np.asarray(scores, dtype=np.float64)
    if window_size is None or window_size <= 1:
        return scores
    half = window_size // 2
    out = np.empty_like(scores)
    for i in range(len(scores)):
        lo = max(0, i - half)
        hi = min(len(scores), i + half + 1)
        out[i] = scores[lo:hi].max()
    return out
