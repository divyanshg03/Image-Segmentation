"""Segmentation quality metrics for boolean masks."""

from __future__ import annotations

import numpy as np

from .postprocess import as_bool_mask


def _pair(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred, gt) as flat bool arrays with any `ignore` pixels removed."""
    p, g = as_bool_mask(pred), as_bool_mask(gt)
    if p.shape != g.shape:
        raise ValueError(f"mask shapes differ: {p.shape} vs {g.shape}")
    if ignore is None:
        return p.ravel(), g.ravel()
    ig = as_bool_mask(ignore)
    if ig.shape != g.shape:
        raise ValueError(f"ignore mask shape {ig.shape} differs from mask shape {g.shape}")
    keep = ~ig
    return p[keep], g[keep]


def iou(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray | None = None) -> float:
    """Intersection over union. Two empty masks count as a perfect match (1.0).

    Pixels where `ignore` is True (e.g. an unlabelled boundary band in the ground
    truth) are excluded from both masks before scoring.
    """
    p, g = _pair(pred, gt, ignore)
    union = np.logical_or(p, g).sum()
    return 1.0 if union == 0 else float(np.logical_and(p, g).sum() / union)


def dice(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray | None = None) -> float:
    """Dice coefficient (F1). Two empty masks count as a perfect match (1.0). See `iou` for `ignore`."""
    p, g = _pair(pred, gt, ignore)
    total = p.sum() + g.sum()
    return 1.0 if total == 0 else float(2 * np.logical_and(p, g).sum() / total)
