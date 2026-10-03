"""Mask post-processing: cleanup, cut-outs and instance label maps."""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np


def as_bool_mask(mask: np.ndarray) -> np.ndarray:
    """Normalise a mask to a 2-D bool array (any nonzero value is foreground)."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D (H, W), got shape {mask.shape}")
    return mask if mask.dtype == bool else mask != 0


SELECTIONS = ("score", "largest")


def select_candidate(candidates: np.ndarray, scores: np.ndarray, select: str = "score") -> int:
    """Index of the candidate mask to return when SAM offers several.

    "score" trusts SAM's predicted IoU (its own recommendation). "largest" takes the
    biggest mask, which suits "segment the whole object" and is a poor choice when you
    clicked on a part (a shirt) and want only that part. On the GeoStar-151 benchmark a
    single click scored mean IoU 0.55 with "score" and 0.79 with "largest".
    """
    if select not in SELECTIONS:
        raise ValueError(f"select must be one of {SELECTIONS}, got {select!r}")
    candidates = np.asarray(candidates)
    if select == "score":
        return int(np.argmax(scores))
    return int(np.argmax(candidates.reshape(len(candidates), -1).sum(axis=1)))


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight (x1, y1, x2, y2) box around a mask (x2/y2 exclusive), or None if empty."""
    m = as_bool_mask(mask)
    rows = np.flatnonzero(m.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(m.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def clean_mask(mask: np.ndarray, min_area: int = 500, keep: str = "all") -> np.ndarray:
    """Remove stray fragments from a mask.

    Connected components (8-connectivity) smaller than `min_area` pixels are
    dropped. With keep="largest" only the biggest remaining component is kept;
    with keep="all" (default) every component that passes `min_area` is kept, so
    multi-object masks survive. Returns a bool mask, empty if nothing qualifies.
    """
    if keep not in ("all", "largest"):
        raise ValueError(f"keep must be 'all' or 'largest', got {keep!r}")

    m = as_bool_mask(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return m

    areas = stats[1:, cv2.CC_STAT_AREA]
    ids = 1 + np.flatnonzero(areas >= min_area)
    if keep == "largest" and ids.size:
        ids = ids[[int(np.argmax(areas[ids - 1]))]]
    return np.isin(labels, ids)


def cutout(image: np.ndarray, mask: np.ndarray, *, crop: bool = False, feather: float = 0.0) -> np.ndarray:
    """Return an RGBA image whose alpha channel is the mask.

    crop=True trims the result to the mask's bounding box. feather > 0 softens the
    edge with a Gaussian of that sigma (in pixels).
    """
    m = as_bool_mask(mask)
    if image.shape[:2] != m.shape:
        raise ValueError(f"image {image.shape[:2]} and mask {m.shape} must have the same height and width")

    alpha = m.astype(np.uint8) * 255
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
    rgba = np.dstack([image, alpha])

    if crop:
        box = mask_bbox(m)
        if box is None:
            raise ValueError("cannot crop to an empty mask")
        pad = math.ceil(3 * feather)
        x1, y1, x2, y2 = box
        h, w = m.shape
        rgba = rgba[max(y1 - pad, 0) : min(y2 + pad, h), max(x1 - pad, 0) : min(x2 + pad, w)]
    return rgba


def instance_labels(masks: Sequence[np.ndarray]) -> np.ndarray:
    """Merge masks into one uint16 label map (0 = background, i+1 = masks[i]).

    Later masks win where they overlap, so pass them largest-first to keep small
    objects visible on top of large ones.
    """
    if len(masks) == 0:
        raise ValueError("need at least one mask")
    if len(masks) > np.iinfo(np.uint16).max:
        raise ValueError("too many masks for a uint16 label map")
    labels = np.zeros(as_bool_mask(masks[0]).shape, dtype=np.uint16)
    for i, mask in enumerate(masks, start=1):
        labels[as_bool_mask(mask)] = i
    return labels
