"""Validation of point and box prompts (pure numpy, no model required)."""

from __future__ import annotations

import numpy as np

Size = tuple[int, int]  # (height, width)


def validate_points(points, labels, size: Size):
    """Check point prompts and return (points float32 (N, 2), labels int64 (N,)).

    Points are (x, y) pixel coordinates; labels are 1 (foreground) or 0
    (background). Returns (None, None) when no points were given.
    """
    if points is None:
        if labels is not None:
            raise ValueError("labels were given without points")
        return None, None

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        raise ValueError(f"points must have shape (N, 2) with N >= 1, got {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("points must be finite numbers")
    if labels is None:
        raise ValueError("labels are required with points (1 = foreground, 0 = background)")

    lab = np.asarray(labels)
    if lab.shape != (len(pts),):
        raise ValueError(f"labels must have shape ({len(pts)},) to match points, got {lab.shape}")
    if not np.isin(lab, (0, 1)).all():
        raise ValueError("labels must be 0 (background) or 1 (foreground)")

    h, w = size
    outside = (pts[:, 0] < 0) | (pts[:, 0] >= w) | (pts[:, 1] < 0) | (pts[:, 1] >= h)
    if outside.any():
        raise ValueError(f"point(s) outside the {w}x{h} image: {pts[outside].tolist()}")
    return pts, lab.astype(np.int64)


def validate_box(box, size: Size):
    """Check a (x1, y1, x2, y2) box and return it as float32 (4,), clipped to the image."""
    if box is None:
        return None

    b = np.asarray(box, dtype=np.float32).reshape(-1)
    if b.shape != (4,):
        raise ValueError(f"box must be (x1, y1, x2, y2), got {b.size} values")
    if not np.isfinite(b).all():
        raise ValueError("box must contain finite numbers")

    h, w = size
    x1, y1, x2, y2 = b
    if not (x1 < x2 and y1 < y2):
        raise ValueError(f"box must have x1 < x2 and y1 < y2, got {b.tolist()}")
    clipped = np.array([max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)], dtype=np.float32)
    if not (clipped[0] < clipped[2] and clipped[1] < clipped[3]):
        raise ValueError(f"box {b.tolist()} lies outside the {w}x{h} image")
    return clipped
