"""Turn masks and prompts into pictures (numpy arrays you can show or save)."""

from __future__ import annotations

import colorsys
from collections.abc import Sequence

import cv2
import numpy as np

from .postprocess import as_bool_mask


def _check(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = as_bool_mask(mask)
    if image.shape[:2] != m.shape:
        raise ValueError(f"image {image.shape[:2]} and mask {m.shape} must have the same height and width")
    return m


def _blend(canvas: np.ndarray, mask: np.ndarray, color, alpha: float, outline: bool) -> None:
    """Blend `color` into `canvas` (float32, modified in place) where `mask` is set."""
    canvas[mask] = (1 - alpha) * canvas[mask] + alpha * np.asarray(color, dtype=np.float32)
    if outline:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edge = np.ascontiguousarray(canvas.astype(np.uint8))
        cv2.drawContours(edge, contours, -1, tuple(int(c) for c in color), 2)
        canvas[...] = edge


def overlay(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha: float = 0.5, outline: bool = True) -> np.ndarray:
    """Tint the masked region of `image` with a translucent colour."""
    m = _check(image, mask)
    canvas = image.astype(np.float32)
    _blend(canvas, m, color, alpha, outline)
    return canvas.astype(np.uint8)


def mask_to_rgba(mask: np.ndarray, color=(30, 144, 255), alpha: float = 0.45, outline: bool = True) -> np.ndarray:
    """A transparent RGBA layer: `color` at `alpha` inside the mask, solid on a 2 px outline.

    Unlike `overlay` this does not need the image, so a browser can composite it over
    the picture itself (and tint, hide or reorder layers).
    """
    m = as_bool_mask(mask)
    layer = np.zeros((*m.shape, 4), dtype=np.uint8)
    layer[m] = (*color, int(round(255 * alpha)))
    if outline:
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edge = np.zeros(m.shape, dtype=np.uint8)
        cv2.drawContours(edge, contours, -1, 255, 2)
        layer[edge > 0] = (*color, 255)
    return layer


def instance_colors(n: int) -> list[tuple[int, int, int]]:
    """n visually distinct, deterministic RGB colours."""
    golden = 0.61803398875
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * golden) % 1.0, 0.65, 1.0)) for i in range(n)]


def overlay_instances(image: np.ndarray, masks: Sequence[np.ndarray], alpha: float = 0.55, outline: bool = True) -> np.ndarray:
    """Overlay several masks, each in its own colour (later masks are drawn on top)."""
    canvas = image.astype(np.float32)
    for mask, color in zip(masks, instance_colors(len(masks))):
        _blend(canvas, _check(image, mask), color, alpha, outline)
    return canvas.astype(np.uint8)


def draw_prompts(image: np.ndarray, points=None, labels=None, box=None) -> np.ndarray:
    """Draw prompts on a copy of `image`: green dots = foreground, red = background, yellow box."""
    out = np.ascontiguousarray(image.copy())
    h, w = out.shape[:2]
    radius = max(4, round(min(h, w) * 0.014))
    thick = max(1, radius // 3)

    if box is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in np.asarray(box, dtype=float).reshape(4))
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 220, 0), thick + 1)
    if points is not None:
        for (x, y), label in zip(np.asarray(points, dtype=float), np.asarray(labels).reshape(-1)):
            center = (int(round(x)), int(round(y)))
            cv2.circle(out, center, radius + thick, (255, 255, 255), -1)
            cv2.circle(out, center, radius, (0, 200, 0) if label == 1 else (230, 0, 0), -1)
    return out
