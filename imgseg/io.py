"""Image and mask I/O.

All images in this package are RGB uint8 arrays of shape (H, W, 3) and all masks
are boolean arrays of shape (H, W). Everything is loaded through `load_image`,
so the model, the renderer and the evaluator always see the same pixels
(including EXIF orientation, which PIL ignores by default but OpenCV applies).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as an RGB uint8 array, honouring its EXIF orientation."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def output_path(image_path: str | Path, mode: str, kind: str, out_dir: str | Path = "outputs") -> Path:
    """Where to write one artefact of a run, e.g. outputs/photo_box_mask.png.

    The prompt mode is part of the name so a point run, a box run and an auto
    run on the same image never overwrite each other.
    """
    return Path(out_dir) / f"{Path(image_path).stem}_{mode}_{kind}.png"


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    """Save a boolean mask as a 0/255 single-channel PNG."""
    return save_image((np.asarray(mask, dtype=bool).astype(np.uint8) * 255), path)


def save_image(array: np.ndarray, path: str | Path) -> Path:
    """Save an array as an image, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


def load_ground_truth(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a ground-truth mask as (foreground, ignore) bool arrays.

    Convention (the one used by the GrabCut family of benchmarks): 255 = foreground,
    0 = background, mid-grey (64-191, typically 128) = unlabelled boundary pixels that
    must not count for or against a prediction.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask not found: {path}")
    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"))
    return gray >= 192, (gray >= 64) & (gray < 192)


def load_mask(path: str | Path) -> np.ndarray:
    """Load a mask PNG (any nonzero-above-127 pixel is foreground) as a bool array."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask not found: {path}")
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 127
