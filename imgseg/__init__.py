"""Promptable image segmentation on top of Meta's Segment Anything Model (SAM)."""

from .io import load_image

__version__ = "0.1.0"
__all__ = ["Segmenter", "Prediction", "Segment", "EncodedImage", "load_image"]

_LAZY = {"Segmenter", "Prediction", "Segment", "EncodedImage"}


def __getattr__(name):
    # Importing the model pulls in torch; keep `import imgseg` (and the pure
    # numpy helpers) cheap by loading it only when it is actually used.
    if name in _LAZY:
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module 'imgseg' has no attribute {name!r}")
