"""SAM wrapper: encode an image once, then answer any number of prompts cheaply."""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

from .io import load_image
from .postprocess import select_candidate
from .prompts import validate_box, validate_points
from .weights import resolve_checkpoint

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    """Result of a prompted segmentation.

    `mask` is the candidate chosen by `predict(select=...)`, by default the one with
    the highest predicted IoU. `candidates`/`scores` hold every mask SAM returned
    (three for an ambiguous single-point prompt).
    """

    mask: np.ndarray  # (H, W) bool
    score: float
    candidates: np.ndarray  # (K, H, W) bool
    scores: np.ndarray  # (K,)
    index: int = 0  # which candidate `mask` is


@dataclass(frozen=True, eq=False)
class EncodedImage:
    """A cached image embedding: keep several and swap between them without re-encoding."""

    image: np.ndarray  # (H, W, 3) uint8 RGB the embedding was computed from
    features: torch.Tensor
    original_size: tuple[int, int]
    input_size: tuple[int, int]
    encode_seconds: float


@dataclass(frozen=True)
class Segment:
    """One mask from segment-everything mode."""

    mask: np.ndarray  # (H, W) bool
    score: float  # SAM's predicted IoU
    stability: float
    area: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """'auto'/None picks CUDA when available, otherwise CPU."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use device='cpu' or 'auto'")
    return dev


class Segmenter:
    """Promptable segmentation with a cached image embedding.

    The ViT image encoder dominates the cost of SAM, so `set_image` runs it once
    and every following `predict` call only runs the light prompt/mask decoder.

        seg = Segmenter()
        seg.set_image("samples/football_dribble.jpeg")
        seg.predict(points=[(355, 215)], labels=[1]).mask
        seg.predict(box=(250, 120, 420, 310)).mask      # no re-encoding
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        model_type: str = "vit_b",
        device: str | torch.device | None = None,
        fp16: bool = False,
    ):
        if model_type not in sam_model_registry:
            raise ValueError(f"model_type must be one of {sorted(sam_model_registry)}, got {model_type!r}")
        self.device = resolve_device(device)
        if fp16 and self.device.type != "cuda":
            raise ValueError("fp16 needs a CUDA device")
        self._fp16 = fp16

        path = resolve_checkpoint(checkpoint)
        log.info("Loading %s from %s onto %s", model_type, path, self.device)
        self.sam = sam_model_registry[model_type](checkpoint=str(path)).to(self.device).eval()
        self._predictor = SamPredictor(self.sam)
        self._image: np.ndarray | None = None
        self.last_encode_seconds: float | None = None

    @property
    def device_name(self) -> str:
        """Human-readable device, e.g. 'NVIDIA GeForce RTX 4060 Laptop GPU' or 'CPU'."""
        return torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"

    @property
    def fp16(self) -> bool:
        return self._fp16

    # -- image -----------------------------------------------------------------

    @property
    def image(self) -> np.ndarray:
        """The RGB array currently encoded (same pixels the masks refer to)."""
        if self._image is None:
            raise RuntimeError("No image set. Call set_image() first.")
        return self._image

    def set_image(self, image: str | Path | np.ndarray) -> Segmenter:
        """Encode an image (path or RGB uint8 array). Do this once per image."""
        if isinstance(image, (str, Path)):
            arr = load_image(image)
        else:
            arr = np.asarray(image)
            if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
                raise ValueError(f"image must be a uint8 RGB array of shape (H, W, 3), got {arr.dtype} {arr.shape}")

        start = time.perf_counter()
        with self._autocast():
            self._predictor.set_image(arr)
        self._sync()
        self.last_encode_seconds = time.perf_counter() - start
        self._image = arr
        return self

    def encode(self, image: str | Path | np.ndarray) -> EncodedImage:
        """Like `set_image`, but return the embedding so `activate` can restore it later.

        This is what lets one model serve several images (e.g. several browser tabs):
        the heavy encoder runs once per image, switching back costs nothing.
        """
        self.set_image(image)
        p = self._predictor
        return EncodedImage(self.image, p.features, p.original_size, p.input_size, self.last_encode_seconds)

    def activate(self, encoded: EncodedImage) -> Segmenter:
        """Make a previously encoded image current again (no encoder run)."""
        p = self._predictor
        p.features, p.original_size, p.input_size, p.is_image_set = (
            encoded.features, encoded.original_size, encoded.input_size, True,
        )
        self._image = encoded.image
        return self

    # -- prompted segmentation ---------------------------------------------------

    def predict(
        self, *, points=None, labels=None, box=None, multimask: bool | None = None, select: str = "score"
    ) -> Prediction:
        """Segment from points and/or a box (pixel coordinates).

        All points are ONE prompt: label 1 marks foreground, label 0 background,
        and together they refine a single object. `multimask` defaults to True only
        for a lone point (genuinely ambiguous), False otherwise. When several
        candidates come back, `select` picks which one becomes `.mask`: "score"
        (SAM's own confidence, default) or "largest"; see `select_candidate`.
        """
        size = self.image.shape[:2]
        pts, lab = validate_points(points, labels, size)
        bx = validate_box(box, size)
        if pts is None and bx is None:
            raise ValueError("Provide points and/or a box (use segment_everything() for automatic mode)")
        if multimask is None:
            multimask = pts is not None and len(pts) == 1 and bx is None

        with self._autocast():
            masks, scores, _ = self._predictor.predict(
                point_coords=pts, point_labels=lab, box=bx, multimask_output=multimask
            )
        best = select_candidate(masks, scores, select)
        return Prediction(mask=masks[best], score=float(scores[best]), candidates=masks, scores=scores, index=best)

    # -- segment everything --------------------------------------------------------

    def segment_everything(
        self,
        *,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_area: int = 100,
    ) -> list[Segment]:
        """Propose masks for everything in the image, filtered by quality, largest first.

        Masks are kept separate (never merged into one), and low-quality proposals
        are dropped by `pred_iou_thresh`, `stability_score_thresh` and `min_area`.
        Runs in fp32 and re-encodes the image internally.
        """
        generator = SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_area,
        )
        raw = generator.generate(self.image)
        segments = []
        for ann in raw:
            x, y, w, h = ann["bbox"]
            segments.append(
                Segment(
                    mask=ann["segmentation"],
                    score=float(ann["predicted_iou"]),
                    stability=float(ann["stability_score"]),
                    area=int(ann["area"]),
                    bbox=(int(x), int(y), int(x + w), int(y + h)),
                )
            )
        segments.sort(key=lambda s: s.area, reverse=True)
        return segments

    # -- helpers ---------------------------------------------------------------------

    def _autocast(self):
        if self._fp16:
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _sync(self) -> None:
        # CUDA kernels are asynchronous; wait so wall-clock timings are honest.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
