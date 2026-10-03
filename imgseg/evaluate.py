"""Quantitative evaluation against ground-truth masks.

Protocol: derive ONE prompt from each ground-truth mask (its tight box, or the
foreground pixel farthest from the mask boundary as a single click, in the spirit of
the SAM paper's single-point evaluation), segment, and score the result with IoU and
Dice. This is a single-prompt score, not a multi-click interactive metric.

    images/  cat.jpg  dog.png ...
    masks/   cat.png  dog.png ...      (same file stem)

Mask convention: 255 = foreground, 0 = background, mid-grey (e.g. 128) = unlabelled
boundary pixels, which are excluded from scoring (see `io.load_ground_truth`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .io import load_ground_truth, load_image
from .metrics import dice, iou
from .postprocess import as_bool_mask, mask_bbox

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROMPTS = ("box", "point")


def match_pairs(images_dir: str | Path, masks_dir: str | Path) -> list[tuple[Path, Path]]:
    """Pair each image with the mask PNG that has the same stem."""
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    for d in (images_dir, masks_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Not a directory: {d}")

    pairs = []
    for image in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        mask = masks_dir / f"{image.stem}.png"
        if mask.is_file():
            pairs.append((image, mask))
        else:
            log.warning("No mask for %s (expected %s); skipping", image.name, mask.name)
    if not pairs:
        raise ValueError(f"No image/mask pairs found in {images_dir} and {masks_dir}")
    return pairs


def box_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Tight (x1, y1, x2, y2) prompt box around a ground-truth mask."""
    box = mask_bbox(mask)
    if box is None:
        raise ValueError("mask is empty")
    return box


def interior_point(mask: np.ndarray) -> tuple[int, int]:
    """The foreground pixel deepest inside the mask, as (x, y).

    Unlike the centroid this always lies on the object, even for rings or
    crescents, so it is a fair stand-in for a user's click.
    """
    m = as_bool_mask(mask)
    if not m.any():
        raise ValueError("mask is empty")
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x), int(y)


@dataclass(frozen=True)
class EvalRow:
    name: str
    iou: float
    dice: float
    oracle_iou: float | None = None  # best IoU among all candidates SAM returned (None = single candidate)

    @property
    def best_iou(self) -> float:
        return self.iou if self.oracle_iou is None else max(self.iou, self.oracle_iou)


@dataclass
class EvalReport:
    prompt: str
    rows: list[EvalRow] = field(default_factory=list)
    select: str = "score"

    @property
    def mean_iou(self) -> float:
        return float(np.mean([r.iou for r in self.rows])) if self.rows else float("nan")

    @property
    def median_iou(self) -> float:
        return float(np.median([r.iou for r in self.rows])) if self.rows else float("nan")

    @property
    def mean_dice(self) -> float:
        return float(np.mean([r.dice for r in self.rows])) if self.rows else float("nan")

    @property
    def mean_oracle_iou(self) -> float:
        """Upper bound: mean IoU if the best of SAM's candidates were always picked."""
        return float(np.mean([r.best_iou for r in self.rows])) if self.rows else float("nan")

    def label(self) -> str:
        return self.prompt if self.prompt == "box" else f"{self.prompt}, select={self.select}"

    def share(self, *, at_least: float | None = None, below: float | None = None) -> float:
        """Fraction of images with IoU >= `at_least` (or < `below`)."""
        if not self.rows:
            return float("nan")
        if (at_least is None) == (below is None):
            raise ValueError("pass exactly one of at_least / below")
        hits = [(r.iou >= at_least) if at_least is not None else (r.iou < below) for r in self.rows]
        return float(np.mean(hits))

    def worst(self, n: int = 5) -> list[EvalRow]:
        return sorted(self.rows, key=lambda r: r.iou)[:n]

    def to_markdown(self) -> str:
        lines = [f"Prompt: {self.label()}", "", "| image | IoU | Dice |", "|---|---|---|"]
        lines += [f"| {r.name} | {r.iou:.3f} | {r.dice:.3f} |" for r in self.rows]
        lines.append(f"| **mean ({len(self.rows)})** | **{self.mean_iou:.3f}** | **{self.mean_dice:.3f}** |")
        return "\n".join(lines)

    def to_summary(self, worst: int = 5) -> str:
        if not self.rows:
            return f"Prompt: {self.label()}\nNo images were scored."
        lines = [
            f"Prompt: {self.label()} | images scored: {len(self.rows)}",
            f"mean IoU {self.mean_iou:.3f} | median IoU {self.median_iou:.3f} | mean Dice {self.mean_dice:.3f}"
            f" | best-of-candidates upper bound {self.mean_oracle_iou:.3f}",
            f"IoU >= 0.9 on {self.share(at_least=0.9):.0%} of images | IoU < 0.5 on {self.share(below=0.5):.0%}",
            f"lowest {worst}: " + ", ".join(f"{r.name} ({r.iou:.2f})" for r in self.worst(worst)),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "select": self.select,
            "mean_iou": self.mean_iou,
            "median_iou": self.median_iou,
            "mean_dice": self.mean_dice,
            "mean_oracle_iou": self.mean_oracle_iou,
            "rows": [
                {"name": r.name, "iou": r.iou, "dice": r.dice, "oracle_iou": r.best_iou} for r in self.rows
            ],
        }


def evaluate(segmenter, pairs: list[tuple[Path, Path]], prompt: str = "box", select: str = "score") -> EvalReport:
    """Score `segmenter` on (image, ground-truth mask) pairs.

    `segmenter` needs `set_image(image)` and `predict(points=, labels=, box=, select=)`.
    `select` chooses among SAM's candidate masks (see `postprocess.select_candidate`);
    each row also records the best candidate's IoU as an upper bound. Images whose
    ground-truth mask is empty are skipped (there is nothing to prompt with).
    """
    if prompt not in PROMPTS:
        raise ValueError(f"prompt must be one of {PROMPTS}, got {prompt!r}")

    report = EvalReport(prompt=prompt, select=select)
    for image_path, mask_path in pairs:
        gt, ignore = load_ground_truth(mask_path)
        image = load_image(image_path)
        if image.shape[:2] != gt.shape:
            raise ValueError(
                f"{image_path.name}: image is {image.shape[1]}x{image.shape[0]} but its mask is "
                f"{gt.shape[1]}x{gt.shape[0]}"
            )
        if not gt.any():
            log.warning("Ground-truth mask for %s is empty; skipping", image_path.name)
            continue

        segmenter.set_image(image)
        if prompt == "box":
            pred = segmenter.predict(box=box_from_mask(gt), select=select)
        else:
            pred = segmenter.predict(points=[interior_point(gt)], labels=[1], select=select)

        candidates = getattr(pred, "candidates", None)
        oracle = max(iou(c, gt, ignore) for c in candidates) if candidates is not None and len(candidates) > 1 else None
        report.rows.append(
            EvalRow(image_path.name, iou(pred.mask, gt, ignore), dice(pred.mask, gt, ignore), oracle)
        )
    return report
