"""Server-side logic of the demo UI, free of web-framework imports so it can be tested alone.

One `DemoEngine` owns one model. Each uploaded image becomes a *session* holding its cached
embedding (`Segmenter.encode`), so any number of browser tabs can share the model and every
click after the first costs only the light mask decoder. Access to the model is serialised
with a lock; the UI is meant for one person on one machine, not for serving traffic.
"""

from __future__ import annotations

import base64
import io
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..evaluate import IMAGE_EXTS
from ..io import load_image
from ..postprocess import cutout, instance_labels, mask_bbox
from ..render import mask_to_rgba, overlay

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = REPO_ROOT / "samples"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DECODE_PIXELS = 80_000_000  # refuse decompression bombs before decoding
MAX_POINTS = 64
EXPORT_KINDS = ("mask", "cutout", "cutout-crop", "overlay")

ACCENT = (30, 144, 255)  # committed mask
PREVIEW = (255, 176, 0)  # hover preview


class DemoError(Exception):
    """A problem the user can fix; carries the HTTP status the server should answer with."""

    status = 400


class UnknownSession(DemoError):
    status = 404


class ImageRejected(DemoError):
    status = 400


def png_bytes(array: np.ndarray, level: int = 1) -> bytes:
    """Fast PNG encoding (low compression: these go over localhost)."""
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG", compress_level=level)
    return buf.getvalue()


def _b64(array: np.ndarray) -> str:
    return base64.b64encode(png_bytes(array)).decode("ascii")


def decode_upload(data: bytes, max_side: int) -> np.ndarray:
    """Decode uploaded bytes to an RGB array: EXIF-rotated, downscaled to `max_side`."""
    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        if width * height > MAX_DECODE_PIXELS:
            raise ImageRejected(f"image is too large ({width}x{height} pixels)")
        image = ImageOps.exif_transpose(image).convert("RGB")
    except ImageRejected:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ImageRejected("could not read that file as an image") from exc
    return _fit(np.asarray(image), max_side)


def _fit(image: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longer side is at most `max_side` (SAM works at 1024 anyway)."""
    height, width = image.shape[:2]
    if max(height, width) <= max_side:
        return image
    scale = max_side / max(height, width)
    resized = Image.fromarray(image).resize((round(width * scale), round(height * scale)), Image.LANCZOS)
    return np.asarray(resized)


def _safe_stem(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._") or "image"


@dataclass
class _Session:
    name: str
    encoded: object  # imgseg.model.EncodedImage
    mask: np.ndarray | None = None  # the last committed (non-preview) mask
    segments: list | None = None  # result of the last segment-everything call


class DemoEngine:
    def __init__(self, segmenter, *, max_side: int = 1600, max_sessions: int = 8, samples_dir: Path | None = SAMPLES_DIR):
        self.seg = segmenter
        self.max_side = max_side
        self.max_sessions = max_sessions
        self.samples_dir = Path(samples_dir) if samples_dir else None
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._lock = threading.RLock()

    # -- info ------------------------------------------------------------------------------

    def sample_names(self) -> list[str]:
        if not self.samples_dir or not self.samples_dir.is_dir():
            return []
        return sorted(p.name for p in self.samples_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    def sample_path(self, name: str) -> Path:
        if name not in self.sample_names():  # whitelist: no path tricks possible
            raise UnknownSession(f"no such sample: {name!r}")
        return self.samples_dir / name

    def info(self) -> dict:
        device = getattr(self.seg, "device", "cpu")
        return {
            "device": getattr(device, "type", str(device)),
            "device_name": getattr(self.seg, "device_name", str(device)),
            "fp16": bool(getattr(self.seg, "fp16", False)),
            "max_side": self.max_side,
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "samples": self.sample_names(),
        }

    def warm_up(self) -> float:
        """Run the encoder and both prompt types once on noise, discarding the results.

        The first GPU call pays one-off kernel set-up (about 1.5 s for the encoder and about 1 s for the
        decoder on an RTX 4060). Doing it at start-up means a user's first hover or click is already fast.
        Returns the seconds spent. No session is created.
        """
        noise = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
        start = time.perf_counter()
        with self._lock:
            self.seg.activate(self.seg.encode(noise))
            self.seg.predict(points=[(128, 128)], labels=[1])
            self.seg.predict(box=(32, 32, 224, 224))
        return time.perf_counter() - start

    # -- sessions -----------------------------------------------------------------------------

    def open_bytes(self, data: bytes, name: str) -> dict:
        return self.open_image(decode_upload(data, self.max_side), name)

    def open_sample(self, name: str) -> dict:
        path = self.sample_path(name)
        return self.open_image(_fit(load_image(path), self.max_side), name)

    def open_image(self, image: np.ndarray, name: str) -> dict:
        with self._lock:
            encoded = self.seg.encode(image)
            sid = uuid.uuid4().hex
            self._sessions[sid] = _Session(name=name, encoded=encoded)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)  # forget the least recently used image
        height, width = image.shape[:2]
        return {"session": sid, "width": width, "height": height, "name": name,
                "encode_ms": round(encoded.encode_seconds * 1000, 1)}

    def _get(self, sid: str) -> _Session:
        with self._lock:
            session = self._sessions.get(sid)
            if session is None:
                raise UnknownSession("this image is no longer loaded; open it again")
            self._sessions.move_to_end(sid)
            return session

    def image_png(self, sid: str) -> bytes:
        return png_bytes(self._get(sid).encoded.image, level=3)

    # -- segmentation -----------------------------------------------------------------------------

    def segment(self, sid: str, *, points=(), box=None, select: str = "score", choose: int | None = None,
                preview: bool = False) -> dict:
        """Segment from the prompts. `preview=True` is a throwaway hover preview: nothing is stored."""
        if len(points) > MAX_POINTS:
            raise DemoError(f"too many points (max {MAX_POINTS})")
        if not points and box is None:
            raise DemoError("add a point or draw a box first")
        session = self._get(sid)
        pts = [(x, y) for x, y, _ in points] or None
        labels = [int(label) for _, _, label in points] or None

        with self._lock:
            self.seg.activate(session.encoded)
            start = time.perf_counter()
            try:
                pred = self.seg.predict(points=pts, labels=labels, box=box, select=select)
            except ValueError as exc:
                raise DemoError(str(exc)) from exc
            ms = (time.perf_counter() - start) * 1000

        index, mask, score = pred.index, pred.mask, pred.score
        if choose is not None and len(pred.candidates) > 1:
            if choose >= len(pred.candidates):
                raise DemoError(f"candidate {choose} does not exist")
            index, mask, score = choose, pred.candidates[choose], float(pred.scores[choose])

        color = PREVIEW if preview else ACCENT
        out = {"mask": _b64(mask_to_rgba(mask, color, alpha=0.35 if preview else 0.45)),
               "score": float(score), "ms": round(ms, 1)}
        if preview:
            return out

        session.mask = mask
        out.update(area=int(mask.sum()), bbox=mask_bbox(mask))
        if len(pred.candidates) > 1:
            out["chosen"] = index
            out["candidates"] = [
                {"mask": _b64(mask_to_rgba(c, ACCENT, alpha=0.45)), "score": float(sc), "area": int(c.sum()),
                 "bbox": mask_bbox(c)}
                for c, sc in zip(pred.candidates, pred.scores)
            ]
        return out

    def everything(self, sid: str, *, points_per_side: int = 32) -> dict:
        """Segment everything: returns a label map (R = id low byte, G = id high byte; 0 = none)."""
        session = self._get(sid)
        with self._lock:
            self.seg.activate(session.encoded)
            start = time.perf_counter()
            segments = self.seg.segment_everything(points_per_side=points_per_side)
            ms = (time.perf_counter() - start) * 1000
        session.segments = segments
        if not segments:
            return {"count": 0, "labels": None, "ms": round(ms, 1)}
        labels = instance_labels([s.mask for s in segments])  # largest first, so small ones stay on top
        rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
        rgb[..., 0] = labels & 0xFF
        rgb[..., 1] = labels >> 8
        return {"count": len(segments), "labels": _b64(rgb), "ms": round(ms, 1)}

    def pick(self, sid: str, segment_id: int) -> dict:
        """Make one segment-everything result the current mask."""
        session = self._get(sid)
        if not session.segments or not 1 <= segment_id <= len(session.segments):
            raise DemoError("unknown segment; run segment everything first")
        segment = session.segments[segment_id - 1]
        session.mask = segment.mask
        return {"mask": _b64(mask_to_rgba(segment.mask, ACCENT)), "score": segment.score,
                "area": int(segment.area), "bbox": mask_bbox(segment.mask), "ms": 0.0}

    # -- export -------------------------------------------------------------------------------------

    def export(self, sid: str, kind: str) -> tuple[bytes, str]:
        if kind not in EXPORT_KINDS:
            raise DemoError(f"kind must be one of {EXPORT_KINDS}")
        session = self._get(sid)
        if session.mask is None or not session.mask.any():
            raise DemoError("there is no mask to export yet")
        image, mask = session.encoded.image, session.mask
        if kind == "mask":
            array = mask.astype(np.uint8) * 255
        elif kind == "overlay":
            array = overlay(image, mask)
        else:
            array = cutout(image, mask, crop=(kind == "cutout-crop"))
        return png_bytes(array, level=3), f"{_safe_stem(session.name)}_{kind}.png"
