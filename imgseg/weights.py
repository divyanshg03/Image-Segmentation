"""Locating and downloading SAM checkpoints."""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPO_ROOT / "weights"
ENV_VAR = "IMGSEG_CHECKPOINT"

# Meta's official ViT-B checkpoint. `sam_b.pt` (the name Ultralytics uses) is the
# same size and loads identically, so either file name is accepted.
OFFICIAL_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
OFFICIAL_NAME = "sam_vit_b_01ec64.pth"
OFFICIAL_SIZE = 375_042_383
SEARCH_NAMES = (OFFICIAL_NAME, "sam_b.pt")


def resolve_checkpoint(checkpoint: str | Path | None = None) -> Path:
    """Find a checkpoint: explicit path, then $IMGSEG_CHECKPOINT, then weights/."""
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    env = os.environ.get(ENV_VAR)
    if env:
        path = Path(env)
        if not path.is_file():
            raise FileNotFoundError(f"${ENV_VAR} points to a missing file: {path}")
        return path

    for name in SEARCH_NAMES:
        path = WEIGHTS_DIR / name
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"No SAM checkpoint found. Put {OFFICIAL_NAME} (or sam_b.pt) in {WEIGHTS_DIR}, "
        f"set ${ENV_VAR}, pass checkpoint=..., or run: python -m imgseg download-weights"
    )


def download_weights(dest_dir: str | Path = WEIGHTS_DIR) -> Path:
    """Download the official ViT-B checkpoint (~375 MB) unless it is already there."""
    dest = Path(dest_dir) / OFFICIAL_NAME
    if dest.is_file() and dest.stat().st_size == OFFICIAL_SIZE:
        log.info("Already downloaded: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading %s", OFFICIAL_URL)
    next_report = 10
    with urllib.request.urlopen(OFFICIAL_URL, timeout=60) as resp, open(part, "wb") as out:
        total = int(resp.headers.get("Content-Length", OFFICIAL_SIZE))
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if done * 100 // total >= next_report:
                log.info("  %d%%", done * 100 // total)
                next_report += 10

    if part.stat().st_size != OFFICIAL_SIZE:
        size = part.stat().st_size
        part.unlink()
        raise OSError(f"Downloaded file has {size} bytes, expected {OFFICIAL_SIZE}; try again")
    part.replace(dest)
    log.info("Saved %s", dest)
    return dest
