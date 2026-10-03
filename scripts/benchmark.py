"""Latency benchmark: where SAM's time goes, and what fp16 / int8 / MobileSAM buy.

    python scripts/benchmark.py                       # CUDA fp32 vs fp16, CPU fp32 vs int8
    python scripts/benchmark.py --skip-cpu            # GPU rows only (CPU ViT-B is slow)
    python scripts/benchmark.py --mobile-sam weights/mobile_sam.pt
    python scripts/benchmark.py --json bench.json

Timings are wall-clock with CUDA synchronised, after warm-up runs (the first call on a
GPU is several times slower than steady state). Every configuration segments the same
image with the same box; "IoU vs baseline" is that mask's IoU against the first row's
mask, so speed-ups come with a measured accuracy cost.

MobileSAM is measured through Ultralytics, which is imported lazily and only for that
optional row. Ultralytics is AGPL-3.0 and is not a dependency of imgseg.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from imgseg import Segmenter  # noqa: E402
from imgseg.io import load_image  # noqa: E402
from imgseg.metrics import iou  # noqa: E402

DEFAULT_IMAGE = "samples/football_dribble.jpeg"
DEFAULT_BOX = (250, 120, 420, 310)  # Messi in the default image


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stats(values_s: list[float]) -> dict:
    ms = [v * 1000 for v in values_s]
    return {"median_ms": statistics.median(ms), "min_ms": min(ms), "max_ms": max(ms)}


def measure_segmenter(seg: Segmenter, image: np.ndarray, box, runs: int, warmup: int) -> tuple[dict, np.ndarray]:
    for _ in range(warmup):
        seg.set_image(image)
        seg.predict(box=box)
    if seg.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    encode, prompt = [], []
    for _ in range(runs):
        seg.set_image(image)
        encode.append(seg.last_encode_seconds)
        start = time.perf_counter()
        pred = seg.predict(box=box)
        _sync()
        prompt.append(time.perf_counter() - start)

    result = {
        "encode": _stats(encode),
        "prompt": _stats(prompt),
        "total_ms": statistics.median(e + p for e, p in zip(encode, prompt)) * 1000,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if seg.device.type == "cuda" else None,
        "runs": runs,
    }
    return result, pred.mask


def measure_mobile_sam(weights: str, image_path: str, box, runs: int, warmup: int) -> tuple[dict, np.ndarray]:
    from ultralytics import SAM  # optional, AGPL-3.0: see module docstring

    model = SAM(weights)
    for _ in range(warmup):
        model.predict(source=image_path, bboxes=[list(box)], verbose=False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    inference = []
    for _ in range(runs):
        result = model.predict(source=image_path, bboxes=[list(box)], verbose=False)[0]
        inference.append(result.speed["inference"] / 1000)  # encoder + decoder, synchronised by Ultralytics
    mask = result.masks.data[0].cpu().numpy() > 0
    return {
        "encode": None,  # Ultralytics re-encodes on every call and does not split the two stages
        "prompt": None,
        "total_ms": statistics.median(inference) * 1000,
        "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        "runs": runs,
    }, mask


def _int8_encoder(seg: Segmenter) -> Segmenter:
    """Dynamic INT8 quantisation of the image encoder's Linear layers (CPU only)."""
    seg.sam.image_encoder = torch.ao.quantization.quantize_dynamic(
        seg.sam.image_encoder, {torch.nn.Linear}, dtype=torch.qint8
    )
    return seg


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args) -> list[dict]:
    image = load_image(args.image)
    box = tuple(args.box)
    rows: list[dict] = []
    baseline: np.ndarray | None = None

    def record(name: str, build, measure_fn):
        nonlocal baseline
        print(f"- {name} ...", file=sys.stderr, flush=True)
        try:
            result, mask = measure_fn(build())
        except Exception as exc:  # keep going: one unsupported config should not kill the table
            print(f"  failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            rows.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            return
        if baseline is None:
            baseline = mask
        result.update(name=name, iou_vs_baseline=iou(mask, baseline), mask_area=int(mask.sum()))
        rows.append(result)
        _free()

    if torch.cuda.is_available():
        record("CUDA fp32", lambda: Segmenter(device="cuda"),
               lambda s: measure_segmenter(s, image, box, args.runs, args.warmup))
        record("CUDA fp16 (autocast)", lambda: Segmenter(device="cuda", fp16=True),
               lambda s: measure_segmenter(s, image, box, args.runs, args.warmup))
        if args.mobile_sam:
            record("CUDA MobileSAM (Ultralytics)", lambda: None,
                   lambda _: measure_mobile_sam(args.mobile_sam, args.image, box, args.runs, args.warmup))
    if not args.skip_cpu:
        record("CPU fp32", lambda: Segmenter(device="cpu"),
               lambda s: measure_segmenter(s, image, box, args.cpu_runs, 1))
        record("CPU int8 encoder (dynamic)", lambda: _int8_encoder(Segmenter(device="cpu")),
               lambda s: measure_segmenter(s, image, box, args.cpu_runs, 1))
    return rows


def _fmt(stats: dict | None, key: str = "median_ms") -> str:
    return "n/a" if stats is None else f"{stats[key]:.0f}"


def to_markdown(rows: list[dict]) -> str:
    lines = [
        "| config | encode ms | prompt ms | encode + prompt ms | peak GPU MB | IoU vs baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['name']} | failed: {r['error']} | | | | |")
            continue
        peak = "-" if r["peak_gpu_mb"] is None else f"{r['peak_gpu_mb']:.0f}"
        lines.append(
            f"| {r['name']} | {_fmt(r['encode'])} | {_fmt(r['prompt'])} | {r['total_ms']:.0f} | {peak} | {r['iou_vs_baseline']:.3f} |"
        )
    return "\n".join(lines)


def hardware() -> dict:
    return {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--box", nargs=4, type=float, default=DEFAULT_BOX, metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--runs", type=int, default=20, help="timed GPU runs per config")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--cpu-runs", type=int, default=3, help="timed CPU runs per config (CPU is slow)")
    p.add_argument("--skip-cpu", action="store_true")
    p.add_argument("--mobile-sam", metavar="WEIGHTS", help="also time MobileSAM via Ultralytics (optional)")
    p.add_argument("--json", metavar="PATH", help="also write raw results here")
    args = p.parse_args()

    rows = run(args)
    hw = hardware()
    print(f"\nHardware: {hw['gpu'] or 'no GPU'} | CPU: {hw['cpu']} | torch {hw['torch']} | Python {hw['python']}")
    print(f"Image: {args.image} | box: {list(map(int, args.box))} | GPU runs: {args.runs}, CPU runs: {args.cpu_runs}\n")
    print(to_markdown(rows))
    if args.json:
        Path(args.json).write_text(json.dumps({"hardware": hw, "image": args.image, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
