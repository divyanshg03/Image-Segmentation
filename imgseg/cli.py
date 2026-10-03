"""Command line interface:  python -m imgseg <command> ...  (or `imgseg ...` once installed)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import evaluate as evaluation
from . import io, postprocess, render
from .postprocess import SELECTIONS
from .weights import download_weights

log = logging.getLogger("imgseg")


def parse_point(text: str) -> tuple[float, float, int]:
    """'X,Y' (foreground) or 'X,Y,LABEL' with LABEL 1 (foreground) or 0 (background)."""
    parts = text.split(",")
    try:
        if len(parts) not in (2, 3):
            raise ValueError
        x, y = float(parts[0]), float(parts[1])
        label = int(parts[2]) if len(parts) == 3 else 1
        if label not in (0, 1):
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid point {text!r}: expected X,Y or X,Y,LABEL with LABEL 0 or 1") from None
    return x, y, label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imgseg", description="Promptable image segmentation with SAM.")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    model_args = argparse.ArgumentParser(add_help=False)
    model_args.add_argument("--checkpoint", help="SAM checkpoint (default: weights/ or $IMGSEG_CHECKPOINT)")
    model_args.add_argument("--device", default="auto", help="auto, cpu, cuda (default: auto)")
    model_args.add_argument("--out-dir", default="outputs", help="where to write results (default: outputs)")

    clean_args = argparse.ArgumentParser(add_help=False)
    clean_args.add_argument("--clean", choices=("none", "largest"), default="none",
                            help="'largest' keeps only the biggest connected region (default: none)")
    clean_args.add_argument("--min-area", type=int, default=500, help="drop fragments below this many pixels when cleaning")
    clean_args.add_argument("--cutout", action="store_true", help="also save an RGBA cut-out of the object")
    clean_args.add_argument("--crop", action="store_true", help="crop the cut-out to the object's bounding box")
    clean_args.add_argument("--feather", type=float, default=0.0, help="soften cut-out edges (Gaussian sigma in px)")

    p = sub.add_parser("point", parents=[model_args, clean_args], help="segment from point prompts")
    p.add_argument("image")
    p.add_argument("--point", dest="points", action="append", required=True, type=parse_point, metavar="X,Y[,LABEL]",
                   help="repeat for several points; all points form ONE prompt (label 0 = background)")
    p.add_argument("--select", choices=SELECTIONS, default="score",
                   help="which candidate a lone click returns: SAM's top-scoring one, or the largest (default: score)")

    b = sub.add_parser("box", parents=[model_args, clean_args], help="segment from a bounding box")
    b.add_argument("image")
    b.add_argument("--box", nargs=4, type=float, required=True, metavar=("X1", "Y1", "X2", "Y2"))

    a = sub.add_parser("auto", parents=[model_args], help="segment everything, keeping masks separate")
    a.add_argument("image")
    a.add_argument("--points-per-side", type=int, default=32)
    a.add_argument("--pred-iou-thresh", type=float, default=0.88)
    a.add_argument("--stability-thresh", type=float, default=0.95)
    a.add_argument("--min-area", type=int, default=100, help="drop masks smaller than this many pixels")

    e = sub.add_parser("eval", parents=[model_args], help="score against ground-truth masks (IoU / Dice)")
    e.add_argument("--images", required=True, help="directory of images")
    e.add_argument("--masks", required=True, help="directory of ground-truth PNG masks with matching file stems")
    e.add_argument("--prompt", choices=evaluation.PROMPTS, default="box", help="prompt derived from each mask")
    e.add_argument("--select", choices=SELECTIONS, default="score", help="candidate choice for the point prompt")
    e.add_argument("--per-image", action="store_true", help="print the full per-image table instead of a summary")
    e.add_argument("--json", metavar="PATH", help="also write the results to this JSON file")

    d = sub.add_parser("demo", help="launch the interactive browser UI (needs the [demo] extra)")
    d.add_argument("--checkpoint", help="SAM checkpoint (default: weights/ or $IMGSEG_CHECKPOINT)")
    d.add_argument("--device", default="auto", help="auto, cpu, cuda (default: auto)")
    d.add_argument("--fp16", action="store_true", help="half precision on CUDA: about 1.5x faster, same masks")
    d.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: this machine only)")
    d.add_argument("--port", type=int, default=8000)
    d.add_argument("--max-side", type=int, default=1600, help="downscale uploads so the longer side is at most this")
    d.add_argument("--open", action="store_true", help="open the page in your browser")

    sub.add_parser("download-weights", help="download the official ViT-B checkpoint into weights/")
    return parser


def _run_prompted(args) -> int:
    from .model import Segmenter

    image = io.load_image(args.image)  # fail fast on a bad path, before the model loads
    seg = Segmenter(args.checkpoint, device=args.device)
    seg.set_image(image)
    if args.command == "point":
        points = [(x, y) for x, y, _ in args.points]
        labels = [label for _, _, label in args.points]
        pred = seg.predict(points=points, labels=labels, select=args.select)
        prompts = dict(points=points, labels=labels)
    else:
        pred = seg.predict(box=args.box)
        prompts = dict(box=args.box)

    mask = pred.mask
    if args.clean == "largest":
        mask = postprocess.clean_mask(mask, min_area=args.min_area, keep="largest")
    if not mask.any():
        log.warning("The resulting mask is empty (SAM score %.3f). Try different prompts.", pred.score)

    saved = [
        io.save_mask(mask, io.output_path(args.image, args.command, "mask", args.out_dir)),
        io.save_image(render.draw_prompts(render.overlay(seg.image, mask), **prompts),
                      io.output_path(args.image, args.command, "overlay", args.out_dir)),
    ]
    if args.cutout:
        rgba = postprocess.cutout(seg.image, mask, crop=args.crop, feather=args.feather)
        saved.append(io.save_image(rgba, io.output_path(args.image, args.command, "cutout", args.out_dir)))

    log.info("score %.3f | encode %.2fs", pred.score, seg.last_encode_seconds)
    for path in saved:
        log.info("wrote %s", path)
    return 0


def _run_auto(args) -> int:
    from .model import Segmenter

    image = io.load_image(args.image)
    seg = Segmenter(args.checkpoint, device=args.device)
    seg.set_image(image)
    segments = seg.segment_everything(
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        min_area=args.min_area,
    )
    if not segments:
        log.error("No masks passed the quality filters; try lowering --pred-iou-thresh / --stability-thresh.")
        return 1

    masks = [s.mask for s in segments]
    saved = [
        io.save_image(render.overlay_instances(seg.image, masks), io.output_path(args.image, "auto", "overlay", args.out_dir)),
        io.save_image(postprocess.instance_labels(masks), io.output_path(args.image, "auto", "labels", args.out_dir)),
    ]
    log.info("%d masks kept", len(segments))
    for path in saved:
        log.info("wrote %s", path)
    return 0


def _run_eval(args) -> int:
    from .model import Segmenter

    pairs = evaluation.match_pairs(args.images, args.masks)
    report = evaluation.evaluate(
        Segmenter(args.checkpoint, device=args.device), pairs, prompt=args.prompt, select=args.select
    )
    print(report.to_markdown() if args.per_image else report.to_summary())
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
    return 0


def _run_demo(args) -> int:
    try:
        from .demo.engine import DemoEngine
        from .demo.server import serve
    except ImportError as exc:
        log.error('The demo needs extra packages (%s). Install them with: pip install -e ".[demo]"', exc)
        return 2
    from .model import Segmenter

    segmenter = Segmenter(args.checkpoint, device=args.device, fp16=args.fp16)
    serve(DemoEngine(segmenter, max_side=args.max_side), args.host, args.port, args.open)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s", stream=sys.stderr)

    try:
        if args.command == "download-weights":
            download_weights()
            return 0
        if args.command in ("point", "box"):
            return _run_prompted(args)
        if args.command == "auto":
            return _run_auto(args)
        if args.command == "demo":
            return _run_demo(args)
        return _run_eval(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("error: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
