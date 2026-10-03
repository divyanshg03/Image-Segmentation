"""Accuracy benchmark on GeoStar-151, a standard interactive-segmentation set.

    python scripts/geostar151.py                  # download (22 MB), then score box and point prompts
    python scripts/geostar151.py --prompt box     # or: point
    python scripts/geostar151.py --json results.json

The set (Gulshan et al., "Geodesic Star Convexity for Interactive Image Segmentation", CVPR 2010;
hosted by Oxford VGG at https://www.robots.ox.ac.uk/~vgg/data/iseg/) has 151 images with
ground truth: 49 from the GrabCut database, 99 from PASCAL VOC 2009 and 3 alpha-matting images.
The page states no license, and some images come from Flickr via PASCAL VOC, so the data is
downloaded on demand into the git-ignored data/ folder and never redistributed here. Check the
original sources' terms before using it for anything beyond local evaluation.

Protocol: derive ONE prompt from each ground-truth mask (its tight box, or its most interior
pixel as a single click), segment, and score IoU / Dice. A single click is ambiguous (SAM returns
three candidate masks), so the click is scored twice: with SAM's own top-scoring candidate and with
the largest candidate. The "best of candidates" column is an oracle upper bound, not an
achievable score. Boundary pixels labelled "unknown" (grey 128) in the ground truth are excluded
from scoring. This is a single-prompt score, not the multi-click NoC metrics reported in
interactive-segmentation papers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from imgseg.evaluate import EvalReport, evaluate, match_pairs  # noqa: E402

BASE_URL = "https://www.robots.ox.ac.uk/~vgg/data/iseg/data/"
ARCHIVES = {"images.tgz": "images", "images-gt.tgz": "masks"}  # archive -> folder under DATA_DIR
DOWNLOAD_DIR = ROOT / "data" / "_downloads"
DATA_DIR = ROOT / "data" / "geostar151"
EXPECTED = {"grabcut": 49, "voc": 99, "matting": 3}


def subset_of(stem: str) -> str:
    """Which source a file came from, judging by its name."""
    if re.search(r"\d{4}_\d{6}$", stem):
        return "voc"
    if re.fullmatch(r"GT\d+", stem):
        return "matting"
    return "grabcut"


def prepare() -> None:
    """Download and unpack the data into data/geostar151/{images,masks} (idempotent)."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for archive, folder in ARCHIVES.items():
        target = DATA_DIR / folder
        if target.is_dir() and any(target.iterdir()):
            continue
        path = DOWNLOAD_DIR / archive
        if not path.is_file():
            print(f"downloading {BASE_URL + archive}", file=sys.stderr)
            urllib.request.urlretrieve(BASE_URL + archive, path)
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                if member.isfile():  # flat names only: nothing in the archive can write elsewhere
                    (target / Path(member.name).name).write_bytes(tar.extractfile(member).read())
        print(f"unpacked {archive} -> {target.relative_to(ROOT)}", file=sys.stderr)


def by_subset(report: EvalReport) -> dict[str, EvalReport]:
    groups = {name: EvalReport(report.prompt, select=report.select) for name in EXPECTED}
    for row in report.rows:
        groups[subset_of(Path(row.name).stem)].rows.append(row)
    return groups


def markdown(reports: dict[str, EvalReport]) -> str:
    lines = ["| prompt | subset | images | mean IoU | median IoU | mean Dice | IoU >= 0.9 | IoU < 0.5 | best of candidates |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, report in reports.items():
        rows = [("all", report)] + [(n, g) for n, g in by_subset(report).items() if g.rows]
        for name, r in rows:
            lines.append(f"| {label} | {name} | {len(r.rows)} | {r.mean_iou:.3f} | {r.median_iou:.3f} | "
                         f"{r.mean_dice:.3f} | {r.share(at_least=0.9):.0%} | {r.share(below=0.5):.0%} | "
                         f"{r.mean_oracle_iou:.3f} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", choices=("box", "point", "both"), default="both")
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoint")
    p.add_argument("--limit", type=int, help="score only the first N images (quick check)")
    p.add_argument("--prepare-only", action="store_true", help="download and unpack, then stop")
    p.add_argument("--json", metavar="PATH", help="write per-image results here")
    args = p.parse_args()

    prepare()
    if args.prepare_only:
        return

    from imgseg import Segmenter

    pairs = match_pairs(DATA_DIR / "images", DATA_DIR / "masks")
    counts = {n: sum(subset_of(i.stem) == n for i, _ in pairs) for n in EXPECTED}
    if counts != EXPECTED:
        print(f"warning: subset sizes {counts} differ from the dataset page's {EXPECTED}", file=sys.stderr)
    if args.limit:
        pairs = pairs[: args.limit]

    segmenter = Segmenter(args.checkpoint, device=args.device)
    configs = {"box": ("box", "score"), "point (SAM's pick)": ("point", "score"), "point (largest)": ("point", "largest")}
    if args.prompt != "both":
        configs = {k: v for k, v in configs.items() if v[0] == args.prompt}
    reports = {}
    for label, (prompt, select) in configs.items():
        print(f"scoring {len(pairs)} images: {label} ...", file=sys.stderr)
        reports[label] = evaluate(segmenter, pairs, prompt, select)
        print(reports[label].to_summary(), file=sys.stderr)

    print(f"\nGeoStar-151, {len(pairs)} images, SAM ViT-B on {segmenter.device}\n")
    print(markdown(reports))
    if args.json:
        Path(args.json).write_text(json.dumps({k: r.to_dict() for k, r in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
