# Image Segmentation with SAM

Promptable, zero-shot image segmentation: click on an object, draw a box around it, or let the model
propose everything it can find. It uses Meta's **Segment Anything Model (SAM, ViT-B)**, so there is **no
training** in this repo. It wraps the pretrained model in a small, tested Python package, a command line
tool, a browser demo and a demo notebook.

* a **browser demo** with live hover preview, click / box prompts, candidate choice and PNG export (`imgseg demo`)
* `imgseg.Segmenter` encodes an image once and answers any number of point or box prompts from the cached embedding
* post-processing: mask clean-up, RGBA cut-outs, per-instance overlays
* an IoU / Dice evaluation harness for when you have ground-truth masks
* a benchmark script (CUDA fp32 / fp16, CPU fp32 / INT8, MobileSAM)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .                  # provides `import imgseg` and the `imgseg` command
```

For GPU use, install the CUDA build of PyTorch from <https://pytorch.org> first. Without a GPU everything
still works on the CPU, just slowly (see [Performance](#performance)). Tested on Python 3.13,
torch 2.8, numpy 2.2, opencv 4.12, segment-anything 1.0.

### Get the weights

The ViT-B checkpoint (375 MB) is not stored in git. Any one of:

* `python -m imgseg download-weights` saves Meta's official `sam_vit_b_01ec64.pth` into `weights/`
* copy `sam_vit_b_01ec64.pth` or `sam_b.pt` into `weights/`
* set `IMGSEG_CHECKPOINT=/path/to/file`, or pass `--checkpoint` / `Segmenter(checkpoint=...)`

## Usage

### Browser demo

```bash
pip install -e ".[demo]"      # adds FastAPI + uvicorn (the core package does not need them)
imgseg demo --open            # http://127.0.0.1:8000   (--fp16 for ~1.5x faster on a GPU)
```

Open one of the sample images or drop / paste / browse your own, then:

| to | do |
|---|---|
| include / exclude a region | click / right-click (or Alt+click). On touch screens use the Include / Exclude tool |
| segment a whole object | drag a box (in any tool), or use the Box tool |
| see the mask before you click | **hover preview** is on by default on a GPU (`H` toggles it) |
| resolve an ambiguous click | pick one of the three candidate thumbnails (keys `1` `2` `3`), or *Largest* |
| refine | keep adding points, `Ctrl+Z` undoes, `R` resets |
| zoom and pan | scroll wheel, `+` `-` `0`; hold Space and drag |
| find every object | *Segment everything*, then click a segment to select it |
| keep the result | export a cut-out (cropped or full size), the mask, or an overlay as PNG |

How it works: uploading an image runs the heavy encoder once (about 0.6 s on the RTX 4060 used for the benchmarks) and the server
keeps that embedding, so every click or hover afterwards costs only the light mask decoder (about 40 to 50 ms there). That is
what makes hover preview feel instant, and why the page is a canvas talking to a small FastAPI server instead of a form-based
UI. Up to 8 images stay loaded at once and share one model, so several tabs work without re-encoding. The model is warmed up at
start-up so the first interaction is as fast as the rest.

Things to know:

* **Local use only.** It listens on `127.0.0.1` with no authentication; requests from other origins and unexpected `Host` headers
  are refused, and the page ships a strict Content-Security-Policy (no inline scripts, nothing loaded from other sites). If you
  pass `--host 0.0.0.0` anyone who can reach the port can use your GPU and upload files.
* On a **CPU** a decode takes a few hundred ms, so hover preview is off by default there.
* Requests are handled one at a time. *Segment everything* takes a few seconds and holds up other requests while it runs.
* Uploads are limited to 25 MB and downscaled so the longer side is at most 1600 px (`--max-side`); exports are at that size.
* Points are placed with a pointer (mouse or touch); there is no keyboard-only way to place them yet.

`python scripts/check_demo_ui.py` starts the real server, drives it in headless Chrome with real mouse and keyboard events
(hover, click, right-click, drag, undo, upload, export, phone width) and fails on any JavaScript error or CSP violation. It saves
screenshots to `outputs/ui_check/`. Needs `pip install websocket-client` and a Chrome / Chromium / Edge install.

### Command line

```bash
# One prompt made of several points: 1 = foreground (default), 0 = background
imgseg point samples/football_dribble.jpeg --point 352,205 --point 330,250 --point 395,262 --point 297,277,0

# A box: x1 y1 x2 y2
imgseg box samples/starry_night.jpg --box 222 70 296 312 --cutout --crop

# Everything, with every mask kept separate
imgseg auto samples/trophy_celebration.jpeg
```

Results go to `outputs/` (git-ignored) and are named by prompt mode, so runs never overwrite each other:
`<image>_<mode>_mask.png` (0/255), `_overlay.png`, `_cutout.png` (RGBA, with `--cutout`), and for `auto`
`_overlay.png` plus `_labels.png` (uint16, pixel value = instance id). Useful flags: `--device cpu`,
`--clean largest` (keep only the biggest region), `--select largest` (for a lone click: take the widest
candidate instead of SAM's top-scoring one), `--feather`, `--out-dir`.

### Python

```python
from imgseg import Segmenter
from imgseg.postprocess import clean_mask, cutout

seg = Segmenter()                                   # CUDA if available, else CPU
seg.set_image("samples/football_dribble.jpeg")      # runs the heavy encoder once

pred = seg.predict(points=[(352, 205), (330, 250)], labels=[1, 1])
pred.mask                                           # (H, W) bool

seg.predict(box=(250, 120, 420, 310))               # milliseconds: the embedding is reused
rgba = cutout(seg.image, clean_mask(pred.mask), crop=True)
segments = seg.segment_everything()                 # list[Segment], largest first
```

Coordinates are pixels `(x, y)` with the origin at the top-left. The notebook
[`notebooks/segmentation_demo.ipynb`](notebooks/segmentation_demo.ipynb) walks through all of this with pictures.

## Behaviour worth knowing

* **All points are one prompt.** Label 1 marks the object, label 0 marks what to exclude. (Ultralytics' API treats a flat
  list of points as separate prompts, which breaks this; the first version of this project hit that.)
* **A single click is ambiguous.** `predict` then returns SAM's three candidates in `.candidates` / `.scores`.
  `.mask` is the top-scoring one by default (`select="score"`); `select="largest"` takes the widest instead. A click on a
  shirt gives the shirt, not the player. Add points or use a box, and see [Evaluation](#evaluation) for how much the choice matters.
* **Segment-everything finds parts, not objects.** It returns separate, quality-filtered masks with no grouping and no labels.
  On the dense poster in `samples/` it covers about 15% of the pixels at default thresholds, and about 78% after relaxing the
  filters (at lower confidence). Prompted segmentation of the same poster works. See section 7 of the notebook.
* **Bad input fails loudly**: out-of-image points, malformed boxes, missing files and empty results raise errors or warnings
  instead of silently returning a blank mask. EXIF rotation is applied once at load so masks always line up with the image.

## Performance

`python scripts/benchmark.py --mobile-sam weights/mobile_sam.pt` on an RTX 4060 Laptop GPU (torch 2.8.0+cu129): same image
and box for every row, median after warm-up, CUDA synchronised.

| config | encode ms | prompt ms | encode + prompt ms | peak GPU MB | IoU vs CUDA fp32 |
|---|---:|---:|---:|---:|---:|
| CUDA fp32 | 586 | 38 | 624 | 2770 | 1.000 |
| CUDA fp16 (autocast) | 358 | 53 | 412 | 1761 | 0.999 |
| CUDA MobileSAM (Ultralytics) | n/a | n/a | 109 | 290 | 0.960 |
| CPU fp32 | 13709 | 193 | 13905 | - | 1.000 |
| CPU int8 encoder (dynamic) | 12800 | 194 | 13006 | - | 0.992 |

* The encoder is almost all the cost, which is why `set_image` caches it: extra prompts on the same image are about 40 ms.
* fp16 autocast is 1.5x faster with 36% less memory and almost identical masks (`Segmenter(fp16=True)`).
* Dynamic INT8 on CPU is only about 6% faster, so a GPU matters far more than quantisation.
* MobileSAM is the fastest here (5.7x) at a mask IoU of 0.960 against ViT-B. It is not integrated into `imgseg`; it is measured
  through Ultralytics, which is imported only by the benchmark and only when `--mobile-sam` is given (see [License](#license)).

These numbers are specific to one machine. IoU is against the fp32 mask, not ground truth.

## Evaluation

Accuracy on **GeoStar-151** (Gulshan et al., CVPR 2010; hosted by [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/iseg/)):
151 images with ground-truth masks, of which 49 come from the GrabCut database, 99 from PASCAL VOC 2009 and 3 are matting images.
SAM ViT-B, CUDA fp32, **one prompt per image** derived from the ground truth: its tight box, or a single click on the foreground
pixel farthest from the boundary. Unlabelled boundary pixels in the ground truth are excluded from scoring.

| prompt | mean IoU | median IoU | mean Dice | IoU >= 0.9 | IoU < 0.5 | best of SAM's 3 candidates |
|---|---:|---:|---:|---:|---:|---:|
| box | 0.915 | 0.952 | 0.951 | 75% | 3% | n/a (one mask) |
| single click, SAM's top-scoring candidate (default) | 0.549 | 0.545 | 0.628 | 28% | 47% | 0.854 |
| single click, largest candidate (`select="largest"`) | 0.794 | 0.895 | 0.862 | 49% | 12% | 0.854 |

Mean IoU by source: box 0.934 GrabCut / 0.906 VOC; click (SAM's pick) 0.641 / 0.517; click (largest) 0.838 / 0.779. The 3 matting
images are too few to read anything into.

What this says, and what it does not:

* **A tight box is very reliable** (mean IoU 0.915), but the box here is the *ground-truth* box. A hand-drawn box is looser, so
  treat this as an optimistic ceiling.
* **A single click is mostly a candidate-selection problem.** SAM returns three masks; the best of them averages 0.854, but the
  one SAM ranks highest averages only 0.549, because its confidence tends to favour small, confident sub-parts (a shirt, not the
  player). Taking the largest candidate recovers most of the gap (0.794).
* **"Largest" is not a free lunch.** This benchmark is mostly one salient, whole object per image, which is exactly where "largest"
  wins. It is the wrong rule when you click a part and want only that part, and it was chosen after seeing these numbers, so expect
  less on other data. That is why the default stays `select="score"`.
* **This is a single-prompt score, not the multi-click metric** (clicks added where the previous mask was wrong) that
  interactive-segmentation papers report, so the numbers are not comparable with published ones.

Reproduce it (downloads 22 MB into the git-ignored `data/` folder, then scores all 151 images in a few minutes on a GPU):

```bash
python scripts/geostar151.py                 # box + both click rules, with the per-source breakdown
python scripts/geostar151.py --prompt point --json results.json
```

The data is downloaded on demand and is not redistributed here. The source page states no license and some images come from
Flickr via PASCAL VOC, so check the original terms before using it beyond local evaluation.

### Your own data

```
data/mine/images/  cat.jpg  dog.png        # images
data/mine/masks/   cat.png  dog.png        # ground truth, same file stem
```

Mask values: 255 = foreground, 0 = background, mid-grey (e.g. 128) = unlabelled pixels to ignore.

```bash
imgseg eval --images data/mine/images --masks data/mine/masks --prompt box
imgseg eval --images data/mine/images --masks data/mine/masks --prompt point --select largest --per-image
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Pure-logic tests, including the demo's API tests (which use a fake model), need nothing else. Tests that run the real model are
marked `slow` and are skipped automatically when no weights are found. `scripts/check_demo_ui.py` is the browser-level check of
the demo (see above). Enable `pre-commit install` once so notebook outputs are stripped on commit.

## Layout

```
imgseg/        Segmenter, prompt validation, post-processing, rendering, metrics, evaluation, CLI, weights
imgseg/demo/   browser demo: engine.py (sessions, no web imports), server.py (FastAPI), static/ (HTML, CSS, JS)
notebooks/     segmentation_demo.ipynb (committed without outputs)
scripts/       benchmark.py (speed), geostar151.py (accuracy), check_demo_ui.py (browser test of the demo)
samples/       four test images (third-party, see samples/README.md)
tests/         pytest suite
weights/       model checkpoints (git-ignored)
outputs/       generated results (git-ignored)
data/          downloaded evaluation data (git-ignored)
```

## Roadmap

* Demo: several objects per image (one mask each, exported together), and keyboard-only point placement
* MobileSAM and SAM 2 as optional backends (MobileSAM would also make hover preview usable on a CPU)
* A multi-click evaluation (clicks added where the previous mask was wrong), the metric interactive-segmentation papers report

## License

This project is Apache-2.0 (see `LICENSE`). SAM's code and weights are Apache-2.0 (Meta AI). `imgseg` depends on Meta's
`segment-anything` package and deliberately **not** on Ultralytics, which is AGPL-3.0. The benchmark can optionally import
Ultralytics to time MobileSAM; that is a measurement convenience, not part of the package. The sample images are third-party
and are not covered by this license (see `samples/README.md`).
