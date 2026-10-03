from dataclasses import dataclass

import numpy as np
import pytest
from PIL import Image

from imgseg.evaluate import box_from_mask, evaluate, interior_point, match_pairs
from imgseg.metrics import dice, iou


def test_iou_and_dice_values():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[:, :6] = True  # 60 px
    b[:, 3:9] = True  # 60 px, overlap 30 px
    assert iou(a, b) == pytest.approx(30 / 90)
    assert dice(a, b) == pytest.approx(2 * 30 / 120)
    assert iou(a, a) == dice(a, a) == 1.0


def test_metrics_edge_cases():
    empty = np.zeros((4, 4), dtype=bool)
    full = np.ones((4, 4), dtype=bool)
    assert iou(empty, empty) == dice(empty, empty) == 1.0
    assert iou(empty, full) == dice(empty, full) == 0.0
    assert iou(full.astype(np.uint8) * 255, full) == 1.0  # uint8 masks accepted
    with pytest.raises(ValueError, match="shapes differ"):
        iou(empty, np.zeros((5, 5), dtype=bool))


def test_box_from_mask():
    m = np.zeros((20, 30), dtype=bool)
    m[5:10, 7:20] = True
    assert box_from_mask(m) == (7, 5, 20, 10)
    with pytest.raises(ValueError, match="empty"):
        box_from_mask(np.zeros((3, 3), dtype=bool))


def test_interior_point_lies_on_the_object_even_for_a_ring():
    ring = np.zeros((101, 101), dtype=bool)
    yy, xx = np.mgrid[:101, :101]
    r = np.hypot(yy - 50, xx - 50)
    ring[(r > 30) & (r < 40)] = True
    cx, cy = (xx[ring].mean(), yy[ring].mean())
    assert not ring[int(round(cy)), int(round(cx))]  # the centroid is in the hole...
    x, y = interior_point(ring)
    assert ring[y, x]  # ...but the interior point is on the ring


def _write_pair(tmp_path, stem, size=(40, 30), ext=".jpg"):
    (tmp_path / "images").mkdir(exist_ok=True)
    (tmp_path / "masks").mkdir(exist_ok=True)
    Image.new("RGB", size, "gray").save(tmp_path / "images" / f"{stem}{ext}")
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    mask[5:20, 8:30] = 255
    Image.fromarray(mask).save(tmp_path / "masks" / f"{stem}.png")


def test_match_pairs(tmp_path):
    _write_pair(tmp_path, "b")
    _write_pair(tmp_path, "a", ext=".jpeg")
    Image.new("RGB", (4, 4)).save(tmp_path / "images" / "orphan.png")  # no mask -> skipped
    pairs = match_pairs(tmp_path / "images", tmp_path / "masks")
    assert [p[0].stem for p in pairs] == ["a", "b"]
    with pytest.raises(FileNotFoundError):
        match_pairs(tmp_path / "missing", tmp_path / "masks")


def test_match_pairs_none_found(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    with pytest.raises(ValueError, match="No image/mask pairs"):
        match_pairs(tmp_path / "images", tmp_path / "masks")


@dataclass
class _Pred:
    mask: np.ndarray


class _OracleSegmenter:
    """Stand-in for Segmenter: returns the box region / a disc, so scores are predictable."""

    def set_image(self, image):
        self.shape = image.shape[:2]

    def predict(self, *, points=None, labels=None, box=None, select="score"):
        mask = np.zeros(self.shape, dtype=bool)
        if box is not None:
            x1, y1, x2, y2 = (int(v) for v in box)
            mask[y1:y2, x1:x2] = True
        else:
            (x, y), = points
            mask[y, x] = True
        return _Pred(mask)


def test_evaluate_box_prompt_scores_perfect_for_rectangular_gt(tmp_path):
    _write_pair(tmp_path, "a")
    report = evaluate(_OracleSegmenter(), match_pairs(tmp_path / "images", tmp_path / "masks"), "box")
    assert report.mean_iou == pytest.approx(1.0) and report.mean_dice == pytest.approx(1.0)
    assert "mean (1)" in report.to_markdown()


def test_evaluate_point_prompt_and_validation(tmp_path):
    _write_pair(tmp_path, "a")
    pairs = match_pairs(tmp_path / "images", tmp_path / "masks")
    report = evaluate(_OracleSegmenter(), pairs, "point")
    assert 0 < report.mean_iou < 0.01  # a single pixel vs a 15x22 rectangle
    with pytest.raises(ValueError, match="prompt must be"):
        evaluate(_OracleSegmenter(), pairs, "scribble")


def test_evaluate_rejects_size_mismatch_and_skips_empty_gt(tmp_path):
    _write_pair(tmp_path, "a")
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(tmp_path / "masks" / "a.png")  # wrong size
    with pytest.raises(ValueError, match="mask is"):
        evaluate(_OracleSegmenter(), match_pairs(tmp_path / "images", tmp_path / "masks"))

    Image.fromarray(np.zeros((30, 40), dtype=np.uint8)).save(tmp_path / "masks" / "a.png")  # empty GT
    report = evaluate(_OracleSegmenter(), match_pairs(tmp_path / "images", tmp_path / "masks"))
    assert report.rows == [] and np.isnan(report.mean_iou)


# --- unlabelled ("ignore") boundary pixels -------------------------------------------------


def test_metrics_exclude_ignored_pixels():
    gt = np.zeros((10, 10), dtype=bool)
    gt[2:8, 2:8] = True
    pred = np.zeros((10, 10), dtype=bool)
    pred[1:9, 1:9] = True  # spills one pixel into the boundary band on every side
    band = np.zeros((10, 10), dtype=bool)
    band[1:9, 1:9] = True
    band &= ~gt  # the ring between gt and pred is unlabelled

    assert iou(pred, gt) < 1.0 and dice(pred, gt) < 1.0
    assert iou(pred, gt, ignore=band) == 1.0 and dice(pred, gt, ignore=band) == 1.0
    with pytest.raises(ValueError, match="ignore mask shape"):
        iou(pred, gt, ignore=np.zeros((3, 3), dtype=bool))


def test_load_ground_truth_conventions(tmp_path):
    from imgseg.io import load_ground_truth

    arr = np.array([[0, 128, 255], [10, 63, 64], [191, 192, 254]], dtype=np.uint8)
    Image.fromarray(arr).save(tmp_path / "gt.png")
    fg, ignore = load_ground_truth(tmp_path / "gt.png")
    assert fg.tolist() == [[False, False, True], [False, False, False], [False, True, True]]
    assert ignore.tolist() == [[False, True, False], [False, False, True], [True, False, False]]
    assert not (fg & ignore).any()
    with pytest.raises(FileNotFoundError):
        load_ground_truth(tmp_path / "missing.png")


def test_evaluate_ignores_the_unlabelled_band_around_the_object(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    Image.new("RGB", (40, 30), "gray").save(tmp_path / "images" / "a.png")
    gt = np.zeros((30, 40), dtype=np.uint8)
    gt[4:21, 7:31] = 128  # unlabelled band...
    gt[5:20, 8:30] = 255  # ...hugging the object
    Image.fromarray(gt).save(tmp_path / "masks" / "a.png")

    class SpillingSegmenter(_OracleSegmenter):
        """Returns the box grown by one pixel, i.e. exactly foreground + band."""

        def predict(self, *, points=None, labels=None, box=None, select="score"):
            x1, y1, x2, y2 = (int(v) for v in box)
            return super().predict(box=(x1 - 1, y1 - 1, x2 + 1, y2 + 1))

    pairs = match_pairs(tmp_path / "images", tmp_path / "masks")
    assert evaluate(SpillingSegmenter(), pairs, "box").mean_iou == pytest.approx(1.0)


def test_eval_report_summary_and_serialisation():
    import json

    from imgseg.evaluate import EvalReport, EvalRow

    report = EvalReport("box", [EvalRow("a", 0.95, 0.97), EvalRow("b", 0.60, 0.75), EvalRow("c", 0.20, 0.33)])
    assert report.median_iou == pytest.approx(0.60)
    assert report.share(at_least=0.9) == pytest.approx(1 / 3)
    assert report.share(below=0.5) == pytest.approx(1 / 3)
    assert [r.name for r in report.worst(2)] == ["c", "b"]
    summary = report.to_summary(worst=1)
    assert "images scored: 3" in summary and "c (0.20)" in summary
    assert json.loads(json.dumps(report.to_dict()))["rows"][0] == {
        "name": "a", "iou": 0.95, "dice": 0.97, "oracle_iou": 0.95,  # single candidate: oracle == iou
    }
    with pytest.raises(ValueError, match="exactly one"):
        report.share()
    assert "No images were scored" in EvalReport("box").to_summary()


# --- choosing among SAM's candidate masks ----------------------------------------------------


def test_select_candidate():
    from imgseg.postprocess import select_candidate

    small = np.zeros((4, 4), dtype=bool)
    small[0, 0] = True
    big = np.zeros((4, 4), dtype=bool)
    big[:3, :3] = True
    mid = np.zeros((4, 4), dtype=bool)
    mid[:2, :2] = True
    candidates, scores = np.stack([small, big, mid]), np.array([0.99, 0.5, 0.7])

    assert select_candidate(candidates, scores) == 0  # SAM's own confidence
    assert select_candidate(candidates, scores, "score") == 0
    assert select_candidate(candidates, scores, "largest") == 1
    with pytest.raises(ValueError, match="select must be"):
        select_candidate(candidates, scores, "smallest")


@dataclass
class _MultiPred:
    mask: np.ndarray
    candidates: np.ndarray
    scores: np.ndarray


class _AmbiguousSegmenter:
    """A confident-but-wrong small part plus the right whole object, like a single click on a shirt."""

    def set_image(self, image):
        self.shape = image.shape[:2]

    def predict(self, *, points=None, labels=None, box=None, select="score"):
        from imgseg.postprocess import select_candidate

        whole = np.zeros(self.shape, dtype=bool)
        whole[5:20, 8:30] = True  # matches the ground truth written by _write_pair
        part = np.zeros(self.shape, dtype=bool)
        part[5:8, 8:12] = True
        candidates, scores = np.stack([part, whole]), np.array([0.99, 0.60])
        best = select_candidate(candidates, scores, select)
        return _MultiPred(candidates[best], candidates, scores)


def test_evaluate_reports_the_oracle_upper_bound_and_honours_select(tmp_path):
    _write_pair(tmp_path, "a")
    pairs = match_pairs(tmp_path / "images", tmp_path / "masks")

    by_score = evaluate(_AmbiguousSegmenter(), pairs, "point", select="score")
    by_largest = evaluate(_AmbiguousSegmenter(), pairs, "point", select="largest")

    assert by_score.mean_iou < 0.1  # trusted the confident sub-part
    assert by_largest.mean_iou == pytest.approx(1.0)
    assert by_score.mean_oracle_iou == by_largest.mean_oracle_iou == pytest.approx(1.0)
    assert "select=largest" in by_largest.to_summary() and "upper bound" in by_score.to_summary()
    assert by_score.to_dict()["select"] == "score"


def test_single_candidate_rows_have_no_oracle_gap(tmp_path):
    _write_pair(tmp_path, "a")
    report = evaluate(_OracleSegmenter(), match_pairs(tmp_path / "images", tmp_path / "masks"), "box")
    assert report.rows[0].oracle_iou is None
    assert report.mean_oracle_iou == pytest.approx(report.mean_iou)
