"""Tests that run the real SAM model (skipped when no weights are present)."""

import numpy as np
import pytest

from imgseg import Prediction, Segmenter
from imgseg.evaluate import interior_point
from imgseg.metrics import iou
from imgseg.model import resolve_device
from imgseg.postprocess import mask_bbox

# Pixel coordinates in samples/football_dribble.jpeg (735x471): Messi is the player
# in the yellow kit near the centre; (404, 235) is open grass just right of him.
MESSI_POINTS = [(352, 205), (300, 285), (395, 262)]
MESSI_BOX = (250, 120, 420, 310)
GRASS = (404, 235)

slow = pytest.mark.slow


def test_fp16_on_cpu_is_rejected_before_loading_anything():
    with pytest.raises(ValueError, match="CUDA"):
        Segmenter(device="cpu", fp16=True)


def test_unknown_model_type_is_rejected():
    with pytest.raises(ValueError, match="model_type"):
        Segmenter(model_type="vit_x")


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda")


@slow
def test_predict_requires_an_image_first():
    with pytest.raises(RuntimeError, match="set_image"):
        Segmenter(device="cpu").predict(box=(0, 0, 5, 5))


@slow
def test_set_image_rejects_wrong_arrays(segmenter):
    with pytest.raises(ValueError, match="uint8 RGB"):
        segmenter.set_image(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="uint8 RGB"):
        segmenter.set_image(np.zeros((10, 10, 3), dtype=np.float32))


@slow
def test_all_points_form_one_prompt(football):
    """Regression: multi-point input used to become N independent prompts (N masks,
    unioned), so a negative point produced its own foreground mask instead of refining."""
    pred = football.predict(points=MESSI_POINTS, labels=[1, 1, 1])
    assert isinstance(pred, Prediction)
    assert pred.candidates.shape[0] == 1  # one prompt -> one mask
    x1, y1, x2, y2 = mask_bbox(pred.mask)
    assert 250 <= x1 and x2 <= 430 and 120 <= y1 and y2 <= 320  # it is Messi, not the pitch
    assert 5_000 < pred.mask.sum() < 30_000


@slow
def test_negative_point_is_excluded_from_the_mask(football):
    base = football.predict(points=MESSI_POINTS, labels=[1, 1, 1]).mask
    x, y = interior_point(base)
    refined = football.predict(points=MESSI_POINTS + [(x, y)], labels=[1, 1, 1, 0])
    assert refined.candidates.shape[0] == 1
    assert not refined.mask[y, x]  # the background click is respected


@slow
def test_single_point_is_ambiguous_and_returns_three_candidates(football):
    pred = football.predict(points=[(352, 205)], labels=[1])
    assert pred.candidates.shape[0] == 3 and pred.scores.shape == (3,)
    assert pred.score == pytest.approx(pred.scores.max())


@slow
def test_select_largest_returns_the_biggest_candidate(football):
    """A click on the shirt: SAM's top-scoring candidate is a small part, 'largest' is the widest."""
    by_score = football.predict(points=[(352, 205)], labels=[1])
    by_largest = football.predict(points=[(352, 205)], labels=[1], select="largest")
    assert by_score.score == pytest.approx(by_score.scores.max())
    assert by_largest.mask.sum() == by_largest.candidates.reshape(3, -1).sum(axis=1).max()
    assert by_largest.mask.sum() > by_score.mask.sum()
    with pytest.raises(ValueError, match="select must be"):
        football.predict(points=[(352, 205)], labels=[1], select="smallest")


@slow
def test_the_old_demo_click_selects_the_pitch_not_a_player(football):
    """The notebook's original point prompt (55% width, 50% height) sat on grass."""
    pred = football.predict(points=[GRASS], labels=[1])
    assert pred.mask.mean() > 0.5


@slow
def test_box_and_points_agree_on_the_same_object(football):
    by_box = football.predict(box=MESSI_BOX).mask
    by_points = football.predict(points=MESSI_POINTS, labels=[1, 1, 1]).mask
    assert iou(by_box, by_points) > 0.9


@slow
def test_embedding_is_cached_across_prompts(segmenter, monkeypatch):
    calls = []
    original = segmenter._predictor.set_image
    monkeypatch.setattr(segmenter._predictor, "set_image", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    from .conftest import SAMPLES

    segmenter.set_image(SAMPLES / "football_dribble.jpeg")
    for _ in range(3):
        segmenter.predict(box=MESSI_BOX)
    segmenter.predict(points=MESSI_POINTS, labels=[1, 1, 1])
    assert len(calls) == 1  # one encode served four prompts
    assert segmenter.last_encode_seconds > 0


@slow
def test_prompts_are_validated_against_the_loaded_image(football):
    with pytest.raises(ValueError, match="outside"):
        football.predict(points=[(5000, 5)], labels=[1])
    with pytest.raises(ValueError, match="Provide points and/or a box"):
        football.predict()


@slow
def test_segment_everything_keeps_masks_separate_and_sorted(football):
    segments = football.segment_everything()
    assert len(segments) > 1
    h, w = football.image.shape[:2]
    assert all(s.mask.shape == (h, w) and s.mask.dtype == bool for s in segments)
    areas = [s.area for s in segments]
    assert areas == sorted(areas, reverse=True)
    assert all(s.score >= 0.88 and s.stability >= 0.95 for s in segments)  # quality filters applied
    assert sum(s.mask.sum() for s in segments) > football.image.shape[0]  # sanity: not a single merged mask
