import numpy as np
import pytest

from imgseg.postprocess import as_bool_mask, clean_mask, cutout, instance_labels, mask_bbox
from imgseg.render import draw_prompts, instance_colors, overlay, overlay_instances


@pytest.fixture
def blobs():
    """A 100x100 mask: big 30x30 blob, medium 10x10 blob, 2x2 speck."""
    m = np.zeros((100, 100), dtype=bool)
    m[10:40, 10:40] = True  # 900 px
    m[60:70, 60:70] = True  # 100 px
    m[90:92, 90:92] = True  # 4 px
    return m


def test_clean_mask_min_area_is_honoured(blobs):
    # Regression: min_area used to be accepted but ignored.
    out = clean_mask(blobs, min_area=50)
    assert out.sum() == 1000  # speck gone, both real blobs kept
    assert clean_mask(blobs, min_area=500).sum() == 900
    assert clean_mask(blobs, min_area=0).sum() == blobs.sum()


def test_clean_mask_keep_all_preserves_multiple_objects(blobs):
    out = clean_mask(blobs, min_area=50, keep="all")
    assert out[20, 20] and out[65, 65]


def test_clean_mask_keep_largest(blobs):
    out = clean_mask(blobs, min_area=0, keep="largest")
    assert out.sum() == 900 and out[20, 20] and not out[65, 65]


def test_clean_mask_empty_when_nothing_qualifies(blobs):
    assert not clean_mask(blobs, min_area=10_000).any()
    assert not clean_mask(blobs, min_area=10_000, keep="largest").any()
    assert not clean_mask(np.zeros((5, 5), dtype=bool)).any()


def test_clean_mask_accepts_uint8_and_returns_bool(blobs):
    out = clean_mask(blobs.astype(np.uint8) * 255, min_area=50)
    assert out.dtype == bool and out.sum() == 1000


def test_clean_mask_bad_keep(blobs):
    with pytest.raises(ValueError, match="keep"):
        clean_mask(blobs, keep="biggest")


def test_as_bool_mask_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        as_bool_mask(np.zeros((4, 4, 3)))


def test_mask_bbox(blobs):
    assert mask_bbox(blobs) == (10, 10, 92, 92)
    assert mask_bbox(np.zeros((5, 5), dtype=bool)) is None


def test_cutout_alpha_matches_mask(blobs):
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    rgba = cutout(image, blobs)
    assert rgba.shape == (100, 100, 4)
    assert (rgba[..., 3] == blobs * 255).all()
    assert (rgba[..., :3] == 200).all()


def test_cutout_crop_and_feather(blobs):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    only_big = clean_mask(blobs, min_area=500)
    assert cutout(image, only_big, crop=True).shape == (30, 30, 4)
    soft = cutout(image, only_big, crop=True, feather=2.0)
    assert soft.shape[0] > 30  # padded so the soft edge is not clipped
    assert 0 < soft[..., 3][soft.shape[0] // 2, 2] < 255 or soft[..., 3].min() < 255


def test_cutout_errors(blobs):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="same height and width"):
        cutout(image[:50], blobs)
    with pytest.raises(ValueError, match="empty"):
        cutout(image, np.zeros((100, 100), dtype=bool), crop=True)


def test_instance_labels_later_masks_win():
    a = np.zeros((6, 6), dtype=bool)
    a[:4, :4] = True
    b = np.zeros((6, 6), dtype=bool)
    b[2:, 2:] = True
    labels = instance_labels([a, b])
    assert labels.dtype == np.uint16
    assert labels[0, 0] == 1 and labels[5, 5] == 2 and labels[3, 3] == 2 and labels[0, 5] == 0
    with pytest.raises(ValueError):
        instance_labels([])


def test_overlay_only_changes_masked_pixels(blobs):
    image = np.full((100, 100, 3), 100, dtype=np.uint8)
    out = overlay(image, blobs, color=(255, 0, 0), alpha=0.5, outline=False)
    assert out[20, 20].tolist() == [177, 50, 50]  # 0.5*100 + 0.5*color
    assert (out[50, 50] == 100).all() and (out[0, 0] == 100).all()
    assert (image == 100).all()  # input untouched


def test_overlay_instances_uses_distinct_colors():
    colors = instance_colors(12)
    assert len(set(colors)) == 12
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    a, b = np.zeros((20, 20), dtype=bool), np.zeros((20, 20), dtype=bool)
    a[:10], b[10:] = True, True
    out = overlay_instances(image, [a, b], alpha=1.0, outline=False)
    assert not np.array_equal(out[2, 2], out[15, 15])


def test_draw_prompts_marks_pixels_without_mutating():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    out = draw_prompts(image, points=[(20, 20), (80, 80)], labels=[1, 0], box=(10, 10, 90, 90))
    assert out.any() and not image.any()
    assert out[20, 20].tolist() == [0, 200, 0]  # foreground = green
    assert out[80, 80].tolist() == [230, 0, 0]  # background = red
