import numpy as np
import pytest
from PIL import Image

from imgseg.io import load_image, load_mask, output_path, save_mask


def test_load_image_returns_rgb_uint8(tmp_path):
    Image.new("L", (8, 6), 100).save(tmp_path / "gray.png")
    arr = load_image(tmp_path / "gray.png")
    assert arr.shape == (6, 8, 3) and arr.dtype == np.uint8


def test_load_image_applies_exif_orientation(tmp_path):
    # Orientation 6 = "rotate 90 degrees clockwise to display". PIL ignores it,
    # OpenCV honours it; load_image must give the displayed (rotated) shape.
    im = Image.new("RGB", (40, 20), "red")
    exif = Image.Exif()
    exif[274] = 6
    im.save(tmp_path / "rotated.jpg", exif=exif)
    assert Image.open(tmp_path / "rotated.jpg").size == (40, 20)
    assert load_image(tmp_path / "rotated.jpg").shape[:2] == (40, 20)  # H, W swapped


def test_load_image_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_image(tmp_path / "nope.jpg")


def test_output_paths_differ_by_mode_and_kind():
    paths = {
        output_path("a/photo (1).jpg", mode, kind)
        for mode in ("point", "box", "auto")
        for kind in ("mask", "overlay")
    }
    assert len(paths) == 6  # point/box/auto runs on one image must not collide
    assert output_path("a/photo (1).jpg", "box", "mask").name == "photo (1)_box_mask.png"


def test_mask_roundtrip(tmp_path):
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:5, 3:9] = True
    save_mask(mask, tmp_path / "sub" / "m.png")  # creates the directory
    assert np.array_equal(load_mask(tmp_path / "sub" / "m.png"), mask)
