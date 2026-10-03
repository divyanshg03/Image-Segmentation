import numpy as np
import pytest

from imgseg.prompts import validate_box, validate_points

SIZE = (100, 200)  # height, width


def test_points_valid():
    pts, lab = validate_points([[10, 20], [30.5, 40]], [1, 0], SIZE)
    assert pts.shape == (2, 2) and pts.dtype == np.float32
    assert lab.tolist() == [1, 0]


def test_points_none_passthrough():
    assert validate_points(None, None, SIZE) == (None, None)


@pytest.mark.parametrize(
    "points, labels, message",
    [
        ([[10, 20]], None, "labels are required"),
        (None, [1], "without points"),
        ([[10, 20]], [1, 1], "match points"),
        ([[10, 20]], [2], "0 .* or 1"),
        ([10, 20], [1], "shape"),
        ([[10, 20, 30]], [1], "shape"),
        ([], [], "shape"),
        ([[float("nan"), 1]], [1], "finite"),
        ([[200, 20]], [1], "outside"),  # x == width is out of bounds
        ([[10, -1]], [1], "outside"),
    ],
)
def test_points_invalid(points, labels, message):
    with pytest.raises(ValueError, match=message):
        validate_points(points, labels, SIZE)


def test_box_valid_and_clipped():
    assert validate_box([10, 10, 50, 60], SIZE).tolist() == [10, 10, 50, 60]
    assert validate_box([-5, -5, 500, 500], SIZE).tolist() == [0, 0, 200, 100]
    assert validate_box([[10, 10, 50, 60]], SIZE).shape == (4,)  # (1, 4) accepted
    assert validate_box(None, SIZE) is None


@pytest.mark.parametrize(
    "box, message",
    [
        ([10, 10, 5, 60], "x1 < x2"),
        ([10, 60, 50, 60], "y1 < y2"),
        ([1, 2, 3], "got 3 values"),
        ([0, 0, float("inf"), 5], "finite"),
        ([300, 10, 400, 50], "outside"),
    ],
)
def test_box_invalid(box, message):
    with pytest.raises(ValueError, match=message):
        validate_box(box, SIZE)
