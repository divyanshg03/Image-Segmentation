from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def _has_weights() -> bool:
    from imgseg.weights import resolve_checkpoint

    try:
        resolve_checkpoint()
    except FileNotFoundError:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    if _has_weights():
        return
    skip = pytest.mark.skip(reason="SAM weights not found (see README, 'Get the weights')")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def segmenter():
    from imgseg import Segmenter

    return Segmenter()


@pytest.fixture
def football(segmenter):
    return segmenter.set_image(SAMPLES / "football_dribble.jpeg")
