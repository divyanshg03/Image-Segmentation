import argparse

import pytest

from imgseg.cli import build_parser, main, parse_point

from .conftest import SAMPLES


def test_parse_point():
    assert parse_point("10,20") == (10.0, 20.0, 1)
    assert parse_point("10.5,20,0") == (10.5, 20.0, 0)
    for bad in ("10", "a,b", "1,2,3", "1,2,0,9", ""):
        with pytest.raises(argparse.ArgumentTypeError, match="invalid point"):
            parse_point(bad)


def test_points_accumulate_into_one_prompt():
    args = build_parser().parse_args(["point", "x.jpg", "--point", "1,2", "--point", "3,4,0"])
    assert args.points == [(1.0, 2.0, 1), (3.0, 4.0, 0)]


def test_select_option():
    parse = build_parser().parse_args
    assert parse(["point", "x.jpg", "--point", "1,2"]).select == "score"
    assert parse(["point", "x.jpg", "--point", "1,2", "--select", "largest"]).select == "largest"
    assert parse(["eval", "--images", "a", "--masks", "b", "--select", "largest"]).select == "largest"
    with pytest.raises(SystemExit):
        parse(["point", "x.jpg", "--point", "1,2", "--select", "smallest"])


def test_parser_requires_a_prompt():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["point", "x.jpg"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["box", "x.jpg", "--box", "1", "2", "3"])


def test_missing_image_fails_fast_with_exit_code_2(tmp_path, caplog):
    # No weights needed: the image is checked before the model is loaded.
    with caplog.at_level("ERROR", logger="imgseg"):
        code = main(["box", str(tmp_path / "nope.jpg"), "--box", "0", "0", "5", "5"])
    assert code == 2
    assert any("not found" in r.message for r in caplog.records)


@pytest.mark.slow
def test_cli_runs_do_not_overwrite_each_other(tmp_path):
    image = str(SAMPLES / "football_dribble.jpeg")
    out = str(tmp_path)
    assert main(["box", image, "--box", "250", "120", "420", "310", "--out-dir", out, "--cutout", "--crop"]) == 0
    assert main(["point", image, "--point", "352,205", "--point", "300,285", "--out-dir", out]) == 0
    assert main(["auto", image, "--out-dir", out]) == 0

    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "football_dribble_auto_labels.png",
        "football_dribble_auto_overlay.png",
        "football_dribble_box_cutout.png",
        "football_dribble_box_mask.png",
        "football_dribble_box_overlay.png",
        "football_dribble_point_mask.png",
        "football_dribble_point_overlay.png",
    ]


@pytest.mark.slow
def test_cli_empty_result_is_reported_not_silent(tmp_path, caplog):
    # A 2x2 pixel box in the grass: whatever SAM returns, a --clean largest with a high
    # min-area must not crash and an empty result must be warned about.
    image = str(SAMPLES / "football_dribble.jpeg")
    with caplog.at_level("WARNING", logger="imgseg"):
        code = main(["box", image, "--box", "20", "20", "22", "22", "--clean", "largest",
                     "--min-area", "10000000", "--out-dir", str(tmp_path)])
    assert code == 0
    assert any("empty" in r.message for r in caplog.records)


def test_demo_without_the_extra_installed_gives_an_install_hint(monkeypatch, caplog):
    import sys

    monkeypatch.setitem(sys.modules, "fastapi", None)  # makes `import fastapi` raise ImportError
    monkeypatch.delitem(sys.modules, "imgseg.demo.server", raising=False)  # force a fresh import
    with caplog.at_level("ERROR", logger="imgseg"):
        code = main(["demo"])
    assert code == 2
    assert any('pip install -e ".[demo]"' in r.message for r in caplog.records)
