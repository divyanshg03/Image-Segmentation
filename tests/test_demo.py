"""Tests for the demo UI backend. Most use a fake segmenter, so no weights or GPU are needed."""

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from imgseg.demo import server
from imgseg.demo.engine import DemoEngine, ImageRejected, decode_upload
from imgseg.demo.server import create_app
from imgseg.postprocess import select_candidate

from .conftest import SAMPLES


# --------------------------------------------------------------------------------- fakes


class FakeSegmenter:
    """Same surface as Segmenter. Masks are fixed rectangles so the tests can predict areas."""

    device = SimpleNamespace(type="cpu")
    device_name = "Fake CPU"
    fp16 = False

    def __init__(self):
        self.encodes = 0
        self.active = None

    def encode(self, image):
        self.encodes += 1
        return SimpleNamespace(image=np.asarray(image), encode_seconds=0.004)

    def activate(self, encoded):
        self.active = encoded

    def predict(self, *, points=None, labels=None, box=None, select="score"):
        h, w = self.active.image.shape[:2]
        for x, y in points or []:
            if not (0 <= x < w and 0 <= y < h):
                raise ValueError(f"point(s) outside the {w}x{h} image: [[{x}, {y}]]")
        if points is None and box is None:
            raise ValueError("Provide points and/or a box")

        def rect(y1, y2, x1, x2):
            m = np.zeros((h, w), dtype=bool)
            m[y1:y2, x1:x2] = True
            return m

        if points is not None and len(points) == 1 and box is None:
            candidates = np.stack([rect(10, 14, 10, 16), rect(10, 20, 10, 25), rect(10, 30, 10, 40)])
            scores = np.array([0.95, 0.80, 0.60])  # confident small part first, like real SAM
        else:
            candidates = rect(10, 30, 10, 40)[None]
            scores = np.array([0.90])
        index = select_candidate(candidates, scores, select)
        return SimpleNamespace(mask=candidates[index], score=float(scores[index]), candidates=candidates,
                               scores=scores, index=index)

    def segment_everything(self, *, points_per_side=32):
        h, w = self.active.image.shape[:2]
        big = np.zeros((h, w), dtype=bool)
        big[:, : w // 2] = True
        small = np.zeros((h, w), dtype=bool)
        small[5:9, 5:9] = True
        return [SimpleNamespace(mask=big, score=0.9, area=int(big.sum())),
                SimpleNamespace(mask=small, score=0.95, area=int(small.sum()))]


def png(size=(120, 80), color=(10, 120, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def decode_b64(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


@pytest.fixture
def fake():
    return FakeSegmenter()


@pytest.fixture
def client(fake):
    return TestClient(create_app(DemoEngine(fake, max_side=200, max_sessions=2, samples_dir=SAMPLES)))


def open_image(client, data=None, name="photo.png"):
    res = client.post("/api/image", files={"file": (name, data or png(), "image/png")})
    assert res.status_code == 200, res.text
    return res.json()


def test_warm_up_exercises_the_model_without_creating_a_session(fake):
    engine = DemoEngine(fake, samples_dir=SAMPLES)
    assert engine.warm_up() >= 0
    assert fake.encodes == 1 and fake.active is not None  # it did run the encoder and decoder...
    assert len(engine._sessions) == 0  # ...but left nothing behind


# ---------------------------------------------------------------------------------- pages


def test_index_and_static_files_are_served_with_a_strict_csp(client):
    page = client.get("/")
    assert page.status_code == 200 and "imgseg" in page.text
    csp = page.headers["content-security-policy"]
    assert "script-src 'self'" in csp and "'unsafe-inline'" not in csp and "frame-ancestors 'none'" in csp
    assert page.headers["x-content-type-options"] == "nosniff"
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_the_page_has_no_inline_script_or_style():
    from imgseg.demo.server import STATIC

    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "<script>" not in html and 'style="' not in html and "<style" not in html
    assert 'src="/static/app.js"' in html


def test_info_lists_samples(client):
    info = client.get("/api/info").json()
    assert info["device"] == "cpu" and info["device_name"] == "Fake CPU"
    assert "football_dribble.jpeg" in info["samples"]
    assert client.get("/api/info").headers["cache-control"] == "no-store"


# ---------------------------------------------------------------------------- security


def test_cross_origin_posts_are_refused_but_same_origin_and_plain_clients_work(client):
    upload = {"file": ("a.png", png(), "image/png")}
    evil = client.post("/api/image", files=upload, headers={"origin": "http://evil.example"})
    assert evil.status_code == 403 and "cross-origin" in evil.json()["error"]
    assert client.post("/api/sample/football_dribble.jpeg", headers={"origin": "http://evil.example"}).status_code == 403

    same = client.post("/api/image", files=upload, headers={"origin": "http://testserver"})  # what our own page sends
    assert same.status_code == 200
    assert client.post("/api/image", files=upload).status_code == 200  # curl, scripts: no Origin header
    assert client.get("/api/info", headers={"origin": "http://evil.example"}).status_code == 200  # reads are not state-changing


def test_host_allow_list_blocks_dns_rebinding(fake):
    engine = DemoEngine(fake, samples_dir=SAMPLES)
    app = create_app(engine, allowed_hosts=["127.0.0.1", "localhost"])
    assert TestClient(app, base_url="http://localhost:8000").get("/api/info").status_code == 200
    assert TestClient(app, base_url="http://127.0.0.1:8000").get("/api/info").status_code == 200
    assert TestClient(app, base_url="http://attacker.example:8000").get("/api/info").status_code == 400


# --------------------------------------------------------------------------- uploads


def test_upload_returns_a_session_and_serves_the_normalised_image(client, fake):
    meta = open_image(client)
    assert (meta["width"], meta["height"]) == (120, 80) and meta["name"] == "photo.png"
    assert meta["encode_ms"] == pytest.approx(4.0) and fake.encodes == 1
    image = client.get(f"/api/image/{meta['session']}")
    assert image.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(image.content)).size == (120, 80)


def test_large_uploads_are_downscaled_so_coordinates_stay_consistent(client):
    meta = open_image(client, png((400, 100)))  # engine max_side is 200
    assert (meta["width"], meta["height"]) == (200, 50)
    assert Image.open(io.BytesIO(client.get(f"/api/image/{meta['session']}").content)).size == (200, 50)


def test_exif_rotation_is_applied(client):
    exif = Image.Exif()
    exif[274] = 6  # displayed rotated by 90 degrees
    buf = io.BytesIO()
    Image.new("RGB", (60, 30), "red").save(buf, format="JPEG", exif=exif)
    meta = open_image(client, buf.getvalue(), "phone.jpg")
    assert (meta["width"], meta["height"]) == (30, 60)


@pytest.mark.parametrize("payload", [b"not an image at all", b"", b"\x89PNG\r\n\x1a\n" + b"\x00" * 20])
def test_unreadable_uploads_are_rejected_cleanly(client, payload):
    res = client.post("/api/image", files={"file": ("bad.png", payload, "image/png")})
    assert res.status_code == 400 and "could not read" in res.json()["error"]


def test_oversized_uploads_are_rejected(client, monkeypatch):
    data = png((300, 300))
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", len(data) - 1)  # one byte over the limit
    res = client.post("/api/image", files={"file": ("big.png", data, "image/png")})
    assert res.status_code == 400 and "larger than" in res.json()["error"]


def test_decompression_bombs_are_refused_before_decoding():
    from imgseg.demo import engine

    buf = io.BytesIO()
    Image.new("1", (10, 10)).save(buf, format="PNG")
    original = engine.MAX_DECODE_PIXELS
    engine.MAX_DECODE_PIXELS = 50
    try:
        with pytest.raises(ImageRejected, match="too large"):
            decode_upload(buf.getvalue(), 1600)
    finally:
        engine.MAX_DECODE_PIXELS = original


# ------------------------------------------------------------------------- segmenting


def test_single_click_returns_candidates_and_honours_select_and_choose(client):
    sid = open_image(client)["session"]
    base = {"session": sid, "points": [[20, 15, 1]]}

    by_score = client.post("/api/segment", json=base).json()
    assert by_score["chosen"] == 0 and len(by_score["candidates"]) == 3  # the confident small part
    assert by_score["area"] == 4 * 6
    assert [c["bbox"] for c in by_score["candidates"]] == [[10, 10, 16, 14], [10, 10, 25, 20], [10, 10, 40, 30]]

    largest = client.post("/api/segment", json={**base, "select": "largest"}).json()
    assert largest["chosen"] == 2 and largest["area"] == 20 * 30

    forced = client.post("/api/segment", json={**base, "choose": 1}).json()
    assert forced["chosen"] == 1 and forced["area"] == 10 * 15
    assert forced["score"] == pytest.approx(0.80)

    mask = decode_b64(by_score["mask"])  # an RGBA layer: translucent fill, solid outline
    assert mask.shape == (80, 120, 4) and (mask[..., 3] > 0).sum() >= by_score["area"]


def test_multi_point_and_box_prompts_return_a_single_mask(client):
    sid = open_image(client)["session"]
    multi = client.post("/api/segment", json={"session": sid, "points": [[20, 15, 1], [30, 20, 1], [5, 5, 0]]}).json()
    assert "candidates" not in multi and multi["area"] == 20 * 30
    boxed = client.post("/api/segment", json={"session": sid, "box": [5, 5, 60, 60]}).json()
    assert boxed["bbox"] == [10, 10, 40, 30]


def test_bad_prompts_are_reported_not_crashed(client):
    sid = open_image(client)["session"]
    empty = client.post("/api/segment", json={"session": sid})
    assert empty.status_code == 400 and "add a point" in empty.json()["error"]
    outside = client.post("/api/segment", json={"session": sid, "points": [[9999, 5, 1]]})
    assert outside.status_code == 400 and "outside" in outside.json()["error"]
    assert client.post("/api/segment", json={"session": sid, "points": [[1, 1, 1]], "choose": 3}).status_code == 422
    assert client.post("/api/segment", json={"session": sid, "points": [[1, 1, 1]], "select": "smallest"}).status_code == 422
    too_many = {"session": sid, "points": [[1, 1, 1]] * 65}
    assert client.post("/api/segment", json=too_many).status_code == 422
    assert client.post("/api/segment", json={"session": "nope", "points": [[1, 1, 1]]}).status_code == 404


def test_previews_do_not_change_what_gets_exported(client):
    sid = open_image(client)["session"]
    preview = client.post("/api/segment", json={"session": sid, "points": [[20, 15, 1]], "preview": True}).json()
    assert set(preview) == {"mask", "score", "ms"}  # nothing stored, no candidates
    assert client.get(f"/api/export/{sid}?kind=mask").status_code == 400  # still no committed mask

    client.post("/api/segment", json={"session": sid, "points": [[20, 15, 1]], "select": "largest"})
    client.post("/api/segment", json={"session": sid, "points": [[20, 15, 1]], "preview": True})  # must not overwrite
    exported = client.get(f"/api/export/{sid}?kind=mask")
    assert exported.status_code == 200
    assert (np.asarray(Image.open(io.BytesIO(exported.content))) > 0).sum() == 20 * 30


def test_sessions_share_one_model_without_re_encoding_and_old_ones_are_evicted(client, fake):
    first = open_image(client, png((100, 60)), "a.png")["session"]
    second = open_image(client, png((160, 90)), "b.png")["session"]
    assert fake.encodes == 2

    for sid, shape in [(first, (60, 100)), (second, (90, 160)), (first, (60, 100))]:  # swap back and forth
        mask = decode_b64(client.post("/api/segment", json={"session": sid, "points": [[20, 15, 1]]}).json()["mask"])
        assert mask.shape[:2] == shape  # the right image's embedding was active each time
    assert fake.encodes == 2  # ...and nothing was re-encoded

    # max_sessions is 2, and `first` was used most recently, so a third image evicts `second` (least
    # recently used), not the oldest-opened one.
    third = open_image(client, png((90, 90)), "c.png")["session"]
    assert client.post("/api/segment", json={"session": second, "points": [[1, 1, 1]]}).status_code == 404
    assert client.post("/api/segment", json={"session": first, "points": [[1, 1, 1]]}).status_code == 200
    assert client.post("/api/segment", json={"session": third, "points": [[1, 1, 1]]}).status_code == 200


# -------------------------------------------------------------------------- everything


def test_segment_everything_returns_a_decodable_label_map_and_can_be_picked(client):
    sid = open_image(client)["session"]
    res = client.post("/api/everything", json={"session": sid}).json()
    assert res["count"] == 2
    rgb = decode_b64(res["labels"])
    ids = rgb[..., 0].astype(np.uint16) | (rgb[..., 1].astype(np.uint16) << 8)
    assert set(np.unique(ids)) == {0, 1, 2}  # background, big, small (small drawn on top)
    assert ids[6, 6] == 2 and ids[40, 10] == 1 and ids[40, 100] == 0

    picked = client.post("/api/pick", json={"session": sid, "segment": 2}).json()
    assert picked["area"] == 16
    assert client.get(f"/api/export/{sid}?kind=mask").status_code == 200  # picking commits the mask
    assert client.post("/api/pick", json={"session": sid, "segment": 9}).status_code == 400


def test_pick_before_segment_everything_is_an_error(client):
    sid = open_image(client)["session"]
    assert client.post("/api/pick", json={"session": sid, "segment": 1}).status_code == 400


# ---------------------------------------------------------------------------- export


def test_exports(client):
    sid = open_image(client)["session"]
    assert client.get(f"/api/export/{sid}?kind=cutout").status_code == 400  # nothing to export yet
    client.post("/api/segment", json={"session": sid, "box": [0, 0, 100, 60]})

    def fetch(kind):
        res = client.get(f"/api/export/{sid}?kind={kind}")
        assert res.status_code == 200 and res.headers["content-type"] == "image/png"
        return res, np.asarray(Image.open(io.BytesIO(res.content)))

    res, cut = fetch("cutout")
    assert cut.shape == (80, 120, 4) and res.headers["content-disposition"] == 'attachment; filename="photo_cutout.png"'
    assert (cut[..., 3] > 0).sum() == 20 * 30
    assert fetch("cutout-crop")[1].shape == (20, 30, 4)
    assert set(np.unique(fetch("mask")[1])) == {0, 255}
    assert fetch("overlay")[1].shape == (80, 120, 3)
    assert client.get(f"/api/export/{sid}?kind=exe").status_code == 400


def test_export_filenames_cannot_inject_headers(client):
    meta = open_image(client, name='evil"\r\nX-Injected: 1/../../name.png')
    client.post("/api/segment", json={"session": meta["session"], "box": [0, 0, 100, 60]})
    res = client.get(f"/api/export/{meta['session']}?kind=mask")
    assert res.status_code == 200 and "X-Injected" not in res.headers
    assert res.headers["content-disposition"].count('"') == 2 and "\n" not in res.headers["content-disposition"]


# ------------------------------------------------------------------------- samples


def test_samples_are_whitelisted(client):
    assert client.get("/api/sample-image/football_dribble.jpeg").status_code == 200
    for bad in ("../README.md", "..%2FREADME.md", "nope.png", "%2e%2e%2fpyproject.toml"):
        assert client.get(f"/api/sample-image/{bad}").status_code in (404, 400)
        assert client.post(f"/api/sample/{bad}").status_code in (404, 400)


def test_opening_a_sample_downscales_to_the_engine_limit(client):
    meta = client.post("/api/sample/starry_night.jpg").json()  # 736x414, engine max_side is 200
    assert max(meta["width"], meta["height"]) == 200 and meta["name"] == "starry_night.jpg"


# ---------------------------------------------------------------- real model (slow)


@pytest.fixture
def real_client(segmenter):
    return TestClient(create_app(DemoEngine(segmenter, samples_dir=SAMPLES)))


@pytest.mark.slow
def test_real_model_end_to_end_through_the_api(real_client, segmenter, monkeypatch):
    encodes = []
    original = segmenter._predictor.set_image
    monkeypatch.setattr(segmenter._predictor, "set_image", lambda *a, **k: (encodes.append(1), original(*a, **k))[1])

    football = real_client.post("/api/sample/football_dribble.jpeg").json()
    starry = real_client.post("/api/sample/starry_night.jpg").json()
    assert len(encodes) == 2

    body = {"session": football["session"], "points": [[352, 205, 1], [330, 250, 1], [395, 262, 1]]}
    first = real_client.post("/api/segment", json=body).json()
    assert 5_000 < first["area"] < 30_000 and "candidates" not in first  # Messi, one mask

    other = real_client.post("/api/segment", json={"session": starry["session"], "box": [222, 70, 296, 312]}).json()
    assert 3_000 < other["area"] < 30_000  # the cypress

    again = real_client.post("/api/segment", json=body).json()  # back to the first image
    assert again["area"] == first["area"] and again["bbox"] == first["bbox"]
    assert len(encodes) == 2  # swapping images never re-ran the encoder

    cut = real_client.get(f"/api/export/{football['session']}?kind=cutout-crop")
    assert cut.status_code == 200 and Image.open(io.BytesIO(cut.content)).mode == "RGBA"


@pytest.mark.slow
def test_real_model_encode_then_activate_matches_a_fresh_encode(segmenter):
    from imgseg.io import load_image

    a, b = load_image(SAMPLES / "football_dribble.jpeg"), load_image(SAMPLES / "starry_night.jpg")
    enc_a, enc_b = segmenter.encode(a), segmenter.encode(b)
    segmenter.activate(enc_a)
    swapped = segmenter.predict(box=(250, 120, 420, 310)).mask
    segmenter.set_image(a)
    fresh = segmenter.predict(box=(250, 120, 420, 310)).mask
    assert np.array_equal(swapped, fresh)
    assert segmenter.activate(enc_b).image is b or np.array_equal(segmenter.image, b)
