"""End-to-end check of the demo UI: starts the real server, drives it in headless Chrome with real mouse and
keyboard events over the DevTools protocol, saves screenshots and fails on any JS error or CSP violation.

    pip install websocket-client            # the only extra needed
    python scripts/check_demo_ui.py         # screenshots land in outputs/ui_check/

It needs the SAM weights and a Chrome/Chromium/Edge install (auto-detected, or pass --chrome). The checks are
written for a GPU, where hover previews answer in about 50 ms; on a CPU pass --settle 3 (or more) to give each
step longer to finish.
"""
import argparse, base64, json, os, shutil, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
import websocket

REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default=str(REPO / "outputs" / "ui_check"), help="where screenshots and the server log go")
ap.add_argument("--chrome", help="path to Chrome / Chromium / Edge")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--debug-port", type=int, default=9333)
ap.add_argument("--settle", type=float, default=0.9, help="seconds to wait after each interaction")
args = ap.parse_args()

def find_chrome():
    if args.chrome: return args.chrome
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge"):
        if shutil.which(name): return shutil.which(name)
    for path in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                 r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                 r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                 "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"):
        if Path(path).exists(): return path
    sys.exit("No Chrome/Chromium/Edge found; pass --chrome PATH")

CHROME = find_chrome()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
APP_PORT, DBG_PORT = args.port, args.debug_port
URL = f"http://127.0.0.1:{APP_PORT}/"
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")

results, problems = [], []
def check(name, ok, detail=""):
    results.append((name, bool(ok), detail)); print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)

def wait_http(url, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        try: return urllib.request.urlopen(url, timeout=2).read()
        except Exception: time.sleep(0.5)
    raise TimeoutError(url)

server = subprocess.Popen([sys.executable, "-m", "imgseg", "demo", "--port", str(APP_PORT)], cwd=REPO, env=env,
                          stdout=open(OUT / "server.log", "w"), stderr=subprocess.STDOUT)
profile = tempfile.mkdtemp(prefix="imgseg-chrome-")
chrome = None
try:
    wait_http(URL + "api/info")
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={DBG_PORT}", f"--user-data-dir={profile}",
                               "--no-first-run", "--no-default-browser-check", "--disable-extensions", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_http(f"http://127.0.0.1:{DBG_PORT}/json/version")
    target = next(t for t in json.loads(urllib.request.urlopen(f"http://127.0.0.1:{DBG_PORT}/json/list").read()) if t["type"] == "page")
    ws = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=60, suppress_origin=True)

    _id = 0; events = []
    def send(method, **params):
        global _id
        _id += 1; mid = _id
        ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(ws.recv())
            if msg.get("id") == mid:
                if "error" in msg: raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            events.append(msg)
    def pump(seconds):
        ws.settimeout(0.2); end = time.time() + seconds
        while time.time() < end:
            try: events.append(json.loads(ws.recv()))
            except websocket.WebSocketTimeoutException: pass
        ws.settimeout(60)
    def js(expr, await_promise=False):
        r = send("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=await_promise)
        if "exceptionDetails" in r: raise RuntimeError(r["exceptionDetails"])
        return r["result"].get("value")
    def shot(name):
        data = send("Page.captureScreenshot", format="png")["data"]
        (OUT / f"{name}.png").write_bytes(base64.b64decode(data))
    def wait_for(expr, timeout=60, what=""):
        end = time.time() + timeout
        while time.time() < end:
            if js(expr): return True
            pump(0.15)
        raise TimeoutError(what or expr)

    for domain in ("Page", "Runtime", "Log", "Network"): send(f"{domain}.enable")
    net = []
    send("Emulation.setDeviceMetricsOverride", width=1440, height=900, deviceScaleFactor=1, mobile=False)
    send("Page.navigate", url=URL)
    wait_for("document.readyState === 'complete' && document.querySelector('#device-chip').textContent !== 'connecting...'", what="page load")
    pump(0.8)
    shot("01_empty")
    chip_text = js("document.querySelector('#device-chip').textContent")
    check("device chip names the hardware", chip_text not in ("", "connecting...", "server not reachable"), chip_text)
    n_samples = js("document.querySelectorAll('.sample').length")
    check("sample thumbnails are listed", n_samples >= 1, f"{n_samples} samples")
    if not js("document.querySelector('#hover-toggle').checked"):   # off by default on CPU: switch it on for the test
        js("document.querySelector('#hover-toggle').click()")
    check("hover preview is on", js("document.querySelector('#hover-toggle').checked") is True)

    # --- open a sample -------------------------------------------------------------------------
    js("document.querySelector('.sample').click()")
    wait_for("!document.querySelector('#image-chip').classList.contains('hidden') && document.querySelector('#busy').classList.contains('hidden')", what="sample load")
    pump(0.5)
    chip = js("document.querySelector('#image-chip').textContent"); print("   image chip:", chip)
    W, H = [int(v) for v in chip.split(" - ")[1].split("x")]
    check("sample opened and encoded", "football_dribble" in chip and js("document.querySelector('#stat-encode').textContent") != "-", chip)
    check("drop zone hidden after opening", js("document.querySelector('#dropzone').classList.contains('hidden')"))
    shot("02_loaded")

    rect = js("(() => { const r = document.querySelector('#canvas').getBoundingClientRect(); return [r.left, r.top, Math.floor(r.width), Math.floor(r.height)]; })()")
    def to_px(x, y):
        s = min((rect[2] - 32) / W, (rect[3] - 32) / H)
        return rect[0] + (rect[2] - W * s) / 2 + x * s, rect[1] + (rect[3] - H * s) / 2 + y * s
    def mouse(kind, x, y, button="none", buttons=0, clicks=0, mods=0):
        px, py = to_px(x, y)
        send("Input.dispatchMouseEvent", type=kind, x=px, y=py, button=button, buttons=buttons, clickCount=clicks, modifiers=mods)
    def click(x, y, button="left", mods=0):
        b = 1 if button == "left" else 2
        mouse("mouseMoved", x, y, mods=mods); mouse("mousePressed", x, y, button, b, 1, mods); mouse("mouseReleased", x, y, button, 0, 1, mods)
    def key(k, code, vk, mods=0):
        send("Input.dispatchKeyEvent", type="keyDown", key=k, code=code, windowsVirtualKeyCode=vk, modifiers=mods, text=k if not mods else "")
        send("Input.dispatchKeyEvent", type="keyUp", key=k, code=code, windowsVirtualKeyCode=vk, modifiers=mods)
    def settled(): pump(args.settle)

    # --- hover preview ---------------------------------------------------------------------------
    mouse("mouseMoved", 20, 20); mouse("mouseMoved", 352, 205); settled()
    decode_after_hover = js("document.querySelector('#stat-decode').textContent")
    check("hover preview ran (decode time shown without a click)", decode_after_hover.endswith("ms"), decode_after_hover)
    shot("03_hover_preview")

    # --- first click: three candidates -------------------------------------------------------------
    click(352, 205); settled()
    check("click produced a score", js("document.querySelector('#stat-score').textContent") not in ("-", ""), js("document.querySelector('#stat-score').textContent"))
    n_cand = js("document.querySelectorAll('#candidates .candidate').length")
    check("a lone click shows 3 candidates", n_cand == 3 and not js("document.querySelector('#candidates-wrap').classList.contains('hidden')"), f"{n_cand}")
    check("exports enabled after a mask exists", js("document.querySelector('.exports a').getAttribute('aria-disabled')") == "false")
    shot("04_single_click")

    # --- refine with more points + an exclude click -------------------------------------------------------
    click(330, 250); settled(); click(395, 262); settled()
    check("three points -> single mask, candidates hidden", js("document.querySelector('#candidates-wrap').classList.contains('hidden')"))
    area3 = js("document.querySelector('#stat-area').textContent")
    click(297, 277, button="right"); settled()
    area_ex = js("document.querySelector('#stat-area').textContent")
    hint = js("document.querySelector('#hint').textContent")
    check("right-click added an exclude point", hint.startswith("4 points"), hint)
    check("exclude point changed the mask", area_ex != area3, f"{area3} -> {area_ex}")
    shot("05_refined_with_exclude")

    key("z", "KeyZ", 90, mods=2); settled()
    check("Ctrl+Z undoes the last point", js("document.querySelector('#hint').textContent").startswith("3 points"), js("document.querySelector('#hint').textContent"))
    key("r", "KeyR", 82); settled()
    check("R resets everything", js("document.querySelector('#stat-score').textContent") == "-" and js("document.querySelector('.exports a').getAttribute('aria-disabled')") == "true")

    # --- box by dragging in the default mode ----------------------------------------------------------------
    mouse("mouseMoved", 250, 120); mouse("mousePressed", 250, 120, "left", 1, 1)
    for t in (0.25, 0.5, 0.75, 1.0): mouse("mouseMoved", 250 + 170 * t, 120 + 190 * t, "left", 1)
    mouse("mouseReleased", 420, 310, "left", 0, 1); settled()
    hint = js("document.querySelector('#hint').textContent")
    check("dragging draws a box and segments", "box" in hint and js("document.querySelector('#stat-score').textContent") != "-", hint)
    shot("06_box")

    # --- candidate selection via keyboard + segmented control ----------------------------------------------------
    key("r", "KeyR", 82); click(352, 205); settled()
    chosen = lambda: js("[...document.querySelectorAll('#candidates .candidate')].findIndex(b => b.getAttribute('aria-pressed') === 'true')")
    default_choice = chosen()
    key("2", "Digit2", 50); settled()
    check("key 2 selects candidate 2", chosen() == 1, f"{default_choice} -> {chosen()}")
    key("3", "Digit3", 51); settled()
    check("key 3 selects candidate 3", chosen() == 2, f"-> {chosen()}")
    js("document.querySelector('#select-group [data-select=largest]').click()"); settled()
    check("'Largest' switches to the widest candidate (index 0 for this click)", chosen() == 0, f"chosen={chosen()}")
    js("document.querySelector('#select-group [data-select=score]').click()"); settled()
    check("'SAM's pick' returns to the top-scoring candidate", chosen() == default_choice, f"chosen={chosen()}")

    # --- zoom ------------------------------------------------------------------------------------------------------
    px, py = to_px(352, 205)
    send("Input.dispatchMouseEvent", type="mouseWheel", x=px, y=py, deltaX=0, deltaY=-400)
    pump(0.5); shot("07_zoomed")

    # --- segment everything ------------------------------------------------------------------------------------------
    key("0", "Digit0", 48); key("r", "KeyR", 82)
    js("document.querySelector('#everything-btn').click()")
    wait_for("document.querySelector('#live').textContent.includes('segments found') && document.querySelector('#busy').classList.contains('hidden')", timeout=90, what="segment everything")
    live = js("document.querySelector('#live').textContent"); print("   ", live)
    mouse("mouseMoved", 20, 20); mouse("mouseMoved", 352, 205); pump(1.0)
    shot("08_everything_hover")
    click(60, 60); pump(1.5)
    check("clicking a segment selects it as the mask", "Segment selected" in js("document.querySelector('#live').textContent"), js("document.querySelector('#live').textContent"))
    check("segment-everything mode exits after picking", js("document.querySelector('#everything-btn').textContent") == "Segment everything")
    shot("09_segment_picked")

    # --- export really downloads a PNG -------------------------------------------------------------------------------------
    info = js("""fetch(document.querySelector('.exports a[data-kind="cutout-crop"]').href).then(async r => { const b = await r.blob(); return [r.status, r.headers.get('content-type'), b.size, r.headers.get('content-disposition')]; })""", await_promise=True)
    check("cut-out export is a PNG download", info[0] == 200 and info[1] == "image/png" and info[2] > 1000 and "attachment" in info[3], str(info))

    # --- real file upload through the file input ---------------------------------------------------------------
    doc = send("DOM.getDocument", depth=0)["root"]["nodeId"]
    inp = send("DOM.querySelector", nodeId=doc, selector="#file-input")["nodeId"]
    upload_path = str(REPO / "samples" / "starry_night.jpg")
    send("DOM.setFileInputFiles", nodeId=inp, files=[upload_path])
    wait_for("document.querySelector('#image-chip').textContent.includes('starry_night')", timeout=30, what="upload")
    pump(0.6)
    check("uploading a file opens it", "starry_night.jpg" in js("document.querySelector('#image-chip').textContent"), js("document.querySelector('#image-chip').textContent"))
    uw, uh = [int(v) for v in js("document.querySelector('#image-chip').textContent").split(" - ")[1].split("x")]
    check("uploaded image keeps its aspect ratio", abs(uw / uh - 736 / 414) < 0.01, f"{uw}x{uh}")
    W, H = uw, uh
    click(250, 190); settled()   # a click on the cypress tree region of the uploaded painting
    check("segmenting the uploaded image works", js("document.querySelector('#stat-score').textContent") != "-", js("document.querySelector('#stat-score').textContent"))
    shot("12_uploaded")
    notimg = str(REPO / "README.md")
    send("DOM.setFileInputFiles", nodeId=inp, files=[notimg]); pump(1.0)
    toast = js("document.querySelector('#toast').textContent")
    check("a non-image file is refused with a message", "image" in toast.lower() and "starry_night" in js("document.querySelector('#image-chip').textContent"), toast)

    # --- other layouts / themes ------------------------------------------------------------------------------------------------
    send("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "dark"}]); pump(0.5); shot("10_dark")
    send("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
    send("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=2, mobile=True); pump(0.8); shot("11_mobile")
    scroll_w = js("document.documentElement.scrollWidth"); check("no horizontal scroll on a phone-width screen", scroll_w <= 392, f"scrollWidth={scroll_w}")

    # --- network summary ----
    for e in events:
        if e.get("method") == "Network.requestWillBeSent":
            r = e["params"]["request"]
            if "/api/" in r["url"]: net.append((r["url"].split("8765")[1], r.get("postData", "")))
    segs = [d for u, d in net if u == "/api/segment"]
    print(f"   API calls: {len(net)} | /api/segment: {len(segs)} (previews: {sum('\"preview\":true' in d for d in segs)})")
    bad = [e["params"]["response"]["url"] + " -> " + str(e["params"]["response"]["status"]) for e in events
           if e.get("method") == "Network.responseReceived" and e["params"]["response"]["status"] >= 400]
    print("   HTTP errors:", bad or "none")
    # --- errors ---------------------------------------------------------------------------------------------------------------------
    pump(0.5)
    for e in events:
        m = e.get("method")
        if m == "Runtime.exceptionThrown": problems.append("exception: " + json.dumps(e["params"]["exceptionDetails"].get("exception", e["params"]["exceptionDetails"]))[:300])
        elif m == "Runtime.consoleAPICalled" and e["params"]["type"] in ("error", "assert"): problems.append("console." + e["params"]["type"] + ": " + json.dumps(e["params"]["args"])[:300])
        elif m == "Log.entryAdded" and e["params"]["entry"]["level"] == "error": problems.append("log: " + e["params"]["entry"]["text"][:300])
        elif m == "Runtime.consoleAPICalled" and e["params"]["type"] == "warning": problems.append("console.warn: " + json.dumps(e["params"]["args"])[:300])
    check("no JavaScript errors, console errors or CSP violations", not problems, "; ".join(problems[:3]))
finally:
    for proc in (chrome, server):
        if proc:
            proc.terminate()
            try: proc.wait(10)
            except Exception: proc.kill()
    shutil.rmtree(profile, ignore_errors=True)

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} UI checks passed")
sys.exit(1 if failed else 0)
