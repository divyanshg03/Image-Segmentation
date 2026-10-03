"use strict";

(() => {
  const $ = (selector) => document.querySelector(selector);
  const canvas = $("#canvas");
  const ctx = canvas.getContext("2d");
  const stage = $("#stage");

  const INCLUDE = "#2fd36b";
  const EXCLUDE = "#ff5a5a";
  const BOX = "#ffb938";
  const MAX_POINTS = 64;
  const DRAG_PX = 6;

  const state = {
    sid: null, name: "", w: 0, h: 0, img: null,
    points: [],        // {x, y, l}  l: 1 include, 0 exclude (image pixels)
    box: null,         // [x1, y1, x2, y2] in image pixels
    history: [],       // undo stack: {kind: "point"} | {kind: "box", prev}
    mode: "include",
    select: "score",
    choose: null,      // forced candidate index, or null
    mask: null,        // committed overlay (Image)
    result: null,      // last committed server response
    preview: null,     // hover-preview overlay (Image)
    hover: null,       // {x, y} in image pixels
    hoverOn: true,
    view: { scale: 1, ox: 0, oy: 0, fitted: true },
    cssW: 1, cssH: 1,
    dragBox: null,
    everything: null,  // {ids, count, overlay, hoverId, hoverLayer}
    spaceDown: false,
  };
  let drag = null;

  // ---------------------------------------------------------------- helpers

  function store(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      localStorage.setItem(key, value);
    } catch (_) { /* private mode: preferences just will not persist */ }
    return null;
  }

  let toastTimer = 0;
  function toast(message) {
    const el = $("#toast");
    el.textContent = message;
    el.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("hidden"), 5000);
  }

  async function api(path, options = {}) {
    const res = await fetch(path, options);
    const isJson = (res.headers.get("content-type") || "").includes("application/json");
    const data = isJson ? await res.json() : null;
    if (!res.ok) {
      const detail = data && (data.error || (typeof data.detail === "string" ? data.detail : "invalid request"));
      const err = new Error(detail || `request failed (${res.status})`);
      err.status = res.status;
      throw err;
    }
    return data;
  }
  const post = (path, body) => api(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("could not load image data"));
      img.src = src;
    });
  }
  const pngImage = (b64) => loadImage("data:image/png;base64," + b64);
  const round1 = (v) => Math.round(v * 10) / 10;

  let busyCount = 0;
  function busy(text) {
    busyCount += 1;
    $("#busy-text").textContent = text;
    $("#busy").classList.remove("hidden");
    return () => {
      busyCount = Math.max(0, busyCount - 1);
      if (!busyCount) $("#busy").classList.add("hidden");
    };
  }

  function announce(text) { $("#live").textContent = text; }

  // ---------------------------------------------------------------- view

  function fitScale() {
    const pad = 16;
    return Math.min((state.cssW - 2 * pad) / state.w, (state.cssH - 2 * pad) / state.h);
  }

  function fitView() {
    if (!state.img) return;
    const s = fitScale();
    state.view = { scale: s, ox: (state.cssW - state.w * s) / 2, oy: (state.cssH - state.h * s) / 2, fitted: true };
    requestDraw();
  }

  function zoomAt(factor, cx, cy) {
    if (!state.img) return;
    const v = state.view;
    const scale = Math.min(40, Math.max(fitScale() * 0.5, v.scale * factor));
    const k = scale / v.scale;
    v.ox = cx - (cx - v.ox) * k;
    v.oy = cy - (cy - v.oy) * k;
    v.scale = scale;
    v.fitted = false;
    requestDraw();
  }

  function resize() {
    const rect = stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    state.cssW = Math.max(1, Math.floor(rect.width));
    state.cssH = Math.max(1, Math.floor(rect.height));
    canvas.width = Math.round(state.cssW * dpr);
    canvas.height = Math.round(state.cssH * dpr);
    if (state.img && state.view.fitted) fitView();
    requestDraw();
  }

  const localPos = (e) => {
    const r = canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  const toImage = (p) => ({ x: (p.x - state.view.ox) / state.view.scale, y: (p.y - state.view.oy) / state.view.scale });
  const toScreen = (x, y) => ({ x: x * state.view.scale + state.view.ox, y: y * state.view.scale + state.view.oy });
  const inside = (q) => q.x >= 0 && q.y >= 0 && q.x < state.w && q.y < state.h;
  const clampBox = (a, b) => [
    Math.max(0, Math.min(a.x, b.x)), Math.max(0, Math.min(a.y, b.y)),
    Math.min(state.w, Math.max(a.x, b.x)), Math.min(state.h, Math.max(a.y, b.y)),
  ];

  // ---------------------------------------------------------------- drawing

  let drawQueued = false;
  function requestDraw() {
    if (drawQueued) return;
    drawQueued = true;
    requestAnimationFrame(draw);
  }

  function dot(pt, color, radius) {
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, radius + 2, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }

  function draw() {
    drawQueued = false;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!state.img) return;

    const { scale, ox, oy } = state.view;
    ctx.setTransform(dpr * scale, 0, 0, dpr * scale, dpr * ox, dpr * oy);
    ctx.imageSmoothingEnabled = scale < 2; // crisp pixels once zoomed in
    ctx.drawImage(state.img, 0, 0);

    const ev = state.everything;
    if (ev) {
      ctx.globalAlpha = 0.5;
      ctx.drawImage(ev.overlay, 0, 0);
      ctx.globalAlpha = 1;
      if (ev.hoverLayer) ctx.drawImage(ev.hoverLayer, 0, 0);
    } else {
      if (state.mask) ctx.drawImage(state.mask, 0, 0);
      if (state.preview && state.hover) ctx.drawImage(state.preview, 0, 0);
    }

    // Prompts are drawn in screen space so they keep a constant size when zooming.
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const box = state.dragBox || state.box;
    if (box) {
      const a = toScreen(box[0], box[1]);
      const b = toScreen(box[2], box[3]);
      ctx.lineWidth = 2;
      ctx.strokeStyle = BOX;
      ctx.setLineDash(state.dragBox ? [6, 4] : []);
      ctx.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y);
      ctx.setLineDash([]);
    }
    for (const p of state.points) dot(toScreen(p.x, p.y), p.l === 1 ? INCLUDE : EXCLUDE, 6);
  }

  // ---------------------------------------------------------------- segmentation requests
  // One request at a time. A committed click always beats a hover preview, and a preview
  // is dropped if anything changed while it was in flight.

  const queue = { commit: false, preview: false, running: false };
  let previewToken = 0;

  function dropPreview() {
    previewToken += 1;
    if (state.preview) { state.preview = null; requestDraw(); }
  }

  function requestCommit() {
    dropPreview();
    queue.commit = true;
    queue.preview = false;
    pump();
  }

  function requestPreview() {
    if (!queue.commit) queue.preview = true;
    pump();
  }

  async function pump() {
    if (queue.running) return;
    queue.running = true;
    try {
      while (queue.commit || queue.preview) {
        const commit = queue.commit;
        queue.commit = false;
        queue.preview = false;
        try {
          if (commit) await doCommit(); else await doPreview();
        } catch (err) {
          if (commit || err.status === 404) handleError(err);
          else console.warn("hover preview failed:", err.message); // no toast: it would fire on every mouse move
        }
      }
    } finally {
      queue.running = false;
    }
  }

  const promptPoints = () => state.points.map((p) => [round1(p.x), round1(p.y), p.l]);

  async function doCommit() {
    if (!state.sid) return;
    if (!state.points.length && !state.box) { clearResult(); return; }
    const res = await post("/api/segment", {
      session: state.sid, points: promptPoints(), box: state.box ? state.box.map(round1) : null,
      select: state.select, choose: state.choose, preview: false,
    });
    state.result = res;
    state.mask = await pngImage(res.mask);
    await showCandidates(res);
    showStats(res);
    updateExports();
    announce(`Mask updated. Score ${res.score.toFixed(2)}, ${percent(res.area)} of the image.`);
    requestDraw();
  }

  async function doPreview() {
    const eligible = state.sid && state.hoverOn && state.hover && !state.everything
      && state.mode !== "box" && !drag && state.points.length < MAX_POINTS;
    if (!eligible) return;
    const token = previewToken;
    const label = state.mode === "exclude" ? 0 : 1;
    const points = promptPoints().concat([[round1(state.hover.x), round1(state.hover.y), label]]);
    const res = await post("/api/segment", {
      session: state.sid, points, box: state.box ? state.box.map(round1) : null,
      select: state.select, choose: null, preview: true,
    });
    if (token !== previewToken || !state.hover) return; // stale
    state.preview = await pngImage(res.mask);
    $("#stat-decode").textContent = `${res.ms.toFixed(0)} ms`;
    requestDraw();
  }

  function handleError(err) {
    toast(err.message || "something went wrong");
    if (err.status === 404) {
      state.sid = null;
      $("#dropzone").classList.remove("hidden");
      updateHint();
    }
  }

  // ---------------------------------------------------------------- result panel

  const percent = (area) => {
    const pct = (100 * area) / (state.w * state.h);
    return `${pct < 10 ? pct.toFixed(1) : Math.round(pct)}%`;
  };

  function showStats(res) {
    $("#stat-score").textContent = res.score.toFixed(2);
    const area = $("#stat-area");
    area.textContent = res.area ? percent(res.area) : "empty";
    area.title = `${res.area.toLocaleString()} px`;
    $("#stat-decode").textContent = `${res.ms.toFixed(0)} ms`;
  }

  function clearResult() {
    state.mask = null;
    state.result = null;
    state.choose = null;
    $("#candidates-wrap").classList.add("hidden");
    $("#candidates").replaceChildren();
    for (const id of ["#stat-score", "#stat-area", "#stat-decode"]) $(id).textContent = "-";
    updateExports();
    updateHint();
    requestDraw();
  }

  // One crop shared by every candidate thumbnail: the union of their boxes plus a margin, so the
  // differences between candidates are visible instead of lost in the full frame.
  function thumbnailCrop(candidates) {
    const boxes = candidates.map((c) => c.bbox).filter(Boolean);
    if (!boxes.length) return [0, 0, state.w, state.h];
    let [x1, y1, x2, y2] = [Infinity, Infinity, -Infinity, -Infinity];
    for (const b of boxes) { x1 = Math.min(x1, b[0]); y1 = Math.min(y1, b[1]); x2 = Math.max(x2, b[2]); y2 = Math.max(y2, b[3]); }
    const margin = 0.2;
    let cw = Math.max((x2 - x1) * (1 + 2 * margin), 32);
    let ch = Math.max((y2 - y1) * (1 + 2 * margin), 24);
    if (cw / ch > 4 / 3) ch = (cw * 3) / 4; else cw = (ch * 4) / 3; // match the 4:3 tile
    cw = Math.min(cw, state.w);
    ch = Math.min(ch, state.h);
    const cx = (x1 + x2) / 2;
    const cy = (y1 + y2) / 2;
    return [Math.min(Math.max(cx - cw / 2, 0), state.w - cw), Math.min(Math.max(cy - ch / 2, 0), state.h - ch), cw, ch];
  }

  async function showCandidates(res) {
    const wrap = $("#candidates-wrap");
    const box = $("#candidates");
    box.replaceChildren();
    if (!res.candidates) { wrap.classList.add("hidden"); return; }
    const layers = await Promise.all(res.candidates.map((c) => pngImage(c.mask)));
    const [sx, sy, sw, sh] = thumbnailCrop(res.candidates);
    const tw = 120;
    const th = 90;
    const fit = Math.min(tw / sw, th / sh);
    const [dw, dh] = [sw * fit, sh * fit];
    const [dx, dy] = [(tw - dw) / 2, (th - dh) / 2];
    res.candidates.forEach((c, i) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "candidate";
      button.setAttribute("aria-pressed", String(i === res.chosen));
      button.setAttribute("aria-label", `Candidate ${i + 1}, score ${c.score.toFixed(2)}`);
      const thumb = document.createElement("canvas");
      thumb.width = tw;
      thumb.height = th;
      const g = thumb.getContext("2d");
      g.fillStyle = "#0b0c0f";
      g.fillRect(0, 0, tw, th);
      g.drawImage(state.img, sx, sy, sw, sh, dx, dy, dw, dh);
      g.drawImage(layers[i], sx, sy, sw, sh, dx, dy, dw, dh);
      const label = document.createElement("span");
      label.textContent = `${i + 1}   ${c.score.toFixed(2)}`;
      button.append(thumb, label);
      button.addEventListener("click", () => chooseCandidate(i));
      box.append(button);
    });
    wrap.classList.remove("hidden");
  }

  function chooseCandidate(index) {
    if (!state.result || !state.result.candidates || index >= state.result.candidates.length) return;
    state.choose = index;
    requestCommit();
  }

  function updateExports() {
    const ok = Boolean(state.sid && state.result && state.result.area > 0);
    document.querySelectorAll(".exports a").forEach((a) => {
      a.setAttribute("aria-disabled", String(!ok));
      if (ok) a.href = `/api/export/${state.sid}?kind=${a.dataset.kind}`;
      else a.removeAttribute("href");
    });
  }

  function updateHint() {
    let text;
    if (!state.sid) text = "Open an image or pick a sample to begin.";
    else if (state.everything) text = "Hover a segment to highlight it and click to select it. Esc leaves this mode.";
    else if (state.mode === "box") text = "Drag a box around the object.";
    else if (!state.points.length && !state.box) {
      text = "Click the object. Right-click or Alt+click excludes. Drag draws a box. Scroll to zoom, hold Space and drag to pan.";
    } else {
      const n = state.points.length;
      text = `${n} point${n === 1 ? "" : "s"}${state.box ? " + box" : ""}. Keep adding points to refine; Ctrl+Z undoes.`;
    }
    $("#hint").textContent = text;
    $("#everything-btn").textContent = state.everything ? "Leave segment everything" : "Segment everything";
  }

  // ---------------------------------------------------------------- prompt actions

  function addPoint(q, label) {
    if (state.points.length >= MAX_POINTS) { toast(`At most ${MAX_POINTS} points.`); return; }
    state.history.push({ kind: "point" });
    state.points.push({ x: q.x, y: q.y, l: label });
    state.choose = null;
    updateHint();
    requestCommit();
  }

  function setBox(box) {
    state.history.push({ kind: "box", prev: state.box });
    state.box = box;
    state.choose = null;
    updateHint();
    requestCommit();
  }

  function undo() {
    const last = state.history.pop();
    if (!last) return;
    if (last.kind === "point") state.points.pop(); else state.box = last.prev;
    state.choose = null;
    updateHint();
    requestCommit();
  }

  function reset() {
    state.points = [];
    state.box = null;
    state.history = [];
    exitEverything();
    dropPreview();
    clearResult();
  }

  function setMode(mode) {
    state.mode = mode;
    document.querySelectorAll(".tool").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === mode)));
    if (mode === "box") dropPreview();
    updateHint();
    requestDraw();
  }

  function setSelect(select) {
    state.select = select;
    store("imgseg.select", select);
    document.querySelectorAll("#select-group button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.select === select)));
    state.choose = null;
    if (state.points.length || state.box) requestCommit();
  }

  function setHover(on, persist = true) {
    state.hoverOn = on;
    $("#hover-toggle").checked = on;
    if (persist) store("imgseg.hover", on ? "1" : "0");
    if (!on) dropPreview();
  }

  // ---------------------------------------------------------------- segment everything

  function hsv(h, s, v) {
    const i = Math.floor(h * 6), f = h * 6 - i;
    const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
    return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6].map((c) => Math.round(c * 255));
  }
  const palette = (n) => Array.from({ length: n }, (_, i) => hsv((i * 0.61803398875) % 1, 0.65, 1));

  async function runEverything() {
    if (!state.sid) return;
    const done = busy("Segmenting everything: a few seconds...");
    try {
      const res = await post("/api/everything", { session: state.sid });
      if (!res.count) { toast("No segments passed the quality filters."); return; }

      const labels = await pngImage(res.labels);
      const probe = document.createElement("canvas");
      probe.width = state.w;
      probe.height = state.h;
      const pg = probe.getContext("2d", { willReadFrequently: true });
      pg.drawImage(labels, 0, 0);
      const px = pg.getImageData(0, 0, state.w, state.h).data;
      const ids = new Uint16Array(state.w * state.h);
      for (let i = 0, j = 0; i < ids.length; i += 1, j += 4) ids[i] = px[j] | (px[j + 1] << 8);

      const overlay = document.createElement("canvas");
      overlay.width = state.w;
      overlay.height = state.h;
      const og = overlay.getContext("2d");
      const out = og.createImageData(state.w, state.h);
      const colors = palette(res.count);
      for (let i = 0, j = 0; i < ids.length; i += 1, j += 4) {
        const id = ids[i];
        if (!id) continue;
        const c = colors[id - 1];
        out.data[j] = c[0]; out.data[j + 1] = c[1]; out.data[j + 2] = c[2]; out.data[j + 3] = 255;
      }
      og.putImageData(out, 0, 0);

      dropPreview();
      state.everything = { ids, count: res.count, overlay, hoverId: 0, hoverLayer: null };
      announce(`${res.count} segments found.`);
      updateHint();
      requestDraw();
    } catch (err) {
      handleError(err);
    } finally {
      done();
    }
  }

  function exitEverything() {
    if (!state.everything) return;
    state.everything = null;
    canvas.classList.remove("picking");
    updateHint();
    requestDraw();
  }

  function setHoverId(id) {
    const ev = state.everything;
    if (!ev || ev.hoverId === id) return;
    ev.hoverId = id;
    ev.hoverLayer = null;
    if (id) {
      const layer = document.createElement("canvas");
      layer.width = state.w;
      layer.height = state.h;
      const g = layer.getContext("2d");
      const data = g.createImageData(state.w, state.h);
      for (let i = 0, j = 0; i < ev.ids.length; i += 1, j += 4) {
        if (ev.ids[i] === id) { data.data[j] = 255; data.data[j + 1] = 255; data.data[j + 2] = 255; data.data[j + 3] = 110; }
      }
      g.putImageData(data, 0, 0);
      ev.hoverLayer = layer;
    }
    requestDraw();
  }

  const idAt = (q) => (state.everything && inside(q)
    ? state.everything.ids[Math.floor(q.y) * state.w + Math.floor(q.x)] : 0);

  async function pickSegment(id) {
    try {
      const res = await post("/api/pick", { session: state.sid, segment: id });
      state.points = [];
      state.box = null;
      state.history = [];
      state.choose = null;
      state.result = res;
      state.mask = await pngImage(res.mask);
      exitEverything();
      await showCandidates({});
      showStats(res);
      updateExports();
      announce(`Segment selected, ${percent(res.area)} of the image.`);
      requestDraw();
    } catch (err) {
      handleError(err);
    }
  }

  // ---------------------------------------------------------------- opening images

  async function startSession(meta) {
    const img = await loadImage(`/api/image/${meta.session}`);
    Object.assign(state, {
      sid: meta.session, name: meta.name, w: img.naturalWidth, h: img.naturalHeight, img,
      points: [], box: null, history: [], everything: null, hover: null,
    });
    dropPreview();
    $("#dropzone").classList.add("hidden");
    const chip = $("#image-chip");
    chip.textContent = `${meta.name} - ${state.w}x${state.h}`;
    chip.classList.remove("hidden");
    $("#stat-encode").textContent = `${meta.encode_ms.toFixed(0)} ms`;
    clearResult();
    fitView();
    announce(`Image ready. Encoded in ${meta.encode_ms.toFixed(0)} milliseconds.`);
    canvas.focus({ preventScroll: true });
  }

  async function openWith(request) {
    const done = busy("Encoding image...");
    try {
      await startSession(await request());
    } catch (err) {
      handleError(err);
    } finally {
      done();
    }
  }

  function openFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) { toast("That does not look like an image."); return; }
    openWith(() => {
      const form = new FormData();
      form.append("file", file, file.name || "pasted.png");
      return api("/api/image", { method: "POST", body: form });
    });
  }

  const openSample = (name) => openWith(() => api(`/api/sample/${encodeURIComponent(name)}`, { method: "POST" }));

  // ---------------------------------------------------------------- events

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  canvas.addEventListener("pointerdown", (e) => {
    if (!state.img) return;
    canvas.focus({ preventScroll: true });
    dropPreview();
    const p = localPos(e);
    if (e.button === 1 || (e.button === 0 && state.spaceDown)) {
      drag = { kind: "pan", last: p };
      canvas.classList.add("panning");
    } else if (e.button === 0 || e.button === 2) {
      drag = { kind: "press", start: p, button: e.button, alt: e.altKey, moved: false };
    } else {
      return;
    }
    canvas.setPointerCapture(e.pointerId);
    e.preventDefault();
  });

  canvas.addEventListener("pointermove", (e) => {
    const p = localPos(e);
    if (drag && drag.kind === "pan") {
      state.view.ox += p.x - drag.last.x;
      state.view.oy += p.y - drag.last.y;
      state.view.fitted = false;
      drag.last = p;
      requestDraw();
    } else if (drag) {
      if (!drag.moved && Math.hypot(p.x - drag.start.x, p.y - drag.start.y) > DRAG_PX) drag.moved = true;
      if (drag.moved && drag.button === 0 && !state.everything) {
        state.dragBox = clampBox(toImage(drag.start), toImage(p));
        requestDraw();
      }
    } else if (state.img) {
      const q = toImage(p);
      state.hover = q;
      if (state.everything) {
        const id = idAt(q);
        setHoverId(id);
        canvas.classList.toggle("picking", id > 0);
      } else if (state.hoverOn && state.mode !== "box" && inside(q)) {
        requestPreview();
      } else if (state.preview) {
        dropPreview();
      }
    }
  });

  canvas.addEventListener("pointerup", (e) => {
    if (!drag) return;
    const d = drag;
    drag = null;
    canvas.classList.remove("panning");
    try { canvas.releasePointerCapture(e.pointerId); } catch (_) { /* already released */ }
    if (d.kind === "pan") return;

    const q = toImage(localPos(e));
    if (!d.moved) {
      if (!inside(q)) return;
      if (state.everything) { const id = idAt(q); if (id) pickSegment(id); return; }
      if (state.mode === "box" && d.button === 0) return; // box mode needs a drag
      addPoint(q, d.button === 2 || d.alt || state.mode === "exclude" ? 0 : 1);
    } else if (d.button === 0 && !state.everything) {
      const box = state.dragBox;
      state.dragBox = null;
      if (box && box[2] - box[0] >= 3 && box[3] - box[1] >= 3) setBox(box); else requestDraw();
    } else {
      state.dragBox = null;
      requestDraw();
    }
  });

  canvas.addEventListener("pointercancel", () => {
    drag = null;
    state.dragBox = null;
    canvas.classList.remove("panning");
    requestDraw();
  });

  canvas.addEventListener("pointerleave", () => {
    if (drag) return;
    state.hover = null;
    dropPreview();
    if (state.everything) { setHoverId(0); canvas.classList.remove("picking"); }
  });

  canvas.addEventListener("wheel", (e) => {
    if (!state.img) return;
    e.preventDefault();
    const p = localPos(e);
    zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)), p.x, p.y);
  }, { passive: false });

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target === canvas) { state.spaceDown = true; e.preventDefault(); return; }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    switch (e.key.toLowerCase()) {
      case "i": setMode("include"); break;
      case "e": setMode("exclude"); break;
      case "b": setMode("box"); break;
      case "r": reset(); break;
      case "h": setHover(!state.hoverOn); break;
      case "0": fitView(); break;
      case "+": case "=": zoomAt(1.25, state.cssW / 2, state.cssH / 2); break;
      case "-": zoomAt(0.8, state.cssW / 2, state.cssH / 2); break;
      case "1": case "2": case "3": chooseCandidate(Number(e.key) - 1); break;
      case "escape":
        if (state.everything) exitEverything();
        else if (drag) { drag = null; state.dragBox = null; requestDraw(); }
        break;
      default: break;
    }
  });
  window.addEventListener("keyup", (e) => { if (e.code === "Space") state.spaceDown = false; });

  document.querySelectorAll(".tool").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  document.querySelectorAll("#select-group button").forEach((b) => b.addEventListener("click", () => setSelect(b.dataset.select)));
  $("#undo-btn").addEventListener("click", undo);
  $("#reset-btn").addEventListener("click", reset);
  $("#hover-toggle").addEventListener("change", (e) => setHover(e.target.checked));
  $("#everything-btn").addEventListener("click", () => (state.everything ? exitEverything() : runEverything()));
  $("#zoom-in").addEventListener("click", () => zoomAt(1.25, state.cssW / 2, state.cssH / 2));
  $("#zoom-out").addEventListener("click", () => zoomAt(0.8, state.cssW / 2, state.cssH / 2));
  $("#zoom-fit").addEventListener("click", fitView);

  const fileInput = $("#file-input");
  $("#open-btn").addEventListener("click", () => fileInput.click());
  $("#drop-open").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => { openFile(fileInput.files[0]); fileInput.value = ""; });

  const dropzone = $("#dropzone");
  stage.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragging"); });
  stage.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
  stage.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragging");
    openFile(e.dataTransfer.files[0]);
  });
  document.addEventListener("paste", (e) => {
    const file = Array.from((e.clipboardData && e.clipboardData.files) || []).find((f) => f.type.startsWith("image/"));
    if (file) { e.preventDefault(); openFile(file); }
  });

  document.querySelectorAll(".exports a").forEach((a) => a.addEventListener("click", (e) => {
    if (a.getAttribute("aria-disabled") === "true") e.preventDefault();
  }));

  new ResizeObserver(resize).observe(stage);
  window.addEventListener("resize", resize);

  // ---------------------------------------------------------------- start

  async function init() {
    resize();
    updateExports();
    updateHint();
    try {
      const info = await api("/api/info");
      const chip = $("#device-chip");
      chip.textContent = `${info.device_name}${info.fp16 ? " - fp16" : ""}`;
      chip.title = `Running on ${info.device}`;
      chip.classList.add("ok");

      const savedHover = store("imgseg.hover");
      setHover(savedHover === null ? info.device === "cuda" : savedHover === "1", false); // CPU decode is too slow for live preview
      const savedSelect = store("imgseg.select");
      if (savedSelect === "score" || savedSelect === "largest") setSelect(savedSelect);

      const samples = $("#samples");
      for (const name of info.samples) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "sample";
        const thumb = document.createElement("img");
        thumb.src = `/api/sample-image/${encodeURIComponent(name)}`;
        thumb.alt = "";
        thumb.loading = "lazy";
        const caption = document.createElement("span");
        caption.textContent = name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ");
        button.append(thumb, caption);
        button.addEventListener("click", () => openSample(name));
        samples.append(button);
      }
    } catch (err) {
      $("#device-chip").textContent = "server not reachable";
      toast(err.message);
    }
  }

  init();
})();
