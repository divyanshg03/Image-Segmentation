"""FastAPI app for the demo UI. Run it with `imgseg demo` (needs `pip install -e ".[demo]"`)."""

from __future__ import annotations

import logging
import webbrowser
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .engine import MAX_POINTS, MAX_UPLOAD_BYTES, DemoEngine, DemoError, ImageRejected

log = logging.getLogger("imgseg.demo")
STATIC = Path(__file__).resolve().parent / "static"

# Everything the page needs is served from this origin; nothing inline, nothing remote.
CSP = (
    "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)


class SegmentRequest(BaseModel):
    session: str
    points: list[tuple[float, float, int]] = Field(default_factory=list, max_length=MAX_POINTS)
    box: tuple[float, float, float, float] | None = None
    select: Literal["score", "largest"] = "score"
    choose: int | None = Field(default=None, ge=0, le=2)
    preview: bool = False


class EverythingRequest(BaseModel):
    session: str
    points_per_side: int = Field(default=32, ge=8, le=64)


class PickRequest(BaseModel):
    session: str
    segment: int = Field(ge=1)


def create_app(engine: DemoEngine, allowed_hosts: list[str] | None = None) -> FastAPI:
    """`allowed_hosts` restricts the Host header (defence against DNS rebinding on a loopback server)."""
    app = FastAPI(title="imgseg demo", docs_url=None, redoc_url=None, openapi_url=None)
    if allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.middleware("http")
    async def security_headers(request, call_next):
        # A page on another site can fire simple cross-origin POSTs (e.g. a form upload) at a server on
        # localhost without any CORS permission. Browsers label them with an Origin; refuse foreign ones.
        origin = request.headers.get("origin")
        if (request.method not in ("GET", "HEAD", "OPTIONS") and origin
                and urlparse(origin).netloc != request.headers.get("host", "")):
            return JSONResponse({"error": "cross-origin requests are not allowed"}, status_code=403)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(DemoError)
    async def demo_error(request, exc: DemoError):
        return JSONResponse({"error": str(exc)}, status_code=exc.status)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/info")
    def info():
        return engine.info()

    @app.post("/api/image")
    async def upload(file: UploadFile = File(...)):
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise ImageRejected(f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        return await run_in_threadpool(engine.open_bytes, data, file.filename or "upload")

    @app.post("/api/sample/{name}")
    def open_sample(name: str):
        return engine.open_sample(name)

    @app.get("/api/sample-image/{name}")
    def sample_image(name: str):
        return FileResponse(engine.sample_path(name))

    @app.get("/api/image/{session}")
    def image(session: str):
        return Response(engine.image_png(session), media_type="image/png")

    @app.post("/api/segment")
    def segment(req: SegmentRequest):
        return engine.segment(req.session, points=req.points, box=req.box, select=req.select,
                              choose=req.choose, preview=req.preview)

    @app.post("/api/everything")
    def everything(req: EverythingRequest):
        return engine.everything(req.session, points_per_side=req.points_per_side)

    @app.post("/api/pick")
    def pick(req: PickRequest):
        return engine.pick(req.session, req.segment)

    @app.get("/api/export/{session}")
    def export(session: str, kind: str = "cutout"):
        data, filename = engine.export(session, kind)
        return Response(data, media_type="image/png",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def serve(engine: DemoEngine, host: str = "127.0.0.1", port: int = 8000, open_browser: bool = False) -> None:
    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1"):
        log.warning("Listening on %s: the demo has no authentication, so anyone who can reach this "
                    "address can use your GPU and upload files.", host)
    log.info("Warming up the model...")
    log.info("Ready in %.1f s of warm-up.", engine.warm_up())
    url = f"http://{host}:{port}"
    log.info("imgseg demo running at %s  (Ctrl+C to stop)", url)
    if open_browser:
        webbrowser.open(url)
    allowed = ["127.0.0.1", "localhost"] if host in ("127.0.0.1", "localhost") else ["*"]
    uvicorn.run(create_app(engine, allowed), host=host, port=port, log_level="warning")
