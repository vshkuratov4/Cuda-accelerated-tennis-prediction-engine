"""FastAPI app factory. Mounts the JSON API under /api and serves the built
React frontend (frontend/dist) as static files on the same port, with an SPA
catch-all so client-side routing works on refresh."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router as api_router
from backend.config import FRONTEND_DIST_DIR

_MISSING_FRONTEND_HTML = """
<!doctype html><html><head><title>Tennis Predictor</title></head>
<body style="font-family: system-ui; max-width: 640px; margin: 80px auto; color: #222;">
<h1>Frontend build missing</h1>
<p>The API is running, but <code>frontend/dist</code> was not found.</p>
<p>Run <code>python run.py</code> again with Node.js/npm installed on PATH so it can
build the frontend, or run <code>npm install &amp;&amp; npm run build</code> inside the
<code>frontend/</code> directory yourself.</p>
<p>The JSON API itself is available under <a href="/api/health">/api/health</a>.</p>
</body></html>
"""


def create_app() -> FastAPI:
    app = FastAPI(title="Tennis Match Predictor")
    app.include_router(api_router)

    if FRONTEND_DIST_DIR.exists():
        assets_dir = FRONTEND_DIST_DIR / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        index_file = FRONTEND_DIST_DIR / "index.html"

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa_catch_all(full_path: str):  # noqa: ARG001
            return HTMLResponse(index_file.read_text())
    else:

        @app.get("/", include_in_schema=False)
        def missing_frontend():
            return HTMLResponse(_MISSING_FRONTEND_HTML)

    return app


app = create_app()
