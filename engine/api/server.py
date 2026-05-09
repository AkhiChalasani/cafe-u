"""
CAFE-u Engine — Main application with FastAPI.
"""

import json
import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure parent is on path for imports
_engine_dir = Path(__file__).parent.parent
if str(_engine_dir) not in sys.path:
    sys.path.insert(0, str(_engine_dir))

from engine.rules.engine import RulesEngine  # noqa: E402
from engine import __version__  # noqa: E402

logger = logging.getLogger("cafeu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the rules engine on startup."""
    rules_dir = Path(__file__).parent / "rules" / "definitions"
    app.state.engine = RulesEngine(rules_dir)
    logger.info(f"CAFE-u Engine v{__version__} initialized")
    yield


app = FastAPI(
    title="CAFE-u Engine",
    version=__version__,
    description="Adaptive UI Decision Engine — detects frustration, returns adaptations",
    lifespan=lifespan,
)

# CORS — allow any origin for self-hosted deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": __version__}


@app.post("/api/signal")
async def ingest_signal(request: Request):
    """
    Receive a signal batch from the CAFE-u Agent.
    Returns adaptation instructions if frustration is detected.
    """
    engine: RulesEngine = request.app.state.engine
    body = await request.json()
    
    signals = body.get("signals", [])
    if not signals:
        return {"adaptations": []}
    
    adaptations = engine.process(signals)
    
    return {"adaptations": adaptations}


@app.get("/api/rules")
async def list_rules(request: Request):
    """List all loaded adaptation rules."""
    engine: RulesEngine = request.app.state.engine
    return {"rules": engine.list_rules(), "ml": engine.get_ml_stats()}


@app.post("/api/rules/reload")
async def reload_rules(request: Request):
    """Reload rules from disk (no restart needed)."""
    engine: RulesEngine = request.app.state.engine
    engine.load_rules()
    return {"status": "reloaded", "count": len(engine.rules)}


# ── WebSocket (real-time) ─────────────────────────────────────────

active_connections: set[WebSocket] = set()

@app.websocket("/ws")
async def signal_websocket(websocket: WebSocket):
    """Real-time signal processing with instant adaptation response."""
    await websocket.accept()
    active_connections.add(websocket)
    engine: RulesEngine = websocket.app.state.engine
    
    try:
        while True:
            data = await websocket.receive_text()
            body = json.loads(data)
            signals = body.get("signals", [])
            
            if signals:
                adaptations = engine.process(signals)
                await websocket.send_json({"adaptations": adaptations})
    except WebSocketDisconnect:
        active_connections.discard(websocket)


# ── Dashboard ─────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CAFE-u Dashboard</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0f; color: #e0e0e0; }
  nav { background: #111118; border-bottom: 1px solid #222; padding: 16px 24px; display: flex; align-items: center; gap: 12px; }
  nav h1 { font-size: 18px; font-weight: 600; }
  nav .badge { background: #7c3aed; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; padding: 24px; }
  .card { background: #111118; border: 1px solid #222; border-radius: 12px; padding: 20px; }
  .card .value { font-size: 32px; font-weight: 700; }
  .card .label { font-size: 13px; color: #888; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 0 24px 24px; }
  .full { padding: 0 24px 24px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: #888; font-weight: 500; padding: 8px 12px; border-bottom: 1px solid #222; }
  td { padding: 8px 12px; border-bottom: 1px solid #1a1a22; font-size: 13px; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .tag.rage { background: rgba(239,68,68,0.15); color: #ef4444; }
  .tag.dead { background: rgba(245,158,11,0.15); color: #f59e0b; }
  .tag.hesitate { background: rgba(59,130,246,0.15); color: #60a5fa; }
  .tag.scroll { background: rgba(139,92,246,0.15); color: #a78bfa; }
  .tag.form { background: rgba(16,185,129,0.15); color: #34d399; }
  .tag.active { background: rgba(16,185,129,0.15); color: #34d399; }
  .stat-label { font-size: 11px; color: #6b7280; margin-bottom: 4px; }
</style>
</head>
<body>
<nav><h1>CAFE-u Engine</h1><span class="badge">v0.1.0</span><span id="status" style="color:#888;font-size:13px;">Live</span></nav>
<div class="grid" id="stats"></div>
<div class="row">
  <div class="card"><h3 style="font-size:14px;margin-bottom:12px;">🎯 Active Adaptations</h3><table><thead><tr><th>Element</th><th>Action</th><th>Signal</th></tr></thead><tbody id="adaptations-table"><tr><td colspan="3" style="text-align:center;color:#666;">No adaptations yet</td></tr></tbody></table></div>
  <div class="card"><h3 style="font-size:14px;margin-bottom:12px;">📊 Recent Signals</h3><table><thead><tr><th>Type</th><th>Element</th><th>Time</th></tr></thead><tbody id="signals-table"><tr><td colspan="3" style="text-align:center;color:#666;">No signals yet</td></tr></tbody></table></div>
</div>
<div class="full card">
  <h3 style="font-size:14px;margin-bottom:12px;">⚙️ Loaded Rules</h3>
  <table><thead><tr><th>Rule</th><th>Signal</th><th>Action</th><th>Threshold</th></tr></thead><tbody id="rules-table"></tbody></table>
</div>
<script>
const API = window.location.origin;
async function refresh() {
  try {
    const [rules] = await Promise.all([fetch(API + '/api/rules').then(r=>r.json())]);
    document.getElementById('stats').innerHTML = `
      <div class="card"><div class="stat-label">Engine Status</div><div class="value" style="color:#34d399">Active</div><div class="label">Ready to adapt</div></div>
      <div class="card"><div class="stat-label">Loaded Rules</div><div class="value">${rules.rules?.length || 0}</div><div class="label">adaptation strategies</div></div>
      <div class="card"><div class="stat-label">Mode</div><div class="value" style="color:#60a5fa">WebSocket</div><div class="label">real-time</div></div>
      <div class="card"><div class="stat-label">Version</div><div class="value">0.1</div><div class="label">CAFE-u alpha</div></div>
    `;
    if (rules.rules) {
      document.getElementById('rules-table').innerHTML = rules.rules.map(r => `
        <tr><td>${r.name}</td><td><span class="tag ${r.signal}">${r.signal}</span></td><td>${r.action}</td><td>${r.threshold || '-'}</td></tr>
      `).join('');
    }
  } catch(e) { document.getElementById('status').textContent = 'Error'; }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    run()
