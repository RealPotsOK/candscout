"""HTTP dashboard, JSON API, and polling loop."""

from __future__ import annotations

import json
import math
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .bot import LivePaperBot, StepResult, now_iso
from .coinbase_exec import RealTradeService
from .config import Config
from .model_runner import LiveModel
from .scheduler import RetrainScheduler
from .store import Store


class AppContext:
    def __init__(
        self,
        cfg: Config,
        store: Store,
        bot: LivePaperBot,
        model: LiveModel,
        retrain_scheduler: RetrainScheduler,
        real_trader: RealTradeService,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.bot = bot
        self.model = model
        self.retrain_scheduler = retrain_scheduler
        self.real_trader = real_trader
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.last_step: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.poll_count = 0
        self.catchup_state: dict[str, Any] = {
            "status": "pending" if cfg.catchup_enabled else "disabled",
            "started_at": None,
            "ended_at": None,
            "result": None,
            "error": None,
        }

    def poll_once(self) -> StepResult:
        result = self.bot.step()
        with self.lock:
            self.last_step = result.to_dict()
            self.last_error = None
            self.poll_count += 1
        self.store.insert_event(result.ts, "info", f"poll action={result.action} reason={result.reason}")
        print(json.dumps({"event": "poll", **result.to_dict()}), flush=True)
        return result

    def record_error(self, message: str) -> None:
        ts = now_iso()
        with self.lock:
            self.last_error = message
        self.store.insert_event(ts, "error", message)
        print(json.dumps({"event": "poll_error", "ts": ts, "error": message}), flush=True)

    def run_catchup(self) -> bool:
        started_at = now_iso()
        with self.lock:
            self.catchup_state = {
                "status": "running",
                "started_at": started_at,
                "ended_at": None,
                "result": None,
                "error": None,
            }
        self.store.insert_event(started_at, "info", "startup catch-up started")
        try:
            result = self.bot.catch_up()
        except Exception as exc:
            ended_at = now_iso()
            message = str(exc)
            with self.lock:
                self.catchup_state = {
                    "status": "error",
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "result": None,
                    "error": message,
                }
            self.store.insert_event(ended_at, "error", f"startup catch-up failed: {message}")
            print(json.dumps({"event": "catchup_error", "ts": ended_at, "error": message}), flush=True)
            return False

        ended_at = now_iso()
        with self.lock:
            self.catchup_state = {
                "status": result.status,
                "started_at": started_at,
                "ended_at": ended_at,
                "result": result.to_dict(),
                "error": None,
            }
        self.store.insert_event(
            ended_at,
            "info",
            f"startup catch-up {result.reason}; processed={result.processed_bars}",
        )
        print(json.dumps({"event": "catchup", "ts": ended_at, **result.to_dict()}), flush=True)
        return True

    def poll_state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "poll_count": self.poll_count,
                "last_step": self.last_step,
                "last_error": self.last_error,
            }

    def catchup_status(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.catchup_state)

    def reset(self) -> None:
        self.store.reset_all(self.cfg.starting_cash, now_iso())
        with self.lock:
            self.last_step = None
            self.last_error = None
            self.poll_count = 0


def run_poll_loop(ctx: AppContext) -> None:
    if ctx.cfg.catchup_enabled:
        while not ctx.stop_event.is_set() and not ctx.run_catchup():
            print(
                json.dumps(
                    {
                        "event": "catchup_retry_wait",
                        "seconds": ctx.cfg.catchup_retry_seconds,
                    }
                ),
                flush=True,
            )
            if ctx.stop_event.wait(ctx.cfg.catchup_retry_seconds):
                return

    if ctx.cfg.poll_on_start:
        try:
            ctx.poll_once()
        except Exception as exc:  # noqa: BLE001 - poll loop should stay alive.
            ctx.record_error(str(exc))

    while not ctx.stop_event.is_set():
        wait_seconds = seconds_until_next_boundary(ctx.cfg.interval_seconds, ctx.cfg.poll_delay_seconds)
        if ctx.stop_event.wait(wait_seconds):
            break
        try:
            ctx.poll_once()
        except Exception as exc:  # noqa: BLE001 - poll loop should stay alive.
            ctx.record_error(str(exc))


def seconds_until_next_boundary(interval_seconds: int, delay_seconds: float) -> float:
    now = time.time()
    next_boundary = (math.floor(now / interval_seconds) + 1) * interval_seconds + delay_seconds
    return max(1.0, next_boundary - now)


def make_handler(ctx: AppContext) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "CryptoPredLiveSim/1.0"

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = parse_qs(parsed.query)
            try:
                if path == "/":
                    self.send_html(DASHBOARD_HTML)
                elif path == "/api/status":
                    self.send_json(api_status(ctx))
                elif path == "/api/config":
                    self.send_json(ctx.cfg.public_dict())
                elif path == "/api/account":
                    self.send_json(api_account(ctx))
                elif path == "/api/position":
                    self.send_json(ctx.store.open_position() or {})
                elif path == "/api/decisions":
                    self.send_json(ctx.store.recent_rows("model_decisions", get_limit(query, 100)))
                elif path == "/api/trades":
                    self.send_json(ctx.store.recent_rows("trades", get_limit(query, 100)))
                elif path == "/api/equity":
                    self.send_json(ctx.store.recent_rows("account_snapshots", get_limit(query, 1000)))
                elif path == "/api/events":
                    self.send_json(ctx.store.recent_rows("server_events", get_limit(query, 100)))
                elif path == "/api/retraining":
                    self.send_json(api_retraining(ctx))
                elif path == "/api/real/status":
                    refresh = query.get("refresh", ["0"])[0].lower() in {"1", "true", "yes"}
                    self.send_json(ctx.real_trader.public_status(refresh=refresh))
                elif path == "/api/real/orders":
                    self.send_json(ctx.store.recent_real_orders(get_limit(query, 100)))
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as exc:  # noqa: BLE001 - return useful API error.
                self.send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API.
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path == "/api/real/arm":
                try:
                    body = self.read_json_body()
                    self.send_json(
                        ctx.real_trader.arm(
                            token=str(body.get("token", "")),
                            confirmation=str(body.get("confirmation", "")),
                            ts=now_iso(),
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - return useful API error.
                    self.send_json({"error": str(exc)}, status=HTTPStatus.FORBIDDEN)
                return
            if path == "/api/real/disarm":
                self.send_json(ctx.real_trader.disarm(ts=now_iso(), reason="api_disarm"))
                return
            if path == "/api/real/flatten":
                try:
                    body = self.read_json_body()
                    self.send_json(
                        ctx.real_trader.flatten(
                            token=str(body.get("token", "")),
                            confirmation=str(body.get("confirmation", "")),
                            ts=now_iso(),
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - return useful API error.
                    self.send_json({"error": str(exc)}, status=HTTPStatus.FORBIDDEN)
                return
            if path != "/api/reset":
                if path == "/api/retrain-now":
                    if not ctx.cfg.allow_reset_api:
                        self.send_json(
                            {"error": "Manual retrain API is disabled. Set ALLOW_RESET_API=true to enable it."},
                            status=HTTPStatus.FORBIDDEN,
                        )
                        return
                    try:
                        ctx.retrain_scheduler.run_now_async("api")
                        self.send_json({"ok": True, "message": "Retraining started"})
                    except Exception as exc:  # noqa: BLE001 - useful API error.
                        self.send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            if not ctx.cfg.allow_reset_api:
                self.send_json(
                    {"error": "Reset API is disabled. Set ALLOW_RESET_API=true to enable it."},
                    status=HTTPStatus.FORBIDDEN,
                )
                return
            ctx.reset()
            self.send_json({"ok": True, "message": "Runtime state reset"})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}", flush=True)

        def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, default=str, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def api_status(ctx: AppContext) -> dict[str, Any]:
    poll_state = ctx.poll_state()
    catchup_state = ctx.catchup_status()
    return {
        "status": (
            "ok"
            if poll_state["last_error"] is None and catchup_state["status"] != "error"
            else "error"
        ),
        "server_time": now_iso(),
        "config": {
            "symbol": ctx.cfg.symbol,
            "interval": ctx.cfg.interval,
            "entry_threshold": ctx.cfg.entry_threshold,
            "exit_threshold": ctx.cfg.exit_threshold,
            "trade_mode": ctx.cfg.trade_mode,
            "short_entry_threshold": ctx.cfg.short_entry_threshold,
            "short_exit_threshold": ctx.cfg.short_exit_threshold,
            "fee": ctx.cfg.fee,
            "max_invest": ctx.cfg.max_invest,
            "max_short_invest": ctx.cfg.max_short_invest,
            "borrow_fee": ctx.cfg.borrow_fee,
            "allow_flip_position": ctx.cfg.allow_flip_position,
            "min_invest": ctx.cfg.min_invest,
            "retrain_frequency": ctx.cfg.retrain_frequency,
            "retrain_train_start": ctx.cfg.retrain_train_start,
            "retrain_train_end": ctx.cfg.retrain_train_end,
            "retrain_lookback_days": ctx.cfg.retrain_lookback_days,
            "train_model_type": ctx.cfg.train_model_type,
            "train_use_full_window": ctx.cfg.train_use_full_window,
            "catchup_enabled": ctx.cfg.catchup_enabled,
            "catchup_spread_pct": ctx.cfg.catchup_spread_pct,
            "catchup_max_bars": ctx.cfg.catchup_max_bars,
            "catchup_retry_seconds": ctx.cfg.catchup_retry_seconds,
            "execution_mode": ctx.cfg.execution_mode,
            "real_trading_enabled": ctx.cfg.real_trading_enabled,
            "coinbase_product_id": ctx.cfg.coinbase_product_id,
            "real_max_total_usd": ctx.cfg.real_max_total_usd,
            "real_max_order_usd": ctx.cfg.real_max_order_usd,
        },
        "model": ctx.model.info(),
        "poller": poll_state,
        "catchup": catchup_state,
        "retraining": ctx.retrain_scheduler.status_dict(),
        "real_trading": ctx.real_trader.public_status(refresh=False),
        "account": ctx.store.latest_account_snapshot(),
        "position": ctx.store.open_position(),
        "latest_decision": ctx.store.latest_decision(),
        "latest_ticker": ctx.store.latest_ticker(),
    }


def api_account(ctx: AppContext) -> dict[str, Any]:
    return {
        "state": ctx.store.account_state(),
        "latest_snapshot": ctx.store.latest_account_snapshot(),
    }


def api_retraining(ctx: AppContext) -> dict[str, Any]:
    return {
        "scheduler": ctx.retrain_scheduler.status_dict(),
        "latest_training_run": ctx.store.latest_training_run(),
        "recent_training_runs": ctx.store.recent_training_runs(20),
    }


def get_limit(query: dict[str, list[str]], default: int) -> int:
    if "limit" not in query:
        return default
    try:
        return max(1, min(int(query["limit"][0]), 5000))
    except (ValueError, IndexError):
        return default


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def serve(ctx: AppContext) -> None:
    handler = make_handler(ctx)
    poll_thread = threading.Thread(target=run_poll_loop, args=(ctx,), name="poll-loop", daemon=True)
    poll_thread.start()
    ctx.retrain_scheduler.start()
    server = ReusableThreadingHTTPServer((ctx.cfg.host, ctx.cfg.port), handler)
    print(f"Live paper-trading dashboard listening on http://{ctx.cfg.host}:{ctx.cfg.port}/", flush=True)
    try:
        server.serve_forever(poll_interval=1.0)
    finally:
        ctx.stop_event.set()
        ctx.retrain_scheduler.stop()
        server.server_close()


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CryptoPred Live Paper Trading</title>
  <style>
    :root { color-scheme: light; --bg:#f6f1e8; --ink:#16140f; --muted:#70685d; --line:#d9cdbd; --card:#fffaf1; --green:#128b4b; --red:#bc2f35; --blue:#2268c4; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Georgia, 'Times New Roman', serif; background: radial-gradient(circle at top left, #fff5cf, var(--bg) 42%, #ece0d0); color:var(--ink); }
    header { padding:22px 28px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:16px; align-items:flex-end; }
    h1 { margin:0; font-size:28px; letter-spacing:-0.03em; }
    .sub { color:var(--muted); font-size:14px; }
    main { padding:22px; display:grid; gap:18px; }
    .cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:12px; }
    .card, .panel { background:rgba(255,250,241,0.88); border:1px solid var(--line); border-radius:16px; box-shadow:0 12px 28px rgba(60,40,10,0.08); }
    .card { padding:14px; }
    .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:0.08em; }
    .value { font-size:22px; margin-top:5px; font-weight:700; }
    .grid { display:grid; grid-template-columns: 1fr 1fr; gap:18px; }
    .panel { padding:16px; min-width:0; }
    .chart-head { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:10px; }
    .panel h2 { margin:0; font-size:18px; }
    .hint { color:var(--muted); font-size:12px; margin-top:3px; }
    .chart-controls { display:flex; gap:6px; align-items:center; }
    button { border:1px solid var(--line); background:#fffdf8; color:var(--ink); border-radius:999px; padding:6px 10px; cursor:pointer; font-family:inherit; }
    button:hover { background:#f3ead9; }
    canvas { width:100%; height:260px; border-radius:12px; background:#fffdf8; border:1px solid #eadfcd; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { padding:8px 6px; border-bottom:1px solid #eadfcd; text-align:right; white-space:nowrap; }
    th:first-child, td:first-child { text-align:left; }
    .buy { color:var(--green); font-weight:700; }
    .sell { color:var(--red); font-weight:700; }
    .hold { color:var(--blue); }
    .error { color:var(--red); font-weight:700; }
    .danger { color:#fff; background:#a80f1b; border-color:#a80f1b; }
    .danger-card { border-color:#a80f1b; box-shadow:0 0 0 2px rgba(168,15,27,0.12); }
    .real-actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } header { display:block; } }
  </style>
</head>
<body>
<header>
  <div>
    <h1>CryptoPred Live Paper Trading</h1>
    <div class="sub" id="subtitle">Loading...</div>
  </div>
  <div class="sub" id="realBanner">Real trading disabled by default.</div>
</header>
<main>
  <section class="cards" id="cards"></section>
  <section class="panel" id="realPanel"></section>
  <section class="grid">
    <div class="panel">
      <div class="chart-head">
        <div><h2>Bot Money Over Time</h2><div class="hint">Wheel to zoom, drag to pan, UTC time.</div></div>
        <div class="chart-controls"><button onclick="resetChart('moneyChart')">Reset</button></div>
      </div>
      <canvas id="moneyChart" width="900" height="300"></canvas>
    </div>
    <div class="panel">
      <div class="chart-head">
        <div><h2>SOL Price Over Time</h2><div class="hint">Wheel to zoom, drag to pan, UTC time.</div></div>
        <div class="chart-controls"><button onclick="resetChart('priceChart')">Reset</button></div>
      </div>
      <canvas id="priceChart" width="900" height="300"></canvas>
    </div>
  </section>
  <section class="grid">
    <div class="panel"><h2>Recent Decisions</h2><div style="overflow:auto"><table id="decisions"></table></div></div>
    <div class="panel"><h2>Recent Trades</h2><div style="overflow:auto"><table id="trades"></table></div></div>
  </section>
</main>
<script>
const fmtUsd = n => n == null ? '-' : '$' + Number(n).toFixed(4);
const fmtNum = n => n == null ? '-' : Number(n).toFixed(6);
const fmtPct = n => n == null ? '-' : (Number(n) * 100).toFixed(3) + '%';
const shortTime = s => !s ? '-' : new Date(s).toISOString().replace('T',' ').slice(0,19);

async function getJson(path) {
  const r = await fetch(path, {cache:'no-store'});
  if (!r.ok) throw new Error(await r.text());
  return await r.json();
}

async function postJson(path, payload={}) {
  const r = await fetch(path, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(payload)
  });
  const data = await r.json();
  if (!r.ok || data.error) throw new Error(data.error || JSON.stringify(data));
  return data;
}

function card(label, value, cls='') { return `<div class="card"><div class="label">${label}</div><div class="value ${cls}">${value}</div></div>`; }

async function armRealTrading(expectedText) {
  const token = prompt('REAL_ARM_TOKEN');
  if (!token) return;
  const confirmation = prompt(`Type exactly: ${expectedText}`);
  if (!confirmation) return;
  await postJson('/api/real/arm', {token, confirmation});
  await refresh();
}

async function flattenRealPosition(expectedText) {
  const token = prompt('REAL_ARM_TOKEN');
  if (!token) return;
  const confirmation = prompt(`Type exactly: ${expectedText}`);
  if (!confirmation) return;
  await postJson('/api/real/flatten', {token, confirmation});
  await refresh();
}

async function disarmRealTrading() {
  await postJson('/api/real/disarm', {});
  await refresh();
}

const chartStates = new Map();

function timeMs(row) {
  const t = Date.parse(row.ts);
  return Number.isFinite(t) ? t : null;
}

function resetChart(canvasId) {
  const state = chartStates.get(canvasId);
  if (state) {
    state.userZoomed = false;
    state.minT = state.fullMinT;
    state.maxT = state.fullMaxT;
  }
  refresh();
}

function initChartInteractions(canvas) {
  if (canvas.dataset.zoomReady === '1') return;
  canvas.dataset.zoomReady = '1';
  canvas.addEventListener('wheel', event => {
    const state = chartStates.get(canvas.id);
    if (!state || !Number.isFinite(state.minT) || !Number.isFinite(state.maxT)) return;
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const plot = chartPlot(canvas);
    const mouseX = (event.clientX - rect.left) * (canvas.width / rect.width);
    const fraction = Math.min(1, Math.max(0, (mouseX - plot.left) / plot.width));
    const oldMin = state.minT;
    const oldMax = state.maxT;
    const oldRange = oldMax - oldMin;
    const zoomFactor = event.deltaY > 0 ? 1.25 : 0.8;
    const minRange = 60 * 1000;
    const newRange = Math.max(minRange, oldRange * zoomFactor);
    const center = oldMin + oldRange * fraction;
    state.minT = center - newRange * fraction;
    state.maxT = center + newRange * (1 - fraction);
    clampChartState(state);
    state.userZoomed = true;
    drawStoredChart(canvas.id);
  }, {passive:false});

  canvas.addEventListener('pointerdown', event => {
    const state = chartStates.get(canvas.id);
    if (!state) return;
    canvas.setPointerCapture(event.pointerId);
    state.dragging = true;
    state.dragStartX = event.clientX;
    state.dragStartMinT = state.minT;
    state.dragStartMaxT = state.maxT;
  });
  canvas.addEventListener('pointermove', event => {
    const state = chartStates.get(canvas.id);
    if (!state || !state.dragging) return;
    const rect = canvas.getBoundingClientRect();
    const plot = chartPlot(canvas);
    const dx = (event.clientX - state.dragStartX) * (canvas.width / rect.width);
    const range = state.dragStartMaxT - state.dragStartMinT;
    const dt = -dx / plot.width * range;
    state.minT = state.dragStartMinT + dt;
    state.maxT = state.dragStartMaxT + dt;
    clampChartState(state);
    state.userZoomed = true;
    drawStoredChart(canvas.id);
  });
  canvas.addEventListener('pointerup', event => {
    const state = chartStates.get(canvas.id);
    if (!state) return;
    state.dragging = false;
    canvas.releasePointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointerleave', () => {
    const state = chartStates.get(canvas.id);
    if (state) state.dragging = false;
  });
}

function clampChartState(state) {
  const fullMin = state.fullMinT;
  const fullMax = state.fullMaxT;
  const fullRange = Math.max(60 * 1000, fullMax - fullMin);
  let range = state.maxT - state.minT;
  if (range >= fullRange) {
    state.minT = fullMin;
    state.maxT = fullMax;
    return;
  }
  if (state.minT < fullMin) {
    state.minT = fullMin;
    state.maxT = fullMin + range;
  }
  if (state.maxT > fullMax) {
    state.maxT = fullMax;
    state.minT = fullMax - range;
  }
}

function chartPlot(canvas) {
  return {left: 58, right: 18, top: 28, bottom: 46, width: canvas.width - 76, height: canvas.height - 74};
}

function prepareRows(rows) {
  return rows.map(r => ({...r, _t: timeMs(r)})).filter(r => r._t != null).sort((a, b) => a._t - b._t);
}

function updateChartState(canvas, rows, series) {
  initChartInteractions(canvas);
  const cleanRows = prepareRows(rows);
  const times = cleanRows.map(r => r._t);
  let fullMinT = times.length ? Math.min(...times) : NaN;
  let fullMaxT = times.length ? Math.max(...times) : NaN;
  if (Number.isFinite(fullMinT) && fullMinT === fullMaxT) {
    fullMinT -= 30 * 60 * 1000;
    fullMaxT += 30 * 60 * 1000;
  }
  let state = chartStates.get(canvas.id);
  if (!state) {
    state = {userZoomed:false, dragging:false, minT:fullMinT, maxT:fullMaxT, fullMinT, fullMaxT, rows:cleanRows, series};
    chartStates.set(canvas.id, state);
  }
  state.rows = cleanRows;
  state.series = series;
  state.fullMinT = fullMinT;
  state.fullMaxT = fullMaxT;
  if (!state.userZoomed || !Number.isFinite(state.minT) || !Number.isFinite(state.maxT)) {
    state.minT = fullMinT;
    state.maxT = fullMaxT;
  } else {
    clampChartState(state);
  }
  return state;
}

function drawStoredChart(canvasId) {
  const canvas = document.getElementById(canvasId);
  const state = chartStates.get(canvasId);
  if (canvas && state) drawChartFromState(canvas, state);
}

function drawMultiLine(canvas, rows, series) {
  const state = updateChartState(canvas, rows, series);
  drawChartFromState(canvas, state);
}

function drawChartFromState(canvas, state) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const plot = chartPlot(canvas);
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = '#fffdf8'; ctx.fillRect(0,0,w,h);
  const rows = state.rows || [];
  const series = state.series || [];
  const vals = [];
  series.forEach(s => rows.forEach(r => {
    if (r._t < state.minT || r._t > state.maxT) return;
    const v = Number(r[s.key]);
    if (Number.isFinite(v)) vals.push(v);
  }));
  ctx.strokeStyle = '#d7c8b6'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(plot.left, plot.top); ctx.lineTo(plot.left, plot.top + plot.height); ctx.lineTo(plot.left + plot.width, plot.top + plot.height); ctx.stroke();
  if (!rows.length || !vals.length) {
    ctx.fillStyle = '#70685d'; ctx.font = '14px Georgia';
    ctx.fillText('Waiting for live snapshots...', plot.left + 10, h / 2);
    return;
  }
  let min = Math.min(...vals), max = Math.max(...vals);
  if (min === max) {
    const bump = Math.max(Math.abs(max) * 0.001, 0.01);
    min -= bump; max += bump;
  }
  ctx.fillStyle = '#70685d'; ctx.font = '12px Georgia';
  ctx.fillText(max.toFixed(4), 6, plot.top + 4); ctx.fillText(min.toFixed(4), 6, plot.top + plot.height);

  drawTimeAxis(ctx, plot, state.minT, state.maxT);
  drawHorizontalGrid(ctx, plot, min, max);

  series.forEach((s, seriesIdx) => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    rows.forEach(r => {
      if (r._t < state.minT || r._t > state.maxT) return;
      const v = Number(r[s.key]); if (!Number.isFinite(v)) return;
      const x = plot.left + plot.width * ((r._t - state.minT) / (state.maxT - state.minT));
      const y = plot.top + plot.height - plot.height * ((v - min)/(max - min));
      if (!started) { ctx.moveTo(x,y); started = true; } else ctx.lineTo(x,y);
    });
    ctx.stroke();
    ctx.fillStyle = s.color;
    ctx.fillRect(plot.left + 8 + seriesIdx * 120, 10, 12, 3);
    ctx.fillText(s.label, plot.left + 26 + seriesIdx * 120, 14);
  });
}

function drawHorizontalGrid(ctx, plot, min, max) {
  ctx.save();
  ctx.strokeStyle = '#eee3d1';
  ctx.fillStyle = '#70685d';
  ctx.font = '11px Georgia';
  for (let i = 1; i < 4; i++) {
    const y = plot.top + plot.height * i / 4;
    const v = max - (max - min) * i / 4;
    ctx.beginPath(); ctx.moveTo(plot.left, y); ctx.lineTo(plot.left + plot.width, y); ctx.stroke();
    ctx.fillText(v.toFixed(4), 6, y + 4);
  }
  ctx.restore();
}

function drawTimeAxis(ctx, plot, minT, maxT) {
  const ticks = buildTimeTicks(minT, maxT);
  const range = maxT - minT;
  ctx.save();
  ctx.strokeStyle = '#eee3d1';
  ctx.fillStyle = '#70685d';
  ctx.font = '11px Georgia';
  ticks.forEach(t => {
    const x = plot.left + plot.width * ((t - minT) / range);
    if (x < plot.left - 1 || x > plot.left + plot.width + 1) return;
    ctx.beginPath(); ctx.moveTo(x, plot.top); ctx.lineTo(x, plot.top + plot.height); ctx.stroke();
    const label = formatTimeTick(t, range);
    ctx.fillText(label, Math.min(x + 3, plot.left + plot.width - 56), plot.top + plot.height + 17);
  });
  ctx.restore();
}

function buildTimeTicks(minT, maxT) {
  const range = maxT - minT;
  const minute = 60 * 1000, hour = 60 * minute, day = 24 * hour;
  let step;
  if (range > 180 * day) step = 30 * day;
  else if (range > 45 * day) step = 14 * day;
  else if (range > 14 * day) step = 7 * day;
  else if (range > 3 * day) step = day;
  else if (range > day) step = 6 * hour;
  else if (range > 6 * hour) step = hour;
  else if (range > 2 * hour) step = 30 * minute;
  else if (range > 30 * minute) step = 10 * minute;
  else if (range > 10 * minute) step = 5 * minute;
  else step = minute;
  const start = Math.ceil(minT / step) * step;
  const ticks = [];
  for (let t = start; t <= maxT; t += step) ticks.push(t);
  return ticks.slice(0, 12);
}

function formatTimeTick(t, range) {
  const d = new Date(t);
  const month = d.toLocaleString('en-US', {month:'short', timeZone:'UTC'});
  const day = d.toLocaleString('en-US', {day:'numeric', timeZone:'UTC'});
  const hour = String(d.getUTCHours()).padStart(2, '0');
  const minute = String(d.getUTCMinutes()).padStart(2, '0');
  if (range > 45 * 24 * 60 * 60 * 1000) {
    return `${month} ${day}`;
  }
  if (range > 24 * 60 * 60 * 1000) {
    return `${month} ${day}`;
  }
  return `${hour}:${minute}`;
}

function renderTable(el, rows, columns) {
  if (!rows.length) { el.innerHTML = '<tr><td>No rows yet</td></tr>'; return; }
  el.innerHTML = '<thead><tr>' + columns.map(c => `<th>${c[0]}</th>`).join('') + '</tr></thead><tbody>' +
    rows.slice().reverse().map(r => '<tr>' + columns.map(c => `<td>${c[1](r)}</td>`).join('') + '</tr>').join('') + '</tbody>';
}

async function refresh() {
  try {
    const [status, equity, decisions, trades, realStatus, realOrders] = await Promise.all([
      getJson('/api/status'), getJson('/api/equity?limit=500'), getJson('/api/decisions?limit=50'), getJson('/api/trades?limit=50'),
      getJson('/api/real/status'), getJson('/api/real/orders?limit=25')
    ]);
    const acct = status.account || {}; const pos = status.position || {}; const tick = status.latest_ticker || {}; const dec = status.latest_decision || {}; const retrain = status.retraining || {}; const catchup = status.catchup || {};
    const invested = acct.equity != null && acct.cash != null ? Math.max(0, Number(acct.equity) - Number(acct.cash)) : null;
    document.getElementById('subtitle').textContent = `${status.config.symbol} ${status.config.interval} | model=${status.model.model_type} lookback=${status.model.lookback} | status=${status.status}`;
    const realModeText = realStatus.armed ? 'REAL TRADING ARMED' : (realStatus.enabled ? 'real trading disarmed' : 'real trading disabled');
    document.getElementById('realBanner').innerHTML = realStatus.armed ? `<span class="danger" style="padding:8px 12px;border-radius:999px">${realModeText}</span>` : realModeText;
    document.getElementById('cards').innerHTML = [
      card('Total Bot Value', fmtUsd(acct.equity)), card('Cash', fmtUsd(acct.cash)), card('Currently Invested', fmtUsd(invested)), card('SOL Held', fmtNum(acct.sol_qty)),
      card('Bid / Ask', `${fmtUsd(tick.bid)} / ${fmtUsd(tick.ask)}`), card('Prob Up', fmtPct(dec.prob_up)),
      card('Position', pos.quantity ? String(pos.side || 'long').toUpperCase() : 'CASH', pos.quantity ? 'buy' : ''), card('Last Action', dec.action || '-'),
      card('Retrain', retrain.running ? 'RUNNING' : (retrain.last_status || 'scheduled'), retrain.last_status === 'failed' ? 'error' : ''),
      card('Update Every', status.config.retrain_frequency || '-'),
      card('Train Window', status.config.retrain_train_start && status.config.retrain_train_end ? `${status.config.retrain_train_start.slice(0, 10)} → ${status.config.retrain_train_end.slice(0, 10)}` : `${status.config.retrain_lookback_days} days`),
      card('Live Train Mode', status.config.train_use_full_window ? 'full window' : 'split'),
      card('Startup Catch-up', catchup.status || '-', catchup.status === 'error' ? 'error' : ''),
      card('Replayed Bars', catchup.result ? catchup.result.processed_bars : 0),
      card('Next Retrain', shortTime(retrain.next_run_at)),
      card('Last Error', status.poller.last_error || catchup.error || 'none', status.poller.last_error || catchup.error ? 'error' : '')
    ].join('');
    const armText = `ARM REAL TRADING ${realStatus.product_id} MAX ${Number(realStatus.max_total_usd || 20).toString()}`;
    const flattenText = `FLATTEN REAL ${realStatus.product_id}`;
    document.getElementById('realPanel').className = `panel ${realStatus.armed ? 'danger-card' : ''}`;
    document.getElementById('realPanel').innerHTML = `
      <div class="chart-head">
        <div>
          <h2>Real Coinbase Spot Trading</h2>
          <div class="hint">Paper simulation remains separate. Real mode is SOL-USD spot long-only, capped at ${fmtUsd(realStatus.max_total_usd)} total and ${fmtUsd(realStatus.max_order_usd)} per order.</div>
        </div>
        <div class="hint">${realStatus.configured ? 'Coinbase configured' : 'Paper mode only'}</div>
      </div>
      <div class="cards">
        ${card('Real Enabled', realStatus.enabled ? 'YES' : 'NO', realStatus.enabled ? 'sell' : '')}
        ${card('Armed', realStatus.armed ? 'YES' : 'NO', realStatus.armed ? 'sell' : '')}
        ${card('Bot Real SOL', fmtNum(realStatus.bot_sol_qty))}
        ${card('Bot Cost Basis', fmtUsd(realStatus.bot_cost_usd))}
        ${card('Realized PnL', fmtUsd(realStatus.realized_pnl_usd), Number(realStatus.realized_pnl_usd) < 0 ? 'sell' : 'buy')}
        ${card('Real Fees', fmtUsd(realStatus.total_fees_usd))}
        ${card('Latest Real Error', realStatus.last_error || 'none', realStatus.last_error ? 'error' : '')}
      </div>
      <div class="real-actions">
        <button class="danger" onclick="armRealTrading('${armText}')">Arm Real Trading</button>
        <button onclick="disarmRealTrading()">Disarm</button>
        <button class="danger" onclick="flattenRealPosition('${flattenText}')">Flatten Bot SOL</button>
      </div>
      <div style="overflow:auto;margin-top:12px"><table id="realOrders"></table></div>
    `;
    renderTable(document.getElementById('realOrders'), realOrders, [
      ['Time', r => shortTime(r.ts)], ['Action', r => r.action], ['Status', r => r.status], ['Side', r => r.side],
      ['Req USD', r => fmtUsd(r.requested_usd)], ['Fill USD', r => fmtUsd(r.filled_usd)], ['Fill SOL', r => fmtNum(r.filled_sol)],
      ['Reason/Error', r => r.error || r.reason || '-']
    ]);
    drawMultiLine(document.getElementById('moneyChart'), equity, [
      {key:'equity', label:'total value', color:'#128b4b'},
      {key:'cash', label:'cash', color:'#2268c4'}
    ]);
    drawMultiLine(document.getElementById('priceChart'), equity, [
      {key:'last_price', label:'SOL bid price', color:'#bc2f35'}
    ]);
    renderTable(document.getElementById('decisions'), decisions, [
      ['Time', r => shortTime(r.ts)], ['Action', r => `<span class="${r.action}">${r.action}</span>`], ['Reason', r => r.reason],
      ['Prob', r => fmtPct(r.prob_up)], ['Equity', r => fmtUsd(r.equity)], ['Bid', r => fmtUsd(r.bid)], ['Ask', r => fmtUsd(r.ask)]
    ]);
    renderTable(document.getElementById('trades'), trades, [
      ['Exit', r => shortTime(r.exit_time)], ['Side', r => String(r.side || 'long').toUpperCase()], ['Reason', r => r.exit_reason], ['Invest', r => fmtUsd(r.investment)],
      ['Net PnL', r => fmtUsd(r.net_profit)], ['Return', r => fmtPct(r.gross_return)], ['Bars', r => r.bars_held]
    ]);
  } catch (e) {
    document.getElementById('subtitle').innerHTML = `<span class="error">${e.message}</span>`;
  }
}
refresh(); setInterval(refresh, 10000);
</script>
</body>
</html>"""
