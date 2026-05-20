#!/usr/bin/env python3
"""Create a standalone HTML report for bank-account simulation trades."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize simulation trades and invested activity.")
    parser.add_argument("--raw-data", required=True, help="Raw candle Parquet path")
    parser.add_argument("--trades", required=True, help="Simulation trade CSV from daily_bank_sim.py")
    parser.add_argument("--report", default=None, help="Optional simulation JSON report")
    parser.add_argument("--output", default="data/reports/sim_visualization.html", help="Output HTML path")
    parser.add_argument(
        "--activity-bucket",
        choices=["auto", "raw", "hour", "day"],
        default="auto",
        help="Bottom-panel active investment bucket size",
    )
    parser.add_argument(
        "--marker-size-basis",
        choices=["usd", "coin"],
        default="usd",
        help="Scale trade markers by investment USD or coin amount",
    )
    parser.add_argument("--title", default="Bank Simulation Visualization", help="Report title")
    parser.add_argument("--nav-home-url", default="/", help="Header home/index URL")
    parser.add_argument("--nav-model-url", default="", help="Header model visualization URL")
    parser.add_argument("--nav-sim-url", default="", help="Header simulation visualization URL")
    return parser.parse_args()


def load_candles(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"open_time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing required columns: {sorted(missing)}")
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "prob_up",
        "investment",
        "gross_return",
        "net_profit",
        "cash_after_trade",
        "account_value_after_trade",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trade CSV missing required columns: {sorted(missing)}")
    if df.empty:
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
        return df
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    numeric_columns = [
        "entry_price",
        "exit_price",
        "prob_up",
        "investment",
        "gross_return",
        "net_profit",
        "cash_after_trade",
        "account_value_after_trade",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("entry_time").reset_index(drop=True)
    df["coin_amount"] = np.where(df["entry_price"] > 0.0, df["investment"] / df["entry_price"], 0.0)
    return df


def load_report(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def infer_bar_delta(candles: pd.DataFrame) -> pd.Timedelta:
    diffs = candles["open_time"].diff().dropna()
    if diffs.empty:
        return pd.Timedelta(minutes=5)
    return pd.Timedelta(diffs.median())


def resolve_window(candles: pd.DataFrame, trades: pd.DataFrame, report: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    if report.get("start_utc") and report.get("end_utc"):
        return pd.Timestamp(report["start_utc"]).tz_convert("UTC"), pd.Timestamp(report["end_utc"]).tz_convert("UTC")

    if not trades.empty:
        pad = infer_bar_delta(candles)
        start = pd.Timestamp(trades["entry_time"].min()) - pad
        end = pd.Timestamp(trades["exit_time"].max()) + pad
        return start, end

    start = pd.Timestamp(candles["open_time"].min())
    end = pd.Timestamp(candles["open_time"].max()) + infer_bar_delta(candles)
    return start, end


def filter_candles(candles: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    out = candles[(candles["open_time"] >= start) & (candles["open_time"] <= end)].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No candles found from {start} to {end}")
    return out


def to_iso_list(series: pd.Series) -> list[str]:
    return pd.to_datetime(series, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()


def choose_bucket(candles: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested
    span = candles["open_time"].max() - candles["open_time"].min()
    if span <= pd.Timedelta(days=3):
        return "raw"
    if span <= pd.Timedelta(days=120):
        return "hour"
    return "day"


def active_investment_series(candles: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    times = candles["open_time"].copy()
    active_values = np.zeros(len(times), dtype=np.float64)

    if not trades.empty:
        time_values = times.astype("int64").to_numpy()
        for trade in trades.itertuples(index=False):
            entry = pd.Timestamp(trade.entry_time).value
            exit_ = pd.Timestamp(trade.exit_time).value
            mask = (time_values >= entry) & (time_values < exit_)
            active_values[mask] += float(trade.investment)

    return pd.DataFrame({"open_time": times, "active_investment": active_values})


def aggregate_activity(activity: pd.DataFrame, trades: pd.DataFrame, bucket: str) -> pd.DataFrame:
    if bucket == "raw":
        out = activity.copy()
        out["total_entry_amount"] = 0.0
        out["trade_count"] = 0
        if not trades.empty:
            entry_groups = trades.groupby("entry_time").agg(total_entry_amount=("investment", "sum"), trade_count=("investment", "size"))
            out = out.merge(entry_groups, how="left", left_on="open_time", right_index=True, suffixes=("", "_entry"))
            out["total_entry_amount"] = out["total_entry_amount_entry"].fillna(out["total_entry_amount"])
            out["trade_count"] = out["trade_count_entry"].fillna(out["trade_count"]).astype(int)
            out = out.drop(columns=["total_entry_amount_entry", "trade_count_entry"])
        return out

    rule = "1h" if bucket == "hour" else "1D"
    indexed = activity.set_index("open_time")
    out = indexed.resample(rule).agg(active_investment=("active_investment", "max")).reset_index()
    out["total_entry_amount"] = 0.0
    out["trade_count"] = 0

    if not trades.empty:
        trade_groups = (
            trades.set_index("entry_time")
            .resample(rule)
            .agg(total_entry_amount=("investment", "sum"), trade_count=("investment", "size"))
            .reset_index()
            .rename(columns={"entry_time": "open_time"})
        )
        out = out.merge(trade_groups, how="left", on="open_time", suffixes=("", "_trade"))
        out["total_entry_amount"] = out["total_entry_amount_trade"].fillna(out["total_entry_amount"])
        out["trade_count"] = out["trade_count_trade"].fillna(out["trade_count"]).astype(int)
        out = out.drop(columns=["total_entry_amount_trade", "trade_count_trade"])

    return out


def candle_payload(candles: pd.DataFrame) -> dict:
    return {
        "t": to_iso_list(candles["open_time"]),
        "open": candles["open"].round(8).tolist(),
        "high": candles["high"].round(8).tolist(),
        "low": candles["low"].round(8).tolist(),
        "close": candles["close"].round(8).tolist(),
    }


def trade_payload(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "entry_t": [],
            "exit_t": [],
            "entry_price": [],
            "exit_price": [],
            "investment": [],
            "coin_amount": [],
            "prob_up": [],
            "net_profit": [],
            "cash_after_trade": [],
        }
    return {
        "entry_t": to_iso_list(trades["entry_time"]),
        "exit_t": to_iso_list(trades["exit_time"]),
        "entry_price": trades["entry_price"].round(8).tolist(),
        "exit_price": trades["exit_price"].round(8).tolist(),
        "investment": trades["investment"].round(6).tolist(),
        "coin_amount": trades["coin_amount"].round(10).tolist(),
        "prob_up": trades["prob_up"].round(8).tolist(),
        "net_profit": trades["net_profit"].round(6).tolist(),
        "cash_after_trade": trades["cash_after_trade"].round(6).tolist(),
    }


def activity_payload(activity: pd.DataFrame) -> dict:
    return {
        "t": to_iso_list(activity["open_time"]),
        "active_investment": activity["active_investment"].round(6).tolist(),
        "total_entry_amount": activity["total_entry_amount"].round(6).tolist(),
        "trade_count": activity["trade_count"].astype(int).tolist(),
    }


def summarize(report: dict, trades: pd.DataFrame, activity: pd.DataFrame, marker_size_basis: str, activity_bucket: str) -> dict:
    if trades.empty:
        wins = 0
        losses = 0
        total_profit = 0.0
        max_investment = 0.0
        inferred_starting_cash = 0.0
        inferred_ending_cash = 0.0
    else:
        wins = int((trades["net_profit"] > 0.0).sum())
        losses = int((trades["net_profit"] <= 0.0).sum())
        total_profit = float(trades["net_profit"].sum())
        max_investment = float(trades["investment"].max())
        inferred_starting_cash = float(trades["cash_after_trade"].iloc[0] - trades["net_profit"].iloc[0])
        inferred_ending_cash = float(trades["cash_after_trade"].iloc[-1])

    return {
        "start_utc": str(report.get("start_utc", "")),
        "end_utc": str(report.get("end_utc", "")),
        "starting_cash": float(report.get("starting_cash", inferred_starting_cash)),
        "ending_cash": float(report.get("ending_cash", inferred_ending_cash)),
        "total_profit": float(report.get("total_profit", total_profit)),
        "trade_count": int(len(trades)),
        "winning_trades": wins,
        "losing_trades": losses,
        "max_investment": max_investment,
        "max_active_investment": float(activity["active_investment"].max()) if len(activity) else 0.0,
        "activity_bucket": activity_bucket,
        "marker_size_basis": marker_size_basis,
    }


def nav_html(home_url: str, model_url: str, sim_url: str, active: str) -> str:
    items = [("Home", home_url, "home"), ("Model", model_url, "model"), ("Simulation", sim_url, "sim")]
    links = []
    for label, url, key in items:
        if not url:
            continue
        css_class = "active" if key == active else ""
        links.append(
            f'<a class="{css_class}" href="{html_lib.escape(url, quote=True)}">'
            f"{html_lib.escape(label)}</a>"
        )
    if not links:
        return ""
    return '<nav class="site-nav">' + "".join(links) + "</nav>"


def html_template(title: str, data: dict, nav: str) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    title_text = html_lib.escape(title)
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --ink: #1b211c;
      --muted: #657064;
      --line: #d9d5c7;
      --paper: #fffdf6;
      --wash: #f3f0e6;
      --green: #147a3d;
      --red: #c33a32;
      --blue: #1f6e9a;
      --pink: #d64a88;
      --gold: #c68b24;
    }
    body {
      margin: 0;
      background: radial-gradient(circle at top left, #fff8df 0, #f4f0e7 34%, #e8ebe4 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
    }
    header {
      padding: 18px 24px 10px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 25px;
    }
    .site-nav {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 10px;
      font-family: "Courier New", monospace;
      font-size: 13px;
    }
    .site-nav a {
      color: var(--ink);
      text-decoration: none;
      border: 1px solid #beb69f;
      background: rgba(255, 255, 255, 0.72);
      padding: 6px 9px;
      border-radius: 4px;
    }
    .site-nav a:hover,
    .site-nav a.active {
      color: #fffdf6;
      background: var(--ink);
      border-color: var(--ink);
    }
    .summary {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-family: "Courier New", monospace;
      font-size: 13px;
    }
    .summary span {
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.78);
      padding: 5px 8px;
      border-radius: 4px;
    }
    main {
      padding: 14px 16px 24px;
    }
    .chart {
      position: relative;
      width: 100%;
      border: 1px solid var(--line);
      background: var(--paper);
      box-shadow: 0 10px 30px rgba(41, 36, 18, 0.08);
    }
    #price {
      height: 54vh;
      min-height: 380px;
    }
    #activity {
      height: 30vh;
      min-height: 250px;
      margin-top: 12px;
    }
    canvas {
      width: 100%;
      height: 100%;
      display: block;
      cursor: crosshair;
    }
    .hint {
      margin: 8px 2px 0;
      color: var(--muted);
      font: 12px "Courier New", monospace;
    }
    .tooltip {
      position: fixed;
      display: none;
      pointer-events: none;
      z-index: 5;
      max-width: 340px;
      background: rgba(255, 253, 246, 0.96);
      border: 1px solid #beb69f;
      color: var(--ink);
      padding: 8px 10px;
      font: 12px "Courier New", monospace;
      box-shadow: 0 8px 22px rgba(0,0,0,.14);
      white-space: pre-line;
    }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    __NAV__
    <div id="summary" class="summary"></div>
  </header>
  <main>
    <section id="price" class="chart"><canvas id="priceCanvas"></canvas></section>
    <section id="activity" class="chart"><canvas id="activityCanvas"></canvas></section>
    <div class="hint">
      Wheel to zoom, drag to pan. Top: price with buy/sell markers. Marker size is based on selected trade amount. Bottom: active dollars invested over time.
    </div>
  </main>
  <div id="tooltip" class="tooltip"></div>
  <script>
    const report = __PAYLOAD__;
    const colors = {
      ink: "#1b211c",
      muted: "#657064",
      grid: "#ece7d8",
      axis: "#837b68",
      up: "#147a3d",
      down: "#c33a32",
      buyWin: "#11944d",
      buyLoss: "#d64234",
      sellWin: "#1f6e9a",
      sellLoss: "#d64a88",
      bar: "#c68b24"
    };

    const priceCanvas = document.getElementById("priceCanvas");
    const activityCanvas = document.getElementById("activityCanvas");
    const tooltip = document.getElementById("tooltip");

    function toMs(values) { return values.map((x) => new Date(x).getTime()); }
    report.candles.t = toMs(report.candles.t);
    report.trades.entry_t = toMs(report.trades.entry_t);
    report.trades.exit_t = toMs(report.trades.exit_t);
    report.activity.t = toMs(report.activity.t);

    const fullStart = report.candles.t[0];
    const fullEnd = report.candles.t[report.candles.t.length - 1];
    let xRange = [fullStart, fullEnd];
    let drag = null;

    function formatMoney(value) {
      return "$" + Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 2 });
    }
    function formatPct(value) {
      return (100 * Number(value || 0)).toFixed(3) + "%";
    }
    function formatTime(ms) {
      return new Date(ms).toISOString().replace("T", " ").slice(0, 16) + " UTC";
    }
    function formatCoin(value) {
      return Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 6 });
    }

    function updateSummary() {
      const s = report.summary;
      document.getElementById("summary").innerHTML = [
        "trades=" + s.trade_count,
        "wins=" + s.winning_trades,
        "losses=" + s.losing_trades,
        "start_cash=" + formatMoney(s.starting_cash),
        "end_cash=" + formatMoney(s.ending_cash),
        "profit=" + formatMoney(s.total_profit),
        "max_invest=" + formatMoney(s.max_investment),
        "max_active=" + formatMoney(s.max_active_investment),
        "activity_bucket=" + s.activity_bucket,
        "marker_size=" + s.marker_size_basis
      ].map((x) => "<span>" + x + "</span>").join("");
    }

    function setupCanvas(canvas) {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * dpr));
      canvas.height = Math.max(1, Math.floor(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { ctx, width: rect.width, height: rect.height };
    }

    function plotBox(width, height) {
      return { left: 58, top: 28, width: width - 82, height: height - 72 };
    }

    function scaleX(t, plot) {
      return plot.left + ((t - xRange[0]) / (xRange[1] - xRange[0])) * plot.width;
    }
    function scaleY(v, min, max, plot) {
      if (max <= min) return plot.top + plot.height / 2;
      return plot.top + plot.height - ((v - min) / (max - min)) * plot.height;
    }

    function lowerBound(values, target) {
      let lo = 0, hi = values.length;
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (values[mid] < target) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    }
    function upperBound(values, target) {
      let lo = 0, hi = values.length;
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (values[mid] <= target) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    }
    function visibleIndexes(times) {
      const start = Math.max(0, lowerBound(times, xRange[0]) - 2);
      const end = Math.min(times.length, upperBound(times, xRange[1]) + 2);
      const out = [];
      for (let i = start; i < end; i++) out.push(i);
      return out;
    }

    function drawAxes(ctx, plot, yMin, yMax, title) {
      ctx.strokeStyle = colors.grid;
      ctx.lineWidth = 1;
      ctx.font = "12px Courier New";
      ctx.fillStyle = colors.axis;
      for (let i = 0; i <= 4; i++) {
        const y = plot.top + (plot.height * i) / 4;
        ctx.beginPath();
        ctx.moveTo(plot.left, y);
        ctx.lineTo(plot.left + plot.width, y);
        ctx.stroke();
        const value = yMax - ((yMax - yMin) * i) / 4;
        ctx.fillText(value.toFixed(2), 8, y + 4);
      }
      ctx.fillStyle = colors.ink;
      ctx.font = "14px Georgia";
      ctx.fillText(title, plot.left, 18);
    }

    function markerRadius(index) {
      const t = report.trades;
      const values = report.summary.marker_size_basis === "coin" ? t.coin_amount : t.investment;
      const maxValue = Math.max(...values, 1);
      const ratio = Math.max(0, values[index]) / maxValue;
      return 4 + Math.sqrt(ratio) * 10;
    }

    function drawPrice() {
      const { ctx, width, height } = setupCanvas(priceCanvas);
      const plot = plotBox(width, height);
      ctx.clearRect(0, 0, width, height);

      const c = report.candles;
      const indexes = visibleIndexes(c.t);
      if (!indexes.length) return;
      const lows = indexes.map((i) => c.low[i]);
      const highs = indexes.map((i) => c.high[i]);
      const yMin = Math.min(...lows);
      const yMax = Math.max(...highs);
      const pad = (yMax - yMin || yMax * 0.01 || 1) * 0.08;
      drawAxes(ctx, plot, yMin - pad, yMax + pad, "Price + simulation trades");

      const candleWidth = Math.max(1, Math.min(9, plot.width / Math.max(1, indexes.length) * 0.7));
      if (candleWidth >= 2.5) {
        for (const i of indexes) {
          const x = scaleX(c.t[i], plot);
          const yOpen = scaleY(c.open[i], yMin - pad, yMax + pad, plot);
          const yClose = scaleY(c.close[i], yMin - pad, yMax + pad, plot);
          const yHigh = scaleY(c.high[i], yMin - pad, yMax + pad, plot);
          const yLow = scaleY(c.low[i], yMin - pad, yMax + pad, plot);
          const up = c.close[i] >= c.open[i];
          ctx.strokeStyle = up ? colors.up : colors.down;
          ctx.fillStyle = up ? colors.up : colors.down;
          ctx.beginPath();
          ctx.moveTo(x, yHigh);
          ctx.lineTo(x, yLow);
          ctx.stroke();
          ctx.fillRect(x - candleWidth / 2, Math.min(yOpen, yClose), candleWidth, Math.max(1, Math.abs(yClose - yOpen)));
        }
      } else {
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        indexes.forEach((i, n) => {
          const x = scaleX(c.t[i], plot);
          const y = scaleY(c.close[i], yMin - pad, yMax + pad, plot);
          if (n === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }

      const t = report.trades;
      for (let i = 0; i < t.entry_t.length; i++) {
        if (t.exit_t[i] < xRange[0] || t.entry_t[i] > xRange[1]) continue;
        const isWin = t.net_profit[i] > 0;
        const radius = markerRadius(i);
        const entryX = scaleX(t.entry_t[i], plot);
        const entryY = scaleY(t.entry_price[i], yMin - pad, yMax + pad, plot) + radius + 3;
        ctx.fillStyle = isWin ? colors.buyWin : colors.buyLoss;
        ctx.beginPath();
        ctx.arc(entryX, entryY, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#151515";
        ctx.stroke();

        const exitX = scaleX(t.exit_t[i], plot);
        const exitY = scaleY(t.exit_price[i], yMin - pad, yMax + pad, plot) - radius - 3;
        ctx.fillStyle = isWin ? colors.sellWin : colors.sellLoss;
        ctx.beginPath();
        ctx.moveTo(exitX, exitY - radius);
        ctx.lineTo(exitX + radius, exitY + radius);
        ctx.lineTo(exitX - radius, exitY + radius);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      drawLegend(ctx, plot.left + 4, plot.top + 8);
    }

    function drawLegend(ctx, x, y) {
      const items = [
        [colors.buyWin, "buy win"],
        [colors.buyLoss, "buy loss"],
        [colors.sellWin, "sell win"],
        [colors.sellLoss, "sell loss"]
      ];
      ctx.font = "12px Courier New";
      let offset = 0;
      for (const [color, label] of items) {
        ctx.fillStyle = color;
        ctx.fillRect(x + offset, y, 10, 10);
        ctx.fillStyle = colors.ink;
        ctx.fillText(label, x + offset + 14, y + 10);
        offset += 86;
      }
    }

    function drawActivity() {
      const { ctx, width, height } = setupCanvas(activityCanvas);
      const plot = plotBox(width, height);
      ctx.clearRect(0, 0, width, height);

      const a = report.activity;
      const indexes = visibleIndexes(a.t);
      if (!indexes.length) return;
      const maxActive = Math.max(...indexes.map((i) => a.active_investment[i]), 1);
      drawAxes(ctx, plot, 0, maxActive * 1.12, "Active investment over time");

      const barWidth = Math.max(1, Math.min(18, plot.width / Math.max(1, indexes.length) * 0.8));
      ctx.fillStyle = colors.bar;
      for (const i of indexes) {
        const x = scaleX(a.t[i], plot);
        const y = scaleY(a.active_investment[i], 0, maxActive * 1.12, plot);
        ctx.fillRect(x - barWidth / 2, y, barWidth, plot.top + plot.height - y);
      }
    }

    function nearestIndex(times, target) {
      const i = lowerBound(times, target);
      if (i <= 0) return 0;
      if (i >= times.length) return times.length - 1;
      return Math.abs(times[i] - target) < Math.abs(times[i - 1] - target) ? i : i - 1;
    }

    function showTooltip(event, text) {
      tooltip.textContent = text;
      tooltip.style.display = "block";
      tooltip.style.left = Math.min(window.innerWidth - 360, event.clientX + 14) + "px";
      tooltip.style.top = Math.min(window.innerHeight - 180, event.clientY + 14) + "px";
    }
    function hideTooltip() {
      tooltip.style.display = "none";
    }

    function priceTooltip(event) {
      const rect = priceCanvas.getBoundingClientRect();
      const plot = plotBox(rect.width, rect.height);
      const x = event.clientX - rect.left;
      const tValue = xRange[0] + ((x - plot.left) / plot.width) * (xRange[1] - xRange[0]);
      const tradeIdx = nearestIndex(report.trades.entry_t, tValue);
      if (!report.trades.entry_t.length || Math.abs(report.trades.entry_t[tradeIdx] - tValue) > (xRange[1] - xRange[0]) / 50) {
        hideTooltip();
        return;
      }
      const t = report.trades;
      showTooltip(event, [
        "entry: " + formatTime(t.entry_t[tradeIdx]),
        "exit:  " + formatTime(t.exit_t[tradeIdx]),
        "investment: " + formatMoney(t.investment[tradeIdx]),
        "coin amount: " + formatCoin(t.coin_amount[tradeIdx]),
        "prob_up: " + t.prob_up[tradeIdx].toFixed(4),
        "net_profit: " + formatMoney(t.net_profit[tradeIdx]),
        "cash_after: " + formatMoney(t.cash_after_trade[tradeIdx])
      ].join("\\n"));
    }

    function activityTooltip(event) {
      const rect = activityCanvas.getBoundingClientRect();
      const plot = plotBox(rect.width, rect.height);
      const x = event.clientX - rect.left;
      const tValue = xRange[0] + ((x - plot.left) / plot.width) * (xRange[1] - xRange[0]);
      const idx = nearestIndex(report.activity.t, tValue);
      const a = report.activity;
      showTooltip(event, [
        "time: " + formatTime(a.t[idx]),
        "active invested: " + formatMoney(a.active_investment[idx]),
        "entries in bucket: " + a.trade_count[idx],
        "entry amount in bucket: " + formatMoney(a.total_entry_amount[idx])
      ].join("\\n"));
    }

    function clampRange() {
      const minSpan = 30 * 60 * 1000;
      let span = xRange[1] - xRange[0];
      if (span < minSpan) {
        const mid = (xRange[0] + xRange[1]) / 2;
        xRange = [mid - minSpan / 2, mid + minSpan / 2];
      }
      span = xRange[1] - xRange[0];
      if (xRange[0] < fullStart) xRange = [fullStart, fullStart + span];
      if (xRange[1] > fullEnd) xRange = [fullEnd - span, fullEnd];
      if (xRange[0] < fullStart) xRange[0] = fullStart;
      if (xRange[1] > fullEnd) xRange[1] = fullEnd;
    }

    function redraw() {
      drawPrice();
      drawActivity();
    }

    function attachInteraction(canvas, tooltipFn) {
      canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const ratio = event.clientX - rect.left;
        const anchor = xRange[0] + (ratio / rect.width) * (xRange[1] - xRange[0]);
        const factor = event.deltaY < 0 ? 0.75 : 1.35;
        xRange = [
          anchor - (anchor - xRange[0]) * factor,
          anchor + (xRange[1] - anchor) * factor
        ];
        clampRange();
        redraw();
      }, { passive: false });

      canvas.addEventListener("pointerdown", (event) => {
        drag = { x: event.clientX, range: [...xRange] };
        canvas.setPointerCapture(event.pointerId);
      });
      canvas.addEventListener("pointermove", (event) => {
        if (drag) {
          const rect = canvas.getBoundingClientRect();
          const dx = event.clientX - drag.x;
          const span = drag.range[1] - drag.range[0];
          const shift = -(dx / rect.width) * span;
          xRange = [drag.range[0] + shift, drag.range[1] + shift];
          clampRange();
          redraw();
        } else {
          tooltipFn(event);
        }
      });
      canvas.addEventListener("pointerup", () => { drag = null; });
      canvas.addEventListener("pointerleave", () => { drag = null; hideTooltip(); });
    }

    updateSummary();
    attachInteraction(priceCanvas, priceTooltip);
    attachInteraction(activityCanvas, activityTooltip);
    window.addEventListener("resize", redraw);
    redraw();
  </script>
</body>
</html>
"""
    return template.replace("__TITLE__", title_text).replace("__NAV__", nav).replace("__PAYLOAD__", payload)


def build_report(
    raw_data: Path,
    trades_path: Path,
    report_path: Path | None,
    output: Path,
    activity_bucket: str,
    marker_size_basis: str,
    title: str,
    nav_home_url: str,
    nav_model_url: str,
    nav_sim_url: str,
) -> None:
    candles = load_candles(raw_data)
    trades = load_trades(trades_path)
    report = load_report(report_path)
    start, end = resolve_window(candles, trades, report)
    candles_window = filter_candles(candles, start, end)
    bucket = choose_bucket(candles_window, activity_bucket)
    active_raw = active_investment_series(candles_window, trades)
    activity = aggregate_activity(active_raw, trades, bucket)

    data = {
        "candles": candle_payload(candles_window),
        "trades": trade_payload(trades),
        "activity": activity_payload(activity),
        "summary": summarize(report, trades, activity, marker_size_basis, bucket),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    nav = nav_html(nav_home_url, nav_model_url, nav_sim_url, active="sim")
    output.write_text(html_template(title, data, nav), encoding="utf-8")


def main() -> None:
    args = parse_args()
    build_report(
        raw_data=Path(args.raw_data),
        trades_path=Path(args.trades),
        report_path=Path(args.report) if args.report else None,
        output=Path(args.output),
        activity_bucket=args.activity_bucket,
        marker_size_basis=args.marker_size_basis,
        title=args.title,
        nav_home_url=args.nav_home_url,
        nav_model_url=args.nav_model_url,
        nav_sim_url=args.nav_sim_url,
    )
    print(f"Saved simulation visualization to {args.output}")


if __name__ == "__main__":
    main()
