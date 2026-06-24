#!/usr/bin/env python3
"""Create a standalone HTML report for bank-account simulation trades."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def to_utc_ns(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).astype("datetime64[ns, UTC]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize simulation trades and invested activity.")
    parser.add_argument("--raw-data", required=True, help="Raw candle Parquet path")
    parser.add_argument("--trades", required=True, help="Simulation trade CSV from daily_bank_sim.py")
    parser.add_argument("--report", default=None, help="Optional simulation JSON report")
    parser.add_argument("--comparison-trades", default=None, help="Optional second simulation trade CSV")
    parser.add_argument("--comparison-report", default=None, help="Optional second simulation JSON report")
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
    parser.add_argument(
        "--baseline-ma-windows",
        default="20,50",
        help="Comma-separated moving-average strategy windows for comparison",
    )
    parser.add_argument("--nav-home-url", default="/", help="Header home/index URL")
    parser.add_argument("--nav-model-url", default="", help="Header model visualization URL")
    parser.add_argument("--nav-sim-url", default="", help="Header simulation visualization URL")
    return parser.parse_args()


def parse_windows(value: str) -> list[int]:
    windows = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        window = int(item)
        if window < 1:
            raise ValueError("--baseline-ma-windows values must be >= 1")
        windows.append(window)
    return windows or [20]


def load_candles(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"open_time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing required columns: {sorted(missing)}")
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["open_time"] = to_utc_ns(df["open_time"])
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
        if "side" not in df.columns:
            df["side"] = pd.Series(dtype="object")
        df["entry_time"] = to_utc_ns(df["entry_time"])
        df["exit_time"] = to_utc_ns(df["exit_time"])
        return df
    df["entry_time"] = to_utc_ns(df["entry_time"])
    df["exit_time"] = to_utc_ns(df["exit_time"])
    if "side" not in df.columns:
        df["side"] = "long"
    df["side"] = df["side"].fillna("long").astype(str).str.lower()
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
    # The activity chart is meant to show "currently invested" at each candle.
    # Bucketing by hour/day can hide short one-bar positions, so auto keeps
    # raw candle resolution and the browser samples the line while drawing.
    return "raw"


def active_investment_series(candles: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    times = candles["open_time"].copy()
    active_values = np.zeros(len(times), dtype=np.float64)
    position_direction = np.zeros(len(times), dtype=np.int8)

    if not trades.empty:
        time_values = times.astype("int64").to_numpy()
        for trade in trades.itertuples(index=False):
            entry = pd.Timestamp(trade.entry_time).value
            exit_ = pd.Timestamp(trade.exit_time).value
            mask = (time_values >= entry) & (time_values < exit_)
            active_values[mask] += float(trade.investment)
            position_direction[mask] = -1 if str(getattr(trade, "side", "long")) == "short" else 1

    return pd.DataFrame(
        {"open_time": times, "active_investment": active_values, "position_direction": position_direction}
    )


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
    out = indexed.resample(rule).agg(
        active_investment=("active_investment", "max"),
        position_direction=("position_direction", "last"),
    ).reset_index()
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
        "open": candles["open"].round(6).tolist(),
        "high": candles["high"].round(6).tolist(),
        "low": candles["low"].round(6).tolist(),
        "close": candles["close"].round(6).tolist(),
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
            "side": [],
        }
    return {
        "entry_t": to_iso_list(trades["entry_time"]),
        "exit_t": to_iso_list(trades["exit_time"]),
        "entry_price": trades["entry_price"].round(6).tolist(),
        "exit_price": trades["exit_price"].round(6).tolist(),
        "investment": trades["investment"].round(6).tolist(),
        "coin_amount": trades["coin_amount"].round(8).tolist(),
        "prob_up": trades["prob_up"].round(6).tolist(),
        "net_profit": trades["net_profit"].round(6).tolist(),
        "cash_after_trade": trades["cash_after_trade"].round(6).tolist(),
        "side": trades["side"].tolist(),
    }


def activity_payload(activity: pd.DataFrame) -> dict:
    return {
        "t": to_iso_list(activity["open_time"]),
        "active_investment": activity["active_investment"].round(6).tolist(),
        "total_entry_amount": activity["total_entry_amount"].round(6).tolist(),
        "trade_count": activity["trade_count"].astype(int).tolist(),
        "position_direction": activity["position_direction"].astype(int).tolist(),
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
        "long_trade_count": int((trades["side"] == "long").sum()) if not trades.empty else 0,
        "short_trade_count": int((trades["side"] == "short").sum()) if not trades.empty else 0,
        "winning_trades": wins,
        "losing_trades": losses,
        "max_investment": max_investment,
        "max_active_investment": float(activity["active_investment"].max()) if len(activity) else 0.0,
        "activity_bucket": activity_bucket,
        "marker_size_basis": marker_size_basis,
    }


def starting_cash_from_report(report: dict, trades: pd.DataFrame) -> float:
    if "starting_cash" in report:
        return float(report["starting_cash"])
    if not trades.empty:
        return float(trades["cash_after_trade"].iloc[0] - trades["net_profit"].iloc[0])
    return 10_000.0


def fee_from_report(report: dict) -> float:
    assumptions = report.get("assumptions", {})
    if isinstance(assumptions, dict) and "fee_per_side" in assumptions:
        return float(assumptions["fee_per_side"])
    return 0.0


def model_equity_series(candles: pd.DataFrame, trades: pd.DataFrame, starting_cash: float) -> np.ndarray:
    equity = pd.Series(np.nan, index=candles["open_time"], dtype=np.float64)
    if len(equity):
        equity.iloc[0] = float(starting_cash)
    if not trades.empty:
        exit_values = (
            trades[["exit_time", "account_value_after_trade"]]
            .dropna()
            .sort_values("exit_time")
            .groupby("exit_time", as_index=True)["account_value_after_trade"]
            .last()
            .astype(float)
        )
        equity.loc[equity.index.isin(exit_values.index)] = exit_values.loc[equity.index[equity.index.isin(exit_values.index)]].to_numpy()
    return equity.ffill().to_numpy(dtype=np.float64)


def buy_hold_equity(candles: pd.DataFrame, starting_cash: float, fee: float) -> np.ndarray:
    closes = candles["close"].to_numpy(dtype=np.float64)
    if len(closes) == 0 or closes[0] <= 0.0:
        return np.full(len(closes), float(starting_cash), dtype=np.float64)
    initial_after_fee = float(starting_cash) * (1.0 - fee)
    return initial_after_fee * closes / closes[0]


def all_in_signal_strategy(
    candles: pd.DataFrame,
    starting_cash: float,
    fee: float,
    signal: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    closes = candles["close"].to_numpy(dtype=np.float64)
    signal_values = signal.fillna(False).to_numpy(dtype=bool)
    cash = float(starting_cash)
    coin = 0.0
    equity_values: list[float] = []
    active_values: list[float] = []

    for close, wants_long in zip(closes, signal_values):
        if close > 0.0 and wants_long and coin <= 0.0 and cash > 0.0:
            entry_fee = cash * fee
            coin = max(0.0, cash - entry_fee) / close
            cash = 0.0
        elif close > 0.0 and not wants_long and coin > 0.0:
            gross_exit = coin * close
            exit_fee = gross_exit * fee
            cash = max(0.0, gross_exit - exit_fee)
            coin = 0.0

        active = coin * close
        active_values.append(active)
        equity_values.append(cash + active)

    return np.asarray(equity_values, dtype=np.float64), np.asarray(active_values, dtype=np.float64)


def comparison_payload(
    candles: pd.DataFrame,
    trades: pd.DataFrame,
    activity: pd.DataFrame,
    report: dict,
    ma_windows: list[int],
    comparison_trades: pd.DataFrame | None = None,
    comparison_report: dict | None = None,
) -> dict:
    starting_cash = starting_cash_from_report(report, trades)
    fee = fee_from_report(report)
    closes = candles["close"]

    active = {
        "model": activity["active_investment"].to_numpy(dtype=np.float64),
    }
    equity = {
        "model": model_equity_series(candles, trades, starting_cash),
    }
    labels = {
        "model": "model long only",
    }
    comparison_trade_counts = {
        "total": 0,
        "long": 0,
        "short": 0,
    }
    if comparison_trades is not None:
        comparison_activity = active_investment_series(candles, comparison_trades)
        comparison_starting_cash = starting_cash_from_report(comparison_report or {}, comparison_trades)
        comparison_values = comparison_activity["active_investment"].to_numpy(dtype=np.float64)
        comparison_direction = comparison_activity["position_direction"].to_numpy(dtype=np.int8)
        active["model_long_short_long"] = np.where(
            comparison_direction > 0,
            comparison_values,
            0.0,
        )
        active["model_long_short_short"] = np.where(
            comparison_direction < 0,
            comparison_activity["active_investment"].to_numpy(dtype=np.float64),
            0.0,
        )
        equity["model_long_short"] = model_equity_series(
            candles, comparison_trades, comparison_starting_cash
        )
        labels["model_long_short"] = "model long + short"
        labels["model_long_short_long"] = "long+short LONG invested"
        labels["model_long_short_short"] = "long+short SHORT invested"
        comparison_trade_counts = {
            "total": int(len(comparison_trades)),
            "long": int((comparison_trades["side"] == "long").sum()),
            "short": int((comparison_trades["side"] == "short").sum()),
        }

    equity["buy_hold"] = buy_hold_equity(candles, starting_cash, fee)
    labels["buy_hold"] = "buy and hold"

    for window in ma_windows:
        ma = closes.rolling(window=window, min_periods=window).mean().shift(1)
        strategy_key = f"ma{window}"
        strategy_equity, _strategy_active = all_in_signal_strategy(
            candles=candles,
            starting_cash=starting_cash,
            fee=fee,
            signal=closes > ma,
        )
        equity[strategy_key] = strategy_equity
        labels[strategy_key] = f"MA{window} trend"

    momentum_signal = closes > closes.shift(1)
    momentum_equity, _momentum_active = all_in_signal_strategy(
        candles=candles,
        starting_cash=starting_cash,
        fee=fee,
        signal=momentum_signal,
    )
    equity["momentum"] = momentum_equity
    labels["momentum"] = "prev-candle momentum"

    return {
        "t": to_iso_list(candles["open_time"]),
        "active": {key: [round(float(x), 6) for x in values] for key, values in active.items()},
        "position_direction": activity["position_direction"].astype(int).tolist(),
        "equity": {key: [round(float(x), 6) for x in values] for key, values in equity.items()},
        "labels": labels,
        "comparison_trade_counts": comparison_trade_counts,
        "primary_ma_key": f"ma{ma_windows[0]}",
        "starting_cash": starting_cash,
        "fee": fee,
    }


def nav_html(home_url: str, model_url: str, sim_url: str, active: str) -> str:
    items = [
        ("Dashboard", "/", "dashboard"),
        ("Models", "/models", "models"),
        ("Compare", "/compare", "compare"),
        ("Reports", home_url, "home"),
        ("Model", model_url, "model"),
        ("Simulation", sim_url, "sim"),
        ("Live", "/live", "live"),
    ]
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
    .chart-controls {
      position: absolute;
      top: 8px;
      right: 8px;
      z-index: 4;
      display: flex;
      gap: 6px;
    }
    .chart-controls button {
      border: 1px solid var(--line);
      background: rgba(255, 253, 246, 0.94);
      color: var(--ink);
      padding: 6px 9px;
      border-radius: 4px;
      cursor: pointer;
      font: 12px "Courier New", monospace;
    }
    .chart.is-collapsed {
      height: 48px !important;
      min-height: 48px !important;
      overflow: hidden;
    }
    .chart.is-collapsed canvas { display: none; }
    .chart.is-expanded {
      position: fixed;
      inset: 12px;
      z-index: 50;
      width: auto;
      height: auto !important;
      min-height: 0 !important;
      margin: 0 !important;
      box-shadow: 0 24px 80px rgba(41, 36, 18, 0.3);
    }
    body.chart-overlay-open { overflow: hidden; }
    #price {
      height: 54vh;
      min-height: 380px;
    }
    #activity {
      height: 30vh;
      min-height: 250px;
      margin-top: 12px;
    }
    #equity {
      height: 34vh;
      min-height: 280px;
      margin-top: 12px;
    }
    #baseline {
      height: 34vh;
      min-height: 280px;
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
  <link rel="stylesheet" href="/assets/candscout.css">
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    __NAV__
    <div id="summary" class="summary"></div>
  </header>
  <main>
    <section id="price" class="chart"><div class="chart-controls"><button data-chart-action="collapse">Collapse</button><button data-chart-action="expand">Expand</button></div><canvas id="priceCanvas"></canvas></section>
    <section id="activity" class="chart"><div class="chart-controls"><button data-chart-action="collapse">Collapse</button><button data-chart-action="expand">Expand</button></div><canvas id="activityCanvas"></canvas></section>
    <section id="equity" class="chart"><div class="chart-controls"><button data-chart-action="collapse">Collapse</button><button data-chart-action="expand">Expand</button></div><canvas id="equityCanvas"></canvas></section>
    <section id="baseline" class="chart"><div class="chart-controls"><button data-chart-action="collapse">Collapse</button><button data-chart-action="expand">Expand</button></div><canvas id="baselineCanvas"></canvas></section>
    <div class="hint">
      Wheel to zoom, drag to pan. Model modes are autoscaled separately from broad market baselines so close results remain visible.
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
      shortWin: "#7b4ab3",
      shortLoss: "#e0ad16",
      shortExposure: "#b23834",
      bar: "#c68b24",
      model: "#173f6b",
      modelLongShort: "#7b4ab3",
      buyHold: "#b06b18",
      ma20: "#2d7c64",
      ma50: "#7a5224",
      momentum: "#6d5ba6"
    };

    const priceCanvas = document.getElementById("priceCanvas");
    const activityCanvas = document.getElementById("activityCanvas");
    const equityCanvas = document.getElementById("equityCanvas");
    const baselineCanvas = document.getElementById("baselineCanvas");
    const tooltip = document.getElementById("tooltip");
    let tooltipPinned = false;

    function setupChartControls() {
      for (const chart of document.querySelectorAll(".chart")) {
        const collapse = chart.querySelector('[data-chart-action="collapse"]');
        const expand = chart.querySelector('[data-chart-action="expand"]');
        collapse.addEventListener("click", (event) => {
          event.stopPropagation();
          if (chart.classList.contains("is-expanded")) {
            chart.classList.remove("is-expanded");
            document.body.classList.remove("chart-overlay-open");
            expand.textContent = "Expand";
          }
          chart.classList.toggle("is-collapsed");
          collapse.textContent = chart.classList.contains("is-collapsed") ? "Show" : "Collapse";
          scheduleRedraw();
        });
        expand.addEventListener("click", (event) => {
          event.stopPropagation();
          chart.classList.remove("is-collapsed");
          collapse.textContent = "Collapse";
          const expanded = chart.classList.toggle("is-expanded");
          document.body.classList.toggle("chart-overlay-open", expanded);
          expand.textContent = expanded ? "Unexpand" : "Expand";
          scheduleRedraw();
        });
      }
      document.addEventListener("keydown", (event) => {
        if (event.key !== "Escape") return;
        const expanded = document.querySelector(".chart.is-expanded");
        if (!expanded) return;
        expanded.classList.remove("is-expanded");
        expanded.querySelector('[data-chart-action="expand"]').textContent = "Expand";
        document.body.classList.remove("chart-overlay-open");
        scheduleRedraw();
      });
    }

    function toMs(values) { return values.map((x) => new Date(x).getTime()); }
    report.candles.t = toMs(report.candles.t);
    report.trades.entry_t = toMs(report.trades.entry_t);
    report.trades.exit_t = toMs(report.trades.exit_t);
    report.comparison.t = toMs(report.comparison.t);
    const markerSizeValues = report.summary.marker_size_basis === "coin"
      ? report.trades.coin_amount
      : report.trades.investment;
    const maxMarkerSizeValue = markerSizeValues.reduce((maxValue, value) => Math.max(maxValue, Number(value) || 0), 1);

    const fullStart = report.candles.t[0];
    const fullEnd = report.candles.t[report.candles.t.length - 1];
    let xRange = [fullStart, fullEnd];
    let drag = null;
    let redrawPending = false;

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
      const equity = report.comparison.equity;
      const finalIndex = Math.max(0, report.comparison.t.length - 1);
      const primaryMa = report.comparison.primary_ma_key;
      const shortExposure = report.comparison.active.model_long_short_short || [];
      const shortExposureRate = shortExposure.length
        ? shortExposure.filter((value) => value > 0).length / shortExposure.length
        : 0;
      document.getElementById("summary").innerHTML = [
        "long_only_trades=" + s.trade_count,
        ...(report.comparison.comparison_trade_counts ? [
          "long_short_trades=" + report.comparison.comparison_trade_counts.total,
          "long_short_longs=" + report.comparison.comparison_trade_counts.long,
          "long_short_shorts=" + report.comparison.comparison_trade_counts.short
        ] : []),
        "short_exposure_time=" + (shortExposureRate * 100).toFixed(1) + "%",
        "start_cash=" + formatMoney(s.starting_cash),
        "end_cash=" + formatMoney(s.ending_cash),
        ...(equity.model_long_short ? ["long_short_end=" + formatMoney(equity.model_long_short[finalIndex])] : []),
        "buy_hold_end=" + formatMoney(equity.buy_hold ? equity.buy_hold[finalIndex] : 0),
        primaryMa + "_end=" + formatMoney(equity[primaryMa] ? equity[primaryMa][finalIndex] : 0),
        "momentum_end=" + formatMoney(equity.momentum ? equity.momentum[finalIndex] : 0),
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

    function drawStartingCashReference(ctx, plot, yMin, yMax) {
      const startingCash = Number(report.comparison.starting_cash || report.summary.starting_cash || 0);
      if (!Number.isFinite(startingCash) || startingCash < yMin || startingCash > yMax) return;
      const y = scaleY(startingCash, yMin, yMax, plot);
      ctx.save();
      ctx.strokeStyle = "#7d5a17";
      ctx.fillStyle = "#7d5a17";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(plot.left, y);
      ctx.lineTo(plot.left + plot.width, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "11px Courier New";
      ctx.fillText("start " + formatMoney(startingCash), plot.left + plot.width - 105, y - 5);
      ctx.restore();
    }

    function markerRadius(index) {
      const ratio = Math.max(0, markerSizeValues[index] || 0) / maxMarkerSizeValue;
      return 3 + Math.sqrt(ratio) * 7;
    }

    const MAX_INDIVIDUAL_TRADE_MARKERS = 250;
    const MIN_TRADE_MARKER_SPACING_PX = 14;
    const TRADE_CLUSTER_BUCKET_PX = 34;
    const SERIES_BUCKET_PX = 6;

    function detailTarget(plot) {
      const visibleSpan = Math.max(1, xRange[1] - xRange[0]);
      const fullSpan = Math.max(visibleSpan, fullEnd - fullStart);
      const zoom = Math.max(1, fullSpan / visibleSpan);
      const zoomBoost = Math.sqrt(Math.min(64, zoom));
      const base = Math.max(160, Math.round(plot.width * 0.35));
      return Math.max(90, Math.min(5000, Math.round(base * zoomBoost)));
    }

    function decimateIndexesByMinMax(indexes, values, target) {
      if (indexes.length <= target) return indexes;
      if (target <= 3) return [indexes[0], indexes[indexes.length - 1]];
      const bucketCount = Math.max(1, Math.floor((target - 2) / 2));
      const bucketSize = (indexes.length - 2) / bucketCount;
      const out = [indexes[0]];
      for (let b = 0; b < bucketCount; b++) {
        const start = 1 + Math.floor(b * bucketSize);
        const end = Math.min(indexes.length - 1, 1 + Math.floor((b + 1) * bucketSize));
        if (start >= end) continue;
        let minIdx = indexes[start];
        let maxIdx = indexes[start];
        for (let n = start + 1; n < end; n++) {
          const idx = indexes[n];
          if ((values[idx] ?? 0) < (values[minIdx] ?? 0)) minIdx = idx;
          if ((values[idx] ?? 0) > (values[maxIdx] ?? 0)) maxIdx = idx;
        }
        if (minIdx === maxIdx) out.push(minIdx);
        else if (minIdx < maxIdx) out.push(minIdx, maxIdx);
        else out.push(maxIdx, minIdx);
      }
      const last = indexes[indexes.length - 1];
      if (out[out.length - 1] !== last) out.push(last);
      return out;
    }

    function strategyColor(key) {
      if (key === "model") return colors.model;
      if (key === "model_long_short") return colors.modelLongShort;
      if (key === "model_long_short_long") return colors.modelLongShort;
      if (key === "model_long_short_short") return colors.shortExposure;
      if (key === "buy_hold") return colors.buyHold;
      if (key === "momentum") return colors.momentum;
      if (key === report.comparison.primary_ma_key) return colors.ma20;
      if (key.startsWith("ma")) return colors.ma50;
      return colors.ink;
    }

    function strategyLabel(key) {
      return report.comparison.labels[key] || key;
    }

    function activeStrategyKeys() {
      const keys = ["model"];
      if (report.comparison.active.model_long_short_long) keys.push("model_long_short_long");
      if (report.comparison.active.model_long_short_short) keys.push("model_long_short_short");
      return keys;
    }

    function equityStrategyKeys() {
      return ["model", "model_long_short"].filter((key) => report.comparison.equity[key]);
    }

    function baselineStrategyKeys() {
      return Object.keys(report.comparison.equity);
    }

    function visibleTradeIndexes() {
      const t = report.trades;
      const indexes = [];
      for (let i = 0; i < t.entry_t.length; i++) {
        if (t.exit_t[i] < xRange[0] || t.entry_t[i] > xRange[1]) continue;
        indexes.push(i);
      }
      return indexes;
    }

    function tradeEvents(tradeIndexes, plot, yMin, yMax) {
      const t = report.trades;
      const events = [];
      for (const i of tradeIndexes) {
        const isWin = t.net_profit[i] > 0;
        const isShort = (t.side[i] || "long") === "short";
        const radius = markerRadius(i);
        const entryX = scaleX(t.entry_t[i], plot);
        const entryY = scaleY(t.entry_price[i], yMin, yMax, plot) + radius + 3;
        events.push({
          kind: isShort ? (isWin ? "short_win" : "short_loss") : (isWin ? "buy_win" : "buy_loss"),
          color: isShort ? (isWin ? colors.shortWin : colors.shortLoss) : (isWin ? colors.buyWin : colors.buyLoss),
          shape: isShort ? "triangle_down" : "triangle_up",
          x: entryX,
          y: entryY,
          radius,
          count: 1
        });

        const exitX = scaleX(t.exit_t[i], plot);
        const exitY = scaleY(t.exit_price[i], yMin, yMax, plot) - radius - 3;
        events.push({
          kind: isShort ? (isWin ? "short_cover_win" : "short_cover_loss") : (isWin ? "sell_win" : "sell_loss"),
          color: isShort ? (isWin ? colors.shortWin : colors.shortLoss) : (isWin ? colors.sellWin : colors.sellLoss),
          shape: isShort ? "triangle_up" : "triangle_down",
          x: exitX,
          y: exitY,
          radius,
          count: 1
        });
      }
      return events;
    }

    function drawTradeEvent(ctx, event) {
      ctx.fillStyle = event.color;
      ctx.strokeStyle = "#151515";
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (event.shape === "triangle_down") {
        ctx.moveTo(event.x, event.y - event.radius);
        ctx.lineTo(event.x + event.radius, event.y + event.radius);
        ctx.lineTo(event.x - event.radius, event.y + event.radius);
        ctx.closePath();
      } else if (event.shape === "triangle_up") {
        ctx.moveTo(event.x, event.y + event.radius);
        ctx.lineTo(event.x + event.radius, event.y - event.radius);
        ctx.lineTo(event.x - event.radius, event.y - event.radius);
        ctx.closePath();
      } else {
        ctx.arc(event.x, event.y, event.radius, 0, Math.PI * 2);
      }
      ctx.fill();
      ctx.stroke();
    }

    function drawClusteredTradeEvents(ctx, events, plot) {
      const clusters = new Map();
      for (const event of events) {
        if (event.x < plot.left - TRADE_CLUSTER_BUCKET_PX || event.x > plot.left + plot.width + TRADE_CLUSTER_BUCKET_PX) continue;
        const bucket = Math.floor((event.x - plot.left) / TRADE_CLUSTER_BUCKET_PX);
        const key = event.kind + ":" + bucket;
        let cluster = clusters.get(key);
        if (!cluster) {
          cluster = { x: 0, y: 0, count: 0, color: event.color };
          clusters.set(key, cluster);
        }
        cluster.x += event.x;
        cluster.y += event.y;
        cluster.count += 1;
      }

      ctx.globalAlpha = 0.82;
      for (const cluster of clusters.values()) {
        const x = cluster.x / cluster.count;
        const y = cluster.y / cluster.count;
        const radius = Math.min(14, 4 + Math.log2(cluster.count + 1) * 1.45);
        ctx.fillStyle = cluster.color;
        ctx.strokeStyle = "#151515";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      return clusters.size;
    }

    function drawTradeMarkers(ctx, plot, yMin, yMax) {
      const tradeIndexes = visibleTradeIndexes();
      if (!tradeIndexes.length) return { totalVisible: 0, drawnMarkers: 0, usedClusters: false };

      const events = tradeEvents(tradeIndexes, plot, yMin, yMax);
      const spacing = plot.width / Math.max(1, events.length);
      const shouldCluster = events.length > MAX_INDIVIDUAL_TRADE_MARKERS || spacing < MIN_TRADE_MARKER_SPACING_PX;
      if (shouldCluster) {
        return {
          totalVisible: events.length,
          drawnMarkers: drawClusteredTradeEvents(ctx, events, plot),
          usedClusters: true
        };
      }

      for (const event of events) {
        drawTradeEvent(ctx, event);
      }
      return { totalVisible: events.length, drawnMarkers: events.length, usedClusters: false };
    }

    function drawTradeMarkerNote(ctx, plot, stats) {
      if (!stats.usedClusters) return;
      ctx.save();
      ctx.fillStyle = colors.axis;
      ctx.font = "12px Courier New";
      ctx.textAlign = "right";
      ctx.textBaseline = "top";
      ctx.fillText(
        "trade markers clustered: " + stats.drawnMarkers + " dots from " + stats.totalVisible + " events",
        plot.left + plot.width,
        plot.top + 4
      );
      ctx.restore();
    }

    function drawPrice() {
      const { ctx, width, height } = setupCanvas(priceCanvas);
      const plot = plotBox(width, height);
      ctx.clearRect(0, 0, width, height);

      const c = report.candles;
      const indexes = visibleIndexes(c.t);
      if (!indexes.length) return;
      let yMin = Infinity;
      let yMax = -Infinity;
      for (const i of indexes) {
        if (c.low[i] < yMin) yMin = c.low[i];
        if (c.high[i] > yMax) yMax = c.high[i];
      }
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
        const sampled = decimateIndexesByMinMax(indexes, c.close, detailTarget(plot));
        ctx.strokeStyle = colors.ink;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        sampled.forEach((i, n) => {
          const x = scaleX(c.t[i], plot);
          const y = scaleY(c.close[i], yMin - pad, yMax + pad, plot);
          if (n === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }

      const markerStats = drawTradeMarkers(ctx, plot, yMin - pad, yMax + pad);
      drawTradeMarkerNote(ctx, plot, markerStats);

      drawLegend(ctx, [
        { color: colors.buyWin, label: "buy win", width: 86 },
        { color: colors.buyLoss, label: "buy loss", width: 86 },
        { color: colors.sellWin, label: "sell win", width: 86 },
        { color: colors.sellLoss, label: "sell loss", width: 92 },
        { color: colors.shortWin, label: "short win", width: 92 },
        { color: colors.shortLoss, label: "short loss", width: 100 }
      ], plot.left + 4, plot.top + 8, plot.width - 8);
    }

    function drawLegend(ctx, items, x, y, maxWidth) {
      ctx.font = "12px Courier New";
      ctx.textBaseline = "top";
      let offset = 0;
      let row = 0;
      const rowHeight = 17;
      for (const item of items) {
        const itemWidth = item.width || Math.max(76, ctx.measureText(item.label).width + 24);
        if (offset > 0 && offset + itemWidth > maxWidth) {
          offset = 0;
          row += 1;
        }
        const itemX = x + offset;
        const itemY = y + row * rowHeight;
        ctx.fillStyle = item.color;
        ctx.fillRect(itemX, itemY, 10, 10);
        ctx.fillStyle = colors.ink;
        ctx.fillText(item.label, itemX + 14, itemY - 1);
        offset += itemWidth;
      }
    }

    function drawSeriesLegend(ctx, items, x, y, maxWidth) {
      drawLegend(ctx, items, x, y - 5, maxWidth);
    }

    function yRangeForSeries(seriesByKey, keys, indexes, zeroMin) {
      let yMin = zeroMin ? 0 : Infinity;
      let yMax = zeroMin ? 1 : -Infinity;
      for (const key of keys) {
        const values = seriesByKey[key];
        if (!values) continue;
        for (const i of indexes) {
          const value = values[i];
          if (!Number.isFinite(value)) continue;
          if (value < yMin) yMin = value;
          if (value > yMax) yMax = value;
        }
      }
      if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
        yMin = 0;
        yMax = 1;
      }
      const pad = (yMax - yMin) * 0.08 || 1;
      return { yMin: yMin - (yMin < 0 ? pad : 0), yMax: yMax + pad };
    }

    function bucketLinePoints(times, values, indexes, plot, yMin, yMax, mode) {
      const buckets = new Map();
      for (const i of indexes) {
        const x = scaleX(times[i], plot);
        if (x < plot.left - SERIES_BUCKET_PX || x > plot.left + plot.width + SERIES_BUCKET_PX) continue;
        const key = Math.floor((x - plot.left) / SERIES_BUCKET_PX);
        let bucket = buckets.get(key);
        if (!bucket) {
          bucket = { x: 0, count: 0, value: values[i] };
          buckets.set(key, bucket);
        }
        bucket.x += x;
        bucket.count += 1;
        if (mode === "max") {
          bucket.value = Math.max(bucket.value, values[i] || 0);
        } else if (mode === "min") {
          bucket.value = Math.min(bucket.value, values[i] || 0);
        } else {
          bucket.value = values[i];
        }
      }
      return Array.from(buckets.entries())
        .sort((left, right) => left[0] - right[0])
        .map((entry) => ({
          x: entry[1].x / entry[1].count,
          y: scaleY(entry[1].value, yMin, yMax, plot),
          value: entry[1].value
        }));
    }

    function drawSeriesLine(ctx, times, values, indexes, plot, yMin, yMax, color, mode, dash = [], width = 2, stepped = false) {
      const points = bucketLinePoints(times, values, indexes, plot, yMin, yMax, mode);
      if (!points.length) return points;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.beginPath();
      for (let n = 0; n < points.length; n++) {
        const point = points[n];
        if (n === 0) ctx.moveTo(point.x, point.y);
        else if (stepped) {
          ctx.lineTo(point.x, points[n - 1].y);
          ctx.lineTo(point.x, point.y);
        } else ctx.lineTo(point.x, point.y);
      }
      ctx.stroke();
      ctx.restore();
      return points;
    }

    function drawActivity() {
      const { ctx, width, height } = setupCanvas(activityCanvas);
      const plot = plotBox(width, height);
      ctx.clearRect(0, 0, width, height);

      const comparison = report.comparison;
      const times = comparison.t;
      const indexes = visibleIndexes(times);
      if (!indexes.length) return;
      const keys = activeStrategyKeys();
      const { yMin, yMax } = yRangeForSeries(comparison.active, keys, indexes, true);
      drawAxes(ctx, plot, yMin, yMax, "Active invested dollars by position side");

      const drawKeys = [...keys.filter((key) => key !== "model"), ...keys.filter((key) => key === "model")];
      for (const key of drawKeys) {
        drawSeriesLine(ctx, times, comparison.active[key], indexes, plot, yMin, yMax, strategyColor(key), "max", key === "model" ? [8, 4] : [], key === "model" ? 2.7 : 2.2, true);
      }
      drawSeriesLegend(ctx, keys.map((key) => ({
        color: strategyColor(key),
        label: strategyLabel(key) + (key === "model" ? " (dashed)" : ""),
        width: key === "model" ? 150 : 120
      })), plot.left + 4, plot.top + 8, plot.width - 8);
    }

    function drawEquity() {
      const { ctx, width, height } = setupCanvas(equityCanvas);
      const plot = plotBox(width, height);
      ctx.clearRect(0, 0, width, height);

      const comparison = report.comparison;
      const times = comparison.t;
      const indexes = visibleIndexes(times);
      if (!indexes.length) return;
      const keys = equityStrategyKeys();
      const { yMin, yMax } = yRangeForSeries(comparison.equity, keys, indexes, false);
      const finalIndex = comparison.t.length - 1;
      const finalGap = (comparison.equity.model_long_short?.[finalIndex] || 0) - (comparison.equity.model?.[finalIndex] || 0);
      const startingCash = Number(comparison.starting_cash || report.summary.starting_cash || 0);
      const longProfit = (comparison.equity.model?.[finalIndex] || 0) - startingCash;
      const longShortProfit = (comparison.equity.model_long_short?.[finalIndex] || 0) - startingCash;
      drawAxes(ctx, plot, yMin, yMax, "Model equity; final P/L long " + formatSignedMoney(longProfit) + ", long+short " + formatSignedMoney(longShortProfit));
      drawStartingCashReference(ctx, plot, yMin, yMax);

      const drawKeys = [...keys.filter((key) => key !== "model"), ...keys.filter((key) => key === "model")];
      for (const key of drawKeys) {
        drawSeriesLine(ctx, times, comparison.equity[key], indexes, plot, yMin, yMax, strategyColor(key), "last", key === "model" ? [8, 4] : [], key === "model" ? 2.9 : 2);
      }
      drawSeriesLegend(ctx, keys.map((key) => ({
        color: strategyColor(key),
        label: strategyLabel(key) + (key === "model" ? " (dashed)" : ""),
        width: key === "buy_hold" ? 125 : 145
      })), plot.left + 4, plot.top + 8, plot.width - 8);
    }

    function drawBaseline() {
      const { ctx, width, height } = setupCanvas(baselineCanvas);
      const plot = plotBox(width, height);
      ctx.clearRect(0, 0, width, height);

      const comparison = report.comparison;
      const times = comparison.t;
      const indexes = visibleIndexes(times);
      if (!indexes.length) return;
      const keys = baselineStrategyKeys();
      const { yMin, yMax } = yRangeForSeries(comparison.equity, keys, indexes, false);
      drawAxes(ctx, plot, yMin, yMax, "Models vs buy-and-hold and simple strategies");
      drawStartingCashReference(ctx, plot, yMin, yMax);

      const drawKeys = [...keys.filter((key) => key !== "model"), ...keys.filter((key) => key === "model")];
      for (const key of drawKeys) {
        drawSeriesLine(ctx, times, comparison.equity[key], indexes, plot, yMin, yMax, strategyColor(key), "last", key === "model" ? [8, 4] : [], key === "model" ? 2.9 : 2);
      }
      drawSeriesLegend(ctx, keys.map((key) => ({
        color: strategyColor(key),
        label: strategyLabel(key) + (key === "model" ? " (dashed)" : ""),
        width: key === "buy_hold" ? 125 : 145
      })), plot.left + 4, plot.top + 8, plot.width - 8);
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
    function formatSignedMoney(value) {
      const number = Number(value || 0);
      return (number >= 0 ? "+" : "-") + formatMoney(Math.abs(number));
    }

    function priceTooltip(event) {
      const rect = priceCanvas.getBoundingClientRect();
      const plot = plotBox(rect.width, rect.height);
      const x = event.clientX - rect.left;
      const tValue = xRange[0] + ((x - plot.left) / plot.width) * (xRange[1] - xRange[0]);
      const candleIdx = nearestIndex(report.candles.t, tValue);
      const c = report.candles;
      const rows = [
        "time: " + formatTime(c.t[candleIdx]),
        "open: " + formatCoin(c.open[candleIdx]),
        "high: " + formatCoin(c.high[candleIdx]),
        "low: " + formatCoin(c.low[candleIdx]),
        "close: " + formatCoin(c.close[candleIdx])
      ];
      const tradeIdx = nearestIndex(report.trades.entry_t, tValue);
      if (report.trades.entry_t.length && Math.abs(report.trades.entry_t[tradeIdx] - tValue) <= (xRange[1] - xRange[0]) / 50) {
        const t = report.trades;
        rows.push(
          "trade side: " + String(t.side[tradeIdx] || "long").toUpperCase(),
          "trade entry: " + formatTime(t.entry_t[tradeIdx]),
          "trade exit:  " + formatTime(t.exit_t[tradeIdx]),
          "investment: " + formatMoney(t.investment[tradeIdx]),
          "prob_up: " + t.prob_up[tradeIdx].toFixed(4),
          "net_profit: " + formatMoney(t.net_profit[tradeIdx]),
          "cash_after: " + formatMoney(t.cash_after_trade[tradeIdx])
        );
      }
      showTooltip(event, rows.join("\\n"));
    }

    function activityTooltip(event) {
      const rect = activityCanvas.getBoundingClientRect();
      const plot = plotBox(rect.width, rect.height);
      const x = event.clientX - rect.left;
      const tValue = xRange[0] + ((x - plot.left) / plot.width) * (xRange[1] - xRange[0]);
      const idx = nearestIndex(report.comparison.t, tValue);
      const rows = [
        "time: " + formatTime(report.comparison.t[idx]),
        "position: " + (report.comparison.position_direction[idx] > 0 ? "LONG" : report.comparison.position_direction[idx] < 0 ? "SHORT" : "CASH"),
      ];
      const shortExposure = report.comparison.active.model_long_short_short?.[idx] || 0;
      const longExposure = report.comparison.active.model_long_short_long?.[idx] || 0;
      rows.push(
        "long+short side: " +
        (shortExposure > 0 ? "SHORT" : longExposure > 0 ? "LONG" : "CASH")
      );
      for (const key of activeStrategyKeys()) {
        rows.push(strategyLabel(key) + ": " + formatMoney(report.comparison.active[key][idx]));
      }
      showTooltip(event, rows.join("\\n"));
    }

    function equityTooltip(event) {
      seriesTooltip(event, equityCanvas, equityStrategyKeys());
    }

    function baselineTooltip(event) {
      seriesTooltip(event, baselineCanvas, baselineStrategyKeys());
    }

    function seriesTooltip(event, canvas, keys) {
      const rect = canvas.getBoundingClientRect();
      const plot = plotBox(rect.width, rect.height);
      const x = event.clientX - rect.left;
      const tValue = xRange[0] + ((x - plot.left) / plot.width) * (xRange[1] - xRange[0]);
      const idx = nearestIndex(report.comparison.t, tValue);
      const rows = [
        "time: " + formatTime(report.comparison.t[idx]),
      ];
      for (const key of keys) {
        const value = report.comparison.equity[key][idx];
        rows.push(strategyLabel(key) + ": " + formatMoney(value));
        if (key === "model" || key === "model_long_short") {
          rows.push("  P/L: " + formatSignedMoney(value - Number(report.comparison.starting_cash || report.summary.starting_cash || 0)));
        }
      }
      showTooltip(event, rows.join("\\n"));
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
      drawEquity();
      drawBaseline();
    }

    function scheduleRedraw() {
      if (redrawPending) return;
      redrawPending = true;
      requestAnimationFrame(() => {
        redrawPending = false;
        redraw();
      });
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
        scheduleRedraw();
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
          scheduleRedraw();
        } else if (!tooltipPinned) {
          tooltipFn(event);
        }
      });
      canvas.addEventListener("click", (event) => {
        tooltipPinned = true;
        tooltipFn(event);
      });
      canvas.addEventListener("pointerup", () => { drag = null; });
      canvas.addEventListener("pointerleave", () => {
        drag = null;
        if (!tooltipPinned) hideTooltip();
      });
    }

    updateSummary();
    attachInteraction(priceCanvas, priceTooltip);
    attachInteraction(activityCanvas, activityTooltip);
    attachInteraction(equityCanvas, equityTooltip);
    attachInteraction(baselineCanvas, baselineTooltip);
    window.addEventListener("resize", scheduleRedraw);
    setupChartControls();
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        tooltipPinned = false;
        hideTooltip();
      }
    });
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
    baseline_ma_windows: list[int],
    title: str,
    nav_home_url: str,
    nav_model_url: str,
    nav_sim_url: str,
    comparison_trades_path: Path | None = None,
    comparison_report_path: Path | None = None,
) -> None:
    candles = load_candles(raw_data)
    trades = load_trades(trades_path)
    report = load_report(report_path)
    comparison_trades = (
        load_trades(comparison_trades_path)
        if comparison_trades_path is not None and comparison_trades_path.exists()
        else None
    )
    comparison_report = load_report(comparison_report_path)
    start, end = resolve_window(candles, trades, report)
    candles_window = filter_candles(candles, start, end)
    bucket = choose_bucket(candles_window, activity_bucket)
    active_raw = active_investment_series(candles_window, trades)
    activity = aggregate_activity(active_raw, trades, bucket)

    data = {
        "candles": candle_payload(candles_window),
        "trades": trade_payload(trades),
        "comparison": comparison_payload(
            candles=candles_window,
            trades=trades,
            activity=activity,
            report=report,
            ma_windows=baseline_ma_windows,
            comparison_trades=comparison_trades,
            comparison_report=comparison_report,
        ),
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
        baseline_ma_windows=parse_windows(args.baseline_ma_windows),
        title=args.title,
        nav_home_url=args.nav_home_url,
        nav_model_url=args.nav_model_url,
        nav_sim_url=args.nav_sim_url,
        comparison_trades_path=Path(args.comparison_trades) if args.comparison_trades else None,
        comparison_report_path=Path(args.comparison_report) if args.comparison_report else None,
    )
    print(f"Saved simulation visualization to {args.output}")


if __name__ == "__main__":
    main()
