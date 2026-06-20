#!/usr/bin/env python3
"""Save active/model-specific live_sim env files and retrainable snapshots."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_key_value(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save live_sim env copies and model-env snapshots.")
    parser.add_argument("--source-env", required=True, help="Generated live env file, usually live_sim/.env")
    parser.add_argument("--active-env", required=True, help="Active env used by live_sim, usually live_sim/env/active.env")
    parser.add_argument("--model-env", required=True, help="Model-specific env path under live_sim/env/models")
    parser.add_argument("--snapshot-root", required=True, help="Snapshot root under models/")
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--data-source", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", required=True)
    parser.add_argument("--sim-report", default="", help="Normal-path simulation report JSON, not live-sim")
    parser.add_argument("--env-file", action="append", default=[], help="Source env file path used by Makefile")
    parser.add_argument("--param", action="append", default=[], help="Resolved KEY=VALUE parameter")
    return parser.parse_args()


def read_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def metric_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def metric_slug(value: float | None, prefix: str, suffix: str = "") -> str:
    if value is None:
        return f"{prefix}_missing{suffix}"
    sign = "neg" if value < 0 else "pos"
    text = f"{abs(value):.2f}".replace(".", "p")
    return f"{prefix}_{sign}{text}{suffix}"


def slug(value: str, fallback: str = "value") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or fallback)).strip("-_.")
    return cleaned or fallback


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


SECRET_ENV_KEYS = {"COINBASE_API_KEY", "COINBASE_API_SECRET", "REAL_ARM_TOKEN"}


def redact_env_text(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            lines.append(line)
            continue
        key, value = line.split("=", 1)
        if key.strip() in SECRET_ENV_KEYS:
            lines.append(f"{key}=<redacted>" if value else f"{key}=")
        else:
            lines.append(line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def main() -> int:
    args = parse_args()
    source_env = Path(args.source_env)
    if not source_env.exists():
        raise FileNotFoundError(f"Source live env not found: {source_env}")

    params = parse_key_value(args.param)
    sim_report = read_json(args.sim_report)
    total_profit = metric_float(sim_report, "total_profit")
    total_return_pct = metric_float(sim_report, "total_return_pct")
    ending_cash = metric_float(sim_report, "ending_cash")
    starting_cash = metric_float(sim_report, "starting_cash")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    active_env = Path(args.active_env)
    model_env = Path(args.model_env)
    copy_file(source_env, active_env)
    live_env_text = source_env.read_text(encoding="utf-8")
    redacted_live_env_text = redact_env_text(live_env_text)
    model_env.parent.mkdir(parents=True, exist_ok=True)
    model_env.write_text(redacted_live_env_text, encoding="utf-8")

    snapshot_root = Path(args.snapshot_root)
    name = "_".join(
        [
            slug(args.model_type, "model"),
            timestamp,
            slug(args.data_source, "source"),
            slug(args.symbol, "symbol"),
            slug(args.interval, "interval"),
            metric_slug(total_profit, "profit"),
            metric_slug(total_return_pct, "return", "pct"),
        ]
    )
    snapshot_env = snapshot_root / f"{name}.env"
    snapshot_meta = snapshot_root / f"{name}.json"
    snapshot_env.parent.mkdir(parents=True, exist_ok=True)

    existing_env_files: dict[str, str] = {}
    for env_file in args.env_file:
        path = Path(env_file)
        existing_env_files[str(path)] = "exists" if path.exists() else "missing"

    header_lines = [
        "# CryptoPred live model env snapshot.",
        "# This is generated by make live-sync. It is intended to preserve enough",
        "# information to recreate a similarly shaped live retrain configuration.",
        f"# created_utc={timestamp}",
        f"# model_type={args.model_type}",
        f"# data_source={args.data_source}",
        f"# symbol={args.symbol}",
        f"# interval={args.interval}",
        f"# normal_sim_report={args.sim_report or 'missing'}",
        f"# normal_sim_starting_cash={starting_cash}",
        f"# normal_sim_ending_cash={ending_cash}",
        f"# normal_sim_total_profit={total_profit}",
        f"# normal_sim_total_return_pct={total_return_pct}",
        "#",
        "# Main Makefile parameters used to create this live env:",
    ]
    for key in sorted(params):
        header_lines.append(f"# make_param {key}={params[key]}")
    header_lines.extend(["", "# Redacted live_sim environment:"])
    snapshot_env.write_text("\n".join(header_lines) + "\n" + redacted_live_env_text, encoding="utf-8")

    metadata = {
        "created_utc": timestamp,
        "model_type": args.model_type,
        "data_source": args.data_source,
        "symbol": args.symbol,
        "interval": args.interval,
        "source_env": str(source_env),
        "active_env": str(active_env),
        "model_env": str(model_env),
        "snapshot_env": str(snapshot_env),
        "sim_report": args.sim_report,
        "sim_metrics": {
            "starting_cash": starting_cash,
            "ending_cash": ending_cash,
            "total_profit": total_profit,
            "total_return_pct": total_return_pct,
            "trade_count": sim_report.get("trade_count"),
            "profitable_trade_rate": sim_report.get("profitable_trade_rate"),
            "start_utc": sim_report.get("start_utc"),
            "end_utc": sim_report.get("end_utc"),
        },
        "make_params": params,
        "env_files": existing_env_files,
    }
    snapshot_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Active live env: {active_env}")
    print(f"Model live env:  {model_env}")
    print(f"Snapshot env:    {snapshot_env}")
    print(f"Snapshot meta:   {snapshot_meta}")
    if total_profit is None or total_return_pct is None:
        print("Simulation metrics: missing normal simulation report or missing total_profit/total_return_pct")
    else:
        print(f"Simulation metrics: profit={total_profit:.2f}, return_pct={total_return_pct:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
