#!/usr/bin/env python3
"""Local CryptoPred dashboard with report serving, job control, and live-sim integration."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import mimetypes
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen

import pandas as pd

from model_catalog import (
    is_supported_model_type,
    model_complexity_group,
    model_complexity_rank,
    model_display_label,
    model_sort_key,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = ROOT / "src" / "model_registry"
REPORT_STYLE_PATH = ROOT / "src" / "report_style.css"
SAFE_VAR_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
EPOCH_RE = re.compile(r"epoch=(\d+)/(\d+)")
STRATEGY_WINDOW_RE = re.compile(r"strategy_window=(\d+)/(\d+)")
MAX_INVEST_NUMERIC_RE = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)$")
MAX_INVEST_COEFF_M_RE = re.compile(r"^((?:\d+(?:\.\d*)?|\.\d+))\s*\*?\s*m$")
MAX_INVEST_M_DIV_RE = re.compile(r"^m\s*/\s*((?:\d+(?:\.\d*)?|\.\d+))$")
REPORT_PAYLOAD_CACHE: dict[tuple[str, int], dict[str, Any]] = {}

COMMON_ALLOWED_VARS = {
    "ASSET_ENV",
    "TRAINER_ENV",
    "XGB_TRAINER_ENV",
    "DATA_SOURCE",
    "SYMBOL",
    "RANDOM_STOCK",
    "STOCK_LIST",
    "INTERVAL",
    "START",
    "END",
    "SPLIT",
    "EDGE",
    "SHORT_EDGE",
    "FEE",
    "THRESHOLD",
    "DECISION_THRESHOLD",
    "THRESHOLD_GRID",
    "OPTIMIZE_METRIC",
    "POSITION_MODE",
    "EXIT_THRESHOLD",
    "TRADE_MODE",
    "SHORT_ENTRY_THRESHOLD",
    "SHORT_EXIT_THRESHOLD",
    "ALLOW_FLIP_POSITION",
    "BORROW_FEE",
    "LEVERAGE",
    "LIQUIDATION_SIMULATION",
    "MAX_HOLD_BARS",
    "STOP_LOSS",
    "TAKE_PROFIT",
    "SIM_START",
    "SIM_DURATION",
    "SIM_DEFAULT_TEST_FRACTION",
    "SIM_STARTING_CASH",
    "SIM_MIN_INVEST",
    "SIM_MAX_INVEST",
    "SIM_MAX_SHORT_INVEST",
    "SIM_CONFIDENCE_MULTIPLIER",
    "SIM_SHORT_CONFIDENCE_MULTIPLIER",
    "SIM_POSITION_MODE",
    "SIM_SLIPPAGE",
    "SIM_SPREAD_PCT",
    "AUTO_SAVE_RUN",
    "STRATEGY_MODEL_TYPE",
    "STRATEGY_MA_WINDOW",
}

SHARED_SETTING_VARS = {
    "ASSET_ENV",
    "DATA_SOURCE",
    "SYMBOL",
    "RANDOM_STOCK",
    "STOCK_LIST",
    "INTERVAL",
    "START",
    "END",
    "SPLIT",
    "EDGE",
    "SHORT_EDGE",
    "FEE",
    "THRESHOLD",
    "DECISION_THRESHOLD",
    "THRESHOLD_GRID",
    "OPTIMIZE_METRIC",
    "POSITION_MODE",
    "EXIT_THRESHOLD",
    "TRADE_MODE",
    "SHORT_ENTRY_THRESHOLD",
    "SHORT_EXIT_THRESHOLD",
    "ALLOW_FLIP_POSITION",
    "BORROW_FEE",
    "LEVERAGE",
    "LIQUIDATION_SIMULATION",
    "MAX_HOLD_BARS",
    "STOP_LOSS",
    "TAKE_PROFIT",
    "SIM_START",
    "SIM_DURATION",
    "SIM_DEFAULT_TEST_FRACTION",
    "SIM_STARTING_CASH",
    "SIM_MIN_INVEST",
    "SIM_MAX_INVEST",
    "SIM_MAX_SHORT_INVEST",
    "SIM_CONFIDENCE_MULTIPLIER",
    "SIM_SHORT_CONFIDENCE_MULTIPLIER",
    "SIM_POSITION_MODE",
    "SIM_SLIPPAGE",
    "SIM_SPREAD_PCT",
    "AUTO_SAVE_RUN",
}

LIVE_ACTION_TARGETS = {
    "sync": ["live-sync"],
    "start": ["live-up-sync"],
    "stop": ["live-down"],
    "update_model": ["live-update-model-sync"],
    "retrain_now": ["live-retrain-now"],
    "retrain_status": ["live-retrain-status"],
}

LIVE_ALLOWED_VARS = {
    "ASSET_ENV",
    "TRAINER_ENV",
    "DATA_SOURCE",
    "SYMBOL",
    "INTERVAL",
    "START",
    "END",
    "SPLIT",
    "EDGE",
    "SHORT_EDGE",
    "FEE",
    "THRESHOLD",
    "EXIT_THRESHOLD",
    "TRADE_MODE",
    "SHORT_ENTRY_THRESHOLD",
    "SHORT_EXIT_THRESHOLD",
    "ALLOW_FLIP_POSITION",
    "BORROW_FEE",
    "LEVERAGE",
    "LIQUIDATION_SIMULATION",
    "SIM_MAX_SHORT_INVEST",
    "LIVE_MAX_SHORT_INVEST",
    "STOP_LOSS",
    "TAKE_PROFIT",
    "MAX_HOLD_BARS",
    "NN_BACKEND",
    "NN_DEVICE",
    "NN_MODEL_TYPE",
    "NN_LOOKBACK",
    "NN_SEQUENCE_FEATURE_SET",
    "NN_CNN_FILTERS",
    "NN_CNN_KERNEL_SIZES",
    "NN_LSTM_HIDDEN_SIZE",
    "NN_LSTM_LAYERS",
    "NN_LSTM_DROPOUT",
    "NN_GRU_HIDDEN_SIZE",
    "NN_GRU_LAYERS",
    "NN_GRU_DROPOUT",
    "NN_TRANSFORMER_D_MODEL",
    "NN_TRANSFORMER_HEADS",
    "NN_TRANSFORMER_LAYERS",
    "NN_TRANSFORMER_FF_DIM",
    "NN_TRANSFORMER_DROPOUT",
    "NN_HIDDEN_LAYERS",
    "NN_LR",
    "NN_EPOCHS",
    "NN_BATCH_SIZE",
    "NN_L2",
    "NN_CLASS_WEIGHT_MODE",
    "NN_SEED",
    "LIVE_MODEL_TYPE",
    "LIVE_MODEL_SOURCE",
    "LIVE_RETRAIN_FREQUENCY",
    "LIVE_RETRAIN_TRAIN_START",
    "LIVE_RETRAIN_TRAIN_END",
    "LIVE_RETRAIN_LOOKBACK_DAYS",
    "LIVE_TRAIN_MODEL_TYPE",
    "LIVE_TRAIN_BACKEND",
    "LIVE_TRAIN_DEVICE",
    "LIVE_TRAIN_LOOKBACK",
    "LIVE_TRAIN_SEQUENCE_FEATURE_SET",
    "LIVE_TRAIN_EDGE",
    "LIVE_TRAIN_USE_FULL_WINDOW",
    "LIVE_STARTING_CASH",
    "LIVE_MAX_INVEST",
    "LIVE_MIN_INVEST",
    "LIVE_CONFIDENCE_MULTIPLIER",
    "LIVE_SLIPPAGE",
    "LIVE_HOST_PORT",
}


@dataclass
class Job:
    id: str
    label: str
    command: list[str]
    cwd: str
    status: str = "queued"
    progress: float = 0.0
    returncode: int | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    logs: list[str] = field(default_factory=list)
    links: dict[str, str] = field(default_factory=dict)
    auto_graphs: dict[str, str] = field(default_factory=dict)
    model_id: str = ""
    action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "command": self.command,
            "cwd": self.cwd,
            "status": self.status,
            "progress": self.progress,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "logs": self.logs[-500:],
            "links": self.links,
            "auto_graphs": self.auto_graphs,
            "model_id": self.model_id,
            "action": self.action,
        }


class JobRunner:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, Job] = {}

    def active_job(self) -> Job | None:
        with self.lock:
            for job in self.jobs.values():
                if job.status in {"queued", "running"}:
                    return job
        return None

    def start(
        self,
        label: str,
        command: list[str],
        cwd: Path,
        links: dict[str, str] | None = None,
        auto_graphs: dict[str, str] | None = None,
        model_id: str = "",
        action: str = "",
    ) -> Job:
        with self.lock:
            active = self.active_job()
            if active is not None:
                raise RuntimeError(f"Job already running: {active.label} ({active.id})")
            job = Job(
                id=uuid.uuid4().hex[:12],
                label=label,
                command=command,
                cwd=str(cwd),
                links=links or {},
                auto_graphs=auto_graphs or {},
                model_id=model_id,
                action=action,
            )
            self.jobs[job.id] = job
        thread = threading.Thread(target=self._run, args=(job,), name=f"job-{job.id}", daemon=True)
        thread.start()
        return job

    def _append(self, job: Job, line: str) -> None:
        line = line.rstrip("\n")
        with self.lock:
            job.logs.append(line)
            if len(job.logs) > 1200:
                job.logs = job.logs[-1200:]
            lower = line.lower()
            match = EPOCH_RE.search(line)
            if match:
                current = int(match.group(1))
                total = max(1, int(match.group(2)))
                job.progress = max(job.progress, min(0.82, 0.18 + 0.64 * current / total))
            else:
                match = STRATEGY_WINDOW_RE.search(line)
                if match:
                    current = int(match.group(1))
                    total = max(1, int(match.group(2)))
                    job.progress = max(job.progress, min(0.82, 0.18 + 0.64 * current / total))
            if match:
                return
            elif "src/download.py" in lower:
                job.progress = max(job.progress, 0.06)
            elif "saved " in lower and " candles" in lower:
                job.progress = max(job.progress, 0.12)
            elif (
                "src/train_sequence_nn.py" in lower
                or "src/train.py" in lower
                or "train_xgboost.py" in lower
                or "train_strategy.py" in lower
            ):
                job.progress = max(job.progress, 0.16)
            elif "torch_device=" in lower:
                job.progress = max(job.progress, 0.2)
            elif "experiment complete" in lower:
                job.progress = max(job.progress, 0.84)
            elif (
                "src/backtest_sequence_nn.py" in lower
                or "src/backtest.py" in lower
                or "backtest_xgboost.py" in lower
                or "backtest_strategy.py" in lower
            ):
                job.progress = max(job.progress, 0.86)
            elif "src/visualize.py" in lower:
                job.progress = max(job.progress, 0.9)
            elif "src/daily_bank_sim.py" in lower or "sim_strategy.py" in lower:
                job.progress = max(job.progress, 0.92)
            elif "src/visualize_sim.py" in lower:
                job.progress = max(job.progress, 0.95)
            elif "reports index:" in lower or "saved report index" in lower:
                job.progress = max(job.progress, 0.98)
            elif "visualization:" in lower:
                job.progress = max(job.progress, 0.86)

    def _run(self, job: Job) -> None:
        with self.lock:
            job.status = "running"
            job.progress = 0.03
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            process = subprocess.Popen(
                job.command,
                cwd=job.cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                self._append(job, line)
            returncode = process.wait()
            with self.lock:
                job.returncode = returncode
                job.status = "completed" if returncode == 0 else "failed"
                job.progress = 1.0 if returncode == 0 else max(job.progress, 0.95)
                job.ended_at = time.time()
        except Exception as exc:  # noqa: BLE001 - surfaced through dashboard.
            with self.lock:
                job.returncode = -1
                job.status = "failed"
                job.progress = max(job.progress, 0.95)
                job.ended_at = time.time()
                job.logs.append(f"ERROR: {exc}")

    def list_jobs(self) -> list[dict[str, Any]]:
        with self.lock:
            return [job.to_dict() for job in sorted(self.jobs.values(), key=lambda item: item.started_at, reverse=True)]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return job.to_dict() if job else None


class DashboardContext:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = Path(args.root).resolve()
        self.reports_root = Path(args.reports_root).resolve()
        self.settings_path = Path(args.settings_file).resolve()
        self.host = args.host
        self.port = args.port
        self.live_url = args.live_url.rstrip("/")
        self.live_public_url = args.live_public_url.rstrip("/")
        self.runner = JobRunner()
        self.registry = load_registry()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CryptoPred local dashboard")
    parser.add_argument("--host", default="192.168.2.197")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reports-root", default="data/reports")
    parser.add_argument("--settings-file", default="data/reports/dashboard_settings.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--live-url", default="http://127.0.0.1:8080")
    parser.add_argument("--live-public-url", default="http://192.168.2.197:8080")
    return parser.parse_args()


def load_registry() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for path in sorted(REGISTRY_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            model = json.load(handle)
        model_type = str(model.get("model_type") or model.get("id") or "")
        model.setdefault("complexity_rank", model_complexity_rank(model_type))
        model.setdefault("complexity_group", model_complexity_group(model_type))
        model["registry_path"] = str(path.relative_to(ROOT))
        models.append(model)
    return sorted(models, key=model_sort_key)


def model_by_id(ctx: DashboardContext, model_id: str) -> dict[str, Any]:
    for model in ctx.registry:
        if model.get("id") == model_id:
            return model
    raise KeyError(f"Unknown model id: {model_id}")


def allowed_vars_for(model: dict[str, Any]) -> set[str]:
    names = {field["name"] for field in model.get("fields", []) if "name" in field}
    names.update(model.get("defaults", {}).keys())
    names.update(COMMON_ALLOWED_VARS)
    return names


def clean_params(model: dict[str, Any], raw: dict[str, Any]) -> dict[str, str]:
    allowed = allowed_vars_for(model)
    cleaned: dict[str, str] = {}
    for key, value in raw.items():
        if not SAFE_VAR_RE.match(key) or key not in allowed:
            continue
        if value is None:
            continue
        text = str(value).strip()
        if not text or "\n" in text or "\r" in text or len(text) > 500:
            continue
        cleaned[key] = text
    return cleaned


def now_utc_string() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_settings() -> dict[str, Any]:
    return {"version": 1, "updated_at": "", "shared": {}, "models": {}, "live": {}}


def normalize_settings(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    settings = default_settings()
    settings.update({key: value for key, value in data.items() if key in settings})
    if not isinstance(settings.get("shared"), dict):
        settings["shared"] = {}
    if not isinstance(settings.get("models"), dict):
        settings["models"] = {}
    if not isinstance(settings.get("live"), dict):
        settings["live"] = {}
    settings["shared"] = {
        str(key): str(value)
        for key, value in settings["shared"].items()
        if SAFE_VAR_RE.match(str(key)) and value is not None
    }
    models: dict[str, dict[str, str]] = {}
    for model_id, model_settings in settings["models"].items():
        if not isinstance(model_settings, dict):
            continue
        models[str(model_id)] = {
            str(key): str(value)
            for key, value in model_settings.items()
            if SAFE_VAR_RE.match(str(key)) and value is not None
        }
    settings["models"] = models
    settings["live"] = {
        str(key): str(value)
        for key, value in settings["live"].items()
        if SAFE_VAR_RE.match(str(key)) and key in LIVE_ALLOWED_VARS and value is not None
    }
    return settings


def load_dashboard_settings(ctx: DashboardContext) -> dict[str, Any]:
    if not ctx.settings_path.exists():
        settings = default_settings()
    else:
        try:
            settings = normalize_settings(json.loads(ctx.settings_path.read_text(encoding="utf-8")))
        except Exception:
            settings = default_settings()
    live_env = read_live_env_settings(ctx.root / "live_sim" / ".env")
    settings["live"] = {**live_env, **settings.get("live", {})}
    settings["path"] = str(ctx.settings_path.relative_to(ctx.root) if ctx.settings_path.is_relative_to(ctx.root) else ctx.settings_path)
    settings["shared_keys"] = sorted(SHARED_SETTING_VARS)
    return settings


def read_live_env_settings(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    mapped = {
        "TRAIN_MODEL_TYPE": "LIVE_TRAIN_MODEL_TYPE",
        "TRAIN_BACKEND": "LIVE_TRAIN_BACKEND",
        "TRAIN_DEVICE": "LIVE_TRAIN_DEVICE",
        "TRAIN_LOOKBACK": "LIVE_TRAIN_LOOKBACK",
        "TRAIN_SEQUENCE_FEATURE_SET": "LIVE_TRAIN_SEQUENCE_FEATURE_SET",
        "TRAIN_EDGE": "LIVE_TRAIN_EDGE",
        "TRAIN_USE_FULL_WINDOW": "LIVE_TRAIN_USE_FULL_WINDOW",
        "RETRAIN_FREQUENCY": "LIVE_RETRAIN_FREQUENCY",
        "RETRAIN_TRAIN_START": "LIVE_RETRAIN_TRAIN_START",
        "RETRAIN_TRAIN_END": "LIVE_RETRAIN_TRAIN_END",
        "RETRAIN_LOOKBACK_DAYS": "LIVE_RETRAIN_LOOKBACK_DAYS",
        "STARTING_CASH": "LIVE_STARTING_CASH",
        "MAX_INVEST": "LIVE_MAX_INVEST",
        "MAX_SHORT_INVEST": "LIVE_MAX_SHORT_INVEST",
        "MIN_INVEST": "LIVE_MIN_INVEST",
        "CONFIDENCE_MULTIPLIER": "LIVE_CONFIDENCE_MULTIPLIER",
        "SLIPPAGE": "LIVE_SLIPPAGE",
        "ENTRY_THRESHOLD": "THRESHOLD",
        "TRAIN_DECISION_THRESHOLD": "THRESHOLD",
        "TRAIN_CNN_FILTERS": "NN_CNN_FILTERS",
        "TRAIN_CNN_KERNEL_SIZES": "NN_CNN_KERNEL_SIZES",
        "TRAIN_LSTM_HIDDEN_SIZE": "NN_LSTM_HIDDEN_SIZE",
        "TRAIN_LSTM_LAYERS": "NN_LSTM_LAYERS",
        "TRAIN_LSTM_DROPOUT": "NN_LSTM_DROPOUT",
        "TRAIN_GRU_HIDDEN_SIZE": "NN_GRU_HIDDEN_SIZE",
        "TRAIN_GRU_LAYERS": "NN_GRU_LAYERS",
        "TRAIN_GRU_DROPOUT": "NN_GRU_DROPOUT",
        "TRAIN_TRANSFORMER_D_MODEL": "NN_TRANSFORMER_D_MODEL",
        "TRAIN_TRANSFORMER_HEADS": "NN_TRANSFORMER_HEADS",
        "TRAIN_TRANSFORMER_LAYERS": "NN_TRANSFORMER_LAYERS",
        "TRAIN_TRANSFORMER_FF_DIM": "NN_TRANSFORMER_FF_DIM",
        "TRAIN_TRANSFORMER_DROPOUT": "NN_TRANSFORMER_DROPOUT",
        "TRAIN_HIDDEN_LAYERS": "NN_HIDDEN_LAYERS",
        "TRAIN_LR": "NN_LR",
        "TRAIN_EPOCHS": "NN_EPOCHS",
        "TRAIN_BATCH_SIZE": "NN_BATCH_SIZE",
        "TRAIN_L2": "NN_L2",
        "TRAIN_CLASS_WEIGHT_MODE": "NN_CLASS_WEIGHT_MODE",
        "TRAIN_SEED": "NN_SEED",
    }
    out: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = mapped.get(key.strip(), key.strip())
        value = value.strip()
        if SAFE_VAR_RE.match(key) and key in LIVE_ALLOWED_VARS:
            out[key] = value
    if out.get("LIVE_TRAIN_MODEL_TYPE") in {"cnn", "mlp", "gru", "lstm", "transformer"}:
        out.setdefault("LIVE_MODEL_TYPE", out["LIVE_TRAIN_MODEL_TYPE"])
    return out


def write_dashboard_settings(ctx: DashboardContext, settings: dict[str, Any]) -> None:
    settings = normalize_settings(settings)
    settings["updated_at"] = now_utc_string()
    ctx.settings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ctx.settings_path.with_suffix(ctx.settings_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(ctx.settings_path)


def save_model_settings(ctx: DashboardContext, payload: dict[str, Any]) -> dict[str, Any]:
    model_id = str(payload.get("model", ""))
    model = model_by_id(ctx, model_id)
    params = clean_params(model, payload.get("params", {}) if isinstance(payload.get("params"), dict) else {})
    settings = load_dashboard_settings(ctx)
    settings.pop("path", None)
    settings.pop("shared_keys", None)
    model_settings = settings.setdefault("models", {}).setdefault(model_id, {})

    for key, value in params.items():
        if key in SHARED_SETTING_VARS:
            settings.setdefault("shared", {})[key] = value
        else:
            model_settings[key] = value

    write_dashboard_settings(ctx, settings)
    return load_dashboard_settings(ctx)


def clean_live_params(raw: dict[str, Any]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in raw.items():
        if not SAFE_VAR_RE.match(key) or key not in LIVE_ALLOWED_VARS:
            continue
        if value is None:
            continue
        text = str(value).strip()
        if not text or "\n" in text or "\r" in text or len(text) > 500:
            continue
        cleaned[key] = text
    return cleaned


def save_live_settings(ctx: DashboardContext, params: dict[str, str]) -> dict[str, Any]:
    settings = load_dashboard_settings(ctx)
    settings.pop("path", None)
    settings.pop("shared_keys", None)
    settings.setdefault("live", {}).update(params)
    write_dashboard_settings(ctx, settings)
    return load_dashboard_settings(ctx)


def build_links(model: dict[str, Any], params: dict[str, str]) -> dict[str, str]:
    merged = {**model.get("defaults", {}), **params}
    links: dict[str, str] = {}
    for name, pattern in model.get("links", {}).items():
        try:
            links[name] = pattern.format(**merged)
        except KeyError:
            continue
    return links


def auto_graph_links(action: str, links: dict[str, str]) -> dict[str, str]:
    if action == "full":
        return {key: value for key, value in links.items() if key in {"model", "simulation"} and value}
    if action == "visualize" and links.get("model"):
        return {"model": links["model"]}
    if action in {"simulate", "quick_simulate"} and links.get("simulation"):
        return {"simulation": links["simulation"]}
    return {}


def deduplicate_targets(*target_groups: list[str]) -> list[str]:
    targets: list[str] = []
    for group in target_groups:
        for target in group:
            if target not in targets:
                targets.append(target)
    return targets


def model_action_targets(model: dict[str, Any], action: str) -> list[str]:
    actions = model.get("actions", {})
    if action == "simulate":
        # A simulation must rebuild predictions when data or training settings changed.
        return deduplicate_targets(actions.get("train", []), actions.get("simulate", []))
    if action == "quick_simulate":
        return list(actions.get("simulate", []))
    return list(actions.get(action, []))


def build_model_job(ctx: DashboardContext, payload: dict[str, Any]) -> Job:
    model = model_by_id(ctx, str(payload.get("model", "")))
    action = str(payload.get("action", "full"))
    targets = model_action_targets(model, action)
    if not targets:
        raise ValueError(f"Unsupported action for {model['id']}: {action}")
    params = clean_params(model, payload.get("params", {}) if isinstance(payload.get("params"), dict) else {})
    command_params = clean_params(model, {**model.get("defaults", {}), **params})
    save_model_settings(ctx, {"model": model["id"], "params": params})
    command = ["make", *targets, *[f"{key}={value}" for key, value in sorted(command_params.items())]]
    links = build_links(model, command_params)
    label = f"{model.get('label', model['id'])} {action}"
    return ctx.runner.start(
        label=label,
        command=command,
        cwd=ctx.root,
        links=links,
        auto_graphs=auto_graph_links(action, links),
        model_id=model["id"],
        action=action,
    )


def build_live_job(ctx: DashboardContext, action: str, raw_params: dict[str, Any] | None = None) -> Job:
    targets = LIVE_ACTION_TARGETS.get(action)
    if not targets:
        raise ValueError(f"Unsupported live action: {action}")
    params = clean_live_params(raw_params or {})
    if params:
        save_live_settings(ctx, params)
    command = ["make", *targets, *[f"{key}={value}" for key, value in sorted(params.items())]]
    return ctx.runner.start(label=f"Live sim {action.replace('_', ' ')}", command=command, cwd=ctx.root)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def proxy_json(url: str, method: str = "GET", timeout: float = 2.5) -> tuple[int, Any]:
    request = Request(url, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - local configured URL.
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except URLError as exc:
        return 503, {"running": False, "error": str(exc), "url": url}
    except Exception as exc:  # noqa: BLE001 - API should return readable errors.
        return 500, {"running": False, "error": str(exc), "url": url}


def query_bool(query: dict[str, list[str]], key: str, default: bool = False) -> bool:
    values = query.get(key)
    if not values:
        return default
    return values[0].strip().lower() in {"1", "true", "yes", "on"}


def query_float(query: dict[str, list[str]], key: str, default: float) -> float:
    values = query.get(key)
    if not values:
        return default
    try:
        return float(values[0])
    except (TypeError, ValueError):
        return default


def query_text(query: dict[str, list[str]], key: str, default: str) -> str:
    values = query.get(key)
    if not values:
        return default
    text = values[0].strip()
    return text if text else default


def query_int(query: dict[str, list[str]], key: str, default: int) -> int:
    values = query.get(key)
    if not values:
        return default
    try:
        return int(float(values[0]))
    except (TypeError, ValueError):
        return default


def list_report_files(reports_root: Path) -> list[dict[str, str]]:
    files: list[dict[str, str]] = []
    if not reports_root.exists():
        return files
    for path in reports_root.rglob("*.html"):
        rel = path.relative_to(reports_root).as_posix()
        model_type = report_model_type(rel)
        if model_type and not is_supported_model_type(model_type):
            continue
        files.append(
            {
                "label": rel,
                "url": "/" + rel,
                "mtime": str(int(path.stat().st_mtime)),
                "complexity_rank": model_complexity_rank(model_type),
                "complexity_group": model_complexity_group(model_type),
            }
        )
    return sorted(files, key=lambda item: (int(item["complexity_rank"]), item["label"]))


def report_model_type(rel: str) -> str:
    parts = rel.split("/")
    if not parts:
        return ""
    if parts[0] == "sim":
        parts = parts[1:]
    if len(parts) >= 2 and parts[0] in {"nn", "strategy"}:
        return parts[1]
    if parts[0] == "xgb":
        return "xgboost"
    if parts[0] == "lr":
        return "logreg"
    return ""


def list_runs(root: Path) -> dict[str, Any]:
    current: list[dict[str, Any]] = []
    for path in sorted((root / "models" / "current").rglob("*.json")):
        item = read_json_summary(path)
        if is_supported_model_type(str(item.get("model_type", ""))):
            current.append(item)
    runs: list[dict[str, Any]] = []
    for path in sorted((root / "models" / "runs").glob("*/manifest.json"), reverse=True):
        runs.append(read_json_summary(path))
    current.sort(key=model_sort_key)
    return {"current": current, "runs": runs[:100]}


def read_json_summary(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "error": str(exc)}
    params = data.get("params", {}) if isinstance(data.get("params"), dict) else data
    return {
        "path": str(path),
        "run_id": data.get("run_id") or data.get("archive_name") or params.get("RUN_ID") or path.parent.name,
        "run_dir": data.get("run_dir") or data.get("archive_dir"),
        "created_at_utc": data.get("created_at_utc"),
        "family": data.get("family") or params.get("family") or params.get("FAMILY"),
        "model_type": data.get("model_type") or params.get("NN_MODEL_TYPE") or params.get("model_type"),
        "backend": data.get("backend") or params.get("NN_BACKEND") or params.get("backend"),
        "data_source": data.get("data_source") or params.get("DATA_SOURCE"),
        "symbol": data.get("symbol") or params.get("SYMBOL"),
        "interval": data.get("interval") or params.get("INTERVAL"),
        "summary_metrics": data.get("summary_metrics", {}),
        "artifacts": data.get("artifacts", {}),
    }


def model_label_from_parts(family: str, model_type: str, backend: str = "") -> str:
    if family == "nn":
        backend_text = f" {backend}" if backend else ""
        return f"{model_display_label(model_type)}{backend_text}"
    if family in {"xgb", "lr", "strategy"}:
        return model_display_label(model_type or family)
    return " ".join(part for part in [family, model_type, backend] if part).strip() or "Unknown model"


def sim_metadata_from_rel(rel: str) -> dict[str, str]:
    parts = rel.split("/")
    meta = {
        "family": "",
        "model_type": "",
        "backend": "",
        "source": "",
        "symbol": "",
        "interval": "",
        "asset_key": "",
        "asset_label": "",
        "model_key": "",
        "model_label": "",
    }
    if len(parts) >= 7 and parts[0] == "sim" and parts[1] == "nn":
        meta.update(
            {
                "family": parts[1],
                "model_type": parts[2],
                "source": parts[3],
                "symbol": parts[4],
                "interval": parts[5],
            }
        )
    elif len(parts) >= 7 and parts[0] == "sim" and parts[1] == "strategy":
        meta.update(
            {
                "family": parts[1],
                "model_type": parts[2],
                "source": parts[3],
                "symbol": parts[4],
                "interval": parts[5],
            }
        )
    elif len(parts) >= 6 and parts[0] == "sim":
        family = parts[1]
        meta.update(
            {
                "family": family,
                "model_type": "xgboost" if family == "xgb" else "logreg" if family == "lr" else family,
                "source": parts[2],
                "symbol": parts[3],
                "interval": parts[4],
            }
        )
    return finalize_series_metadata(meta)


def training_metadata_from_rel(rel: str) -> dict[str, str]:
    parts = rel.split("/")
    meta = {
        "family": "",
        "model_type": "",
        "backend": "",
        "source": "",
        "symbol": "",
        "interval": "",
        "asset_key": "",
        "asset_label": "",
        "model_key": "",
        "model_label": "",
    }
    if len(parts) >= 7 and parts[0] == "models" and parts[1] == "nn":
        meta.update(
            {
                "family": parts[1],
                "model_type": parts[2],
                "source": parts[3],
                "symbol": parts[4],
                "interval": parts[5],
            }
        )
    elif len(parts) >= 7 and parts[0] == "models" and parts[1] == "strategy":
        meta.update(
            {
                "family": parts[1],
                "model_type": parts[2],
                "source": parts[3],
                "symbol": parts[4],
                "interval": parts[5],
            }
        )
    elif len(parts) >= 6 and parts[0] == "models":
        family = parts[1]
        meta.update(
            {
                "family": family,
                "model_type": "xgboost" if family == "xgb" else "logreg" if family == "lr" else family,
                "source": parts[2],
                "symbol": parts[3],
                "interval": parts[4],
            }
        )
    return finalize_series_metadata(meta)


def finalize_series_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    source = meta.get("source", "")
    symbol = meta.get("symbol", "")
    interval = meta.get("interval", "")
    family = meta.get("family", "")
    model_type = meta.get("model_type", "")
    backend = meta.get("backend", "")
    meta["asset_key"] = "/".join(part for part in [source, symbol] if part)
    meta["asset_label"] = "/".join(part for part in [source, symbol] if part) or "Unknown asset"
    meta["model_key"] = "/".join(part for part in [family, model_type, backend] if part)
    meta["model_label"] = model_label_from_parts(family, model_type, backend)
    meta["complexity_rank"] = model_complexity_rank(model_type or family)
    meta["complexity_group"] = model_complexity_group(model_type or family)
    meta["series_key"] = "/".join(part for part in [source, symbol, interval, family, model_type, backend] if part)
    return meta


def list_sim_series(reports_root: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if not reports_root.exists():
        return items
    for path in sorted(reports_root.rglob("bank_trades.csv")):
        rel = path.relative_to(reports_root).as_posix()
        label = rel.replace("sim/", "").replace("/bank_trades.csv", "")
        html_path = path.with_name("visualization.html")
        html_rel = html_path.relative_to(reports_root).as_posix() if html_path.exists() else ""
        prediction_rel = prediction_rel_from_sim_rel(rel)
        prediction_path = safe_child(reports_root, prediction_rel) if prediction_rel else None
        item = {
            "path": rel,
            "label": label,
            "visualization_url": "/" + html_rel if html_rel else "",
            "prediction_path": prediction_rel if prediction_path and prediction_path.exists() else "",
        }
        item.update(sim_metadata_from_rel(rel))
        if is_supported_model_type(str(item.get("model_type", ""))):
            items.append(item)
    return sorted(items, key=model_sort_key)


def prediction_rel_from_sim_rel(rel: str) -> str:
    parts = rel.split("/")
    if len(parts) >= 7 and parts[0] == "sim" and parts[1] == "nn":
        return "/".join(["nn", parts[2], parts[3], parts[4], parts[5], "predictions.parquet"])
    if len(parts) >= 7 and parts[0] == "sim" and parts[1] == "strategy":
        return "/".join(["strategy", parts[2], parts[3], parts[4], parts[5], "predictions.parquet"])
    if len(parts) >= 6 and parts[0] == "sim" and parts[1] in {"xgb", "lr"}:
        return "/".join([parts[1], parts[2], parts[3], parts[4], "predictions.parquet"])
    return ""


def downsample_points(points: list[dict[str, Any]], max_points: int = 2500) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points
    step = max(1, math.ceil(len(points) / max_points))
    out = points[::step]
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def compare_investment_size(
    prob_up: float,
    threshold: float,
    min_invest: float,
    max_invest: float,
    confidence_multiplier: float,
) -> float:
    confidence = (prob_up - threshold) / max(1e-12, 1.0 - threshold)
    confidence *= confidence_multiplier
    confidence = min(max(confidence, 0.0), 1.0)
    return min_invest + (max_invest - min_invest) * math.sqrt(confidence)


def compare_short_investment_size(
    prob_down: float,
    threshold: float,
    min_invest: float,
    max_invest: float,
    confidence_multiplier: float,
) -> float:
    confidence = (prob_down - threshold) / max(1e-12, 1.0 - threshold)
    confidence *= confidence_multiplier
    confidence = min(max(confidence, 0.0), 1.0)
    return min_invest + (max_invest - min_invest) * math.sqrt(confidence)


def compare_choose_side(prob_up: float, prob_down: float, trade_mode: str, threshold: float, short_threshold: float) -> str:
    long_signal = trade_mode in {"long_only", "long_short"} and prob_up >= threshold
    short_signal = trade_mode in {"short_only", "long_short"} and prob_down >= short_threshold
    if long_signal and short_signal:
        long_conf = (prob_up - threshold) / max(1e-12, 1.0 - threshold)
        short_conf = (prob_down - short_threshold) / max(1e-12, 1.0 - short_threshold)
        return "long" if long_conf >= short_conf else "short"
    if long_signal:
        return "long"
    if short_signal:
        return "short"
    return ""


def parse_compare_max_invest(expr: Any, available_cash: float) -> float:
    raw = str(expr).strip().lower().replace(" ", "")
    if raw == "m":
        return available_cash
    if MAX_INVEST_NUMERIC_RE.match(raw):
        return float(raw)
    coeff_match = MAX_INVEST_COEFF_M_RE.match(raw)
    if coeff_match:
        return float(coeff_match.group(1)) * available_cash
    div_match = MAX_INVEST_M_DIV_RE.match(raw)
    if div_match:
        divisor = float(div_match.group(1))
        return available_cash / divisor if divisor > 0.0 else 0.0
    return 0.0


def parse_optional_utc_timestamp(raw: str | None, *, end_of_day: bool = False) -> pd.Timestamp | None:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    if len(text) == 10:
        text = f"{text}T{'23:59:59' if end_of_day else '00:00:00'}"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def normalized_compare_points(
    predictions_path: Path,
    *,
    model_type: str,
    starting_cash: float,
    min_invest: float,
    max_invest: Any,
    threshold: float,
    exit_threshold: float,
    trade_mode: str,
    short_entry_threshold: float,
    short_exit_threshold: float,
    max_short_invest: Any,
    borrow_fee: float,
    allow_flip_position: bool,
    fee: float,
    confidence_multiplier: float,
    max_hold_bars: int,
    stop_loss: float,
    take_profit: float,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> list[dict[str, Any]]:
    df = pd.read_parquet(predictions_path)
    required = {"open_time", "close", "prob_up"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{predictions_path} missing columns: {sorted(missing)}")

    if "dataset_split" in df.columns:
        split_values = df["dataset_split"].astype(str).str.lower()
        test_rows = df[split_values == "test"]
        if not test_rows.empty:
            df = test_rows
    if "prob_down" not in df.columns:
        df["prob_down"] = 0.0
    df = df[["open_time", "close", "prob_up", "prob_down"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["prob_up"] = pd.to_numeric(df["prob_up"], errors="coerce")
    df["prob_down"] = pd.to_numeric(df["prob_down"], errors="coerce")
    df = df.dropna(subset=["open_time", "close", "prob_up", "prob_down"])
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    if start is not None:
        df = df[df["open_time"] >= start]
    if end is not None:
        df = df[df["open_time"] <= end]
    df = df.reset_index(drop=True)
    if df.empty:
        return []

    starting_cash = max(0.01, float(starting_cash))
    min_invest = max(0.01, float(min_invest))
    threshold = min(max(float(threshold), 1e-6), 0.999999)
    exit_threshold = min(max(float(exit_threshold), 0.0), 0.999999)
    trade_mode = trade_mode if trade_mode in {"long_only", "short_only", "long_short"} else "long_only"
    short_entry_threshold = min(max(float(short_entry_threshold), 1e-6), 0.999999)
    short_exit_threshold = min(max(float(short_exit_threshold), 1e-6), 0.999999)
    borrow_fee = max(0.0, float(borrow_fee))
    fee = min(max(float(fee), 0.0), 0.99)
    confidence_multiplier = max(1e-6, float(confidence_multiplier))
    max_hold_bars = max(0, int(max_hold_bars))
    stop_loss = max(0.0, float(stop_loss))
    take_profit = max(0.0, float(take_profit))

    cash = starting_cash
    quantity = 0.0
    position_side = ""
    investment = 0.0
    entry_fee = 0.0
    entry_price = 0.0
    entry_prob = 0.0
    bars_held = 0
    completed_buy_hold = False
    points: list[dict[str, Any]] = [{"t": df["open_time"].iloc[0].isoformat(), "v": cash}]

    for idx, row in enumerate(df.itertuples(index=False)):
        timestamp = pd.Timestamp(row.open_time)
        close = float(row.close)
        prob_up = float(row.prob_up)
        prob_down = float(row.prob_down)
        last_row = idx == len(df) - 1

        had_position = bool(position_side)
        closed_side = ""
        if position_side:
            bars_held += 1
            gross_return = (
                close / entry_price - 1.0
                if position_side == "long" and entry_price > 0.0
                else (entry_price - close) / entry_price if entry_price > 0.0 else 0.0
            )
            should_exit = False
            if model_type == "buy_hold":
                should_exit = last_row
            elif last_row:
                should_exit = True
            elif position_side == "long" and prob_up < exit_threshold:
                should_exit = True
            elif position_side == "short" and prob_down < short_exit_threshold:
                should_exit = True
            elif stop_loss > 0.0 and gross_return <= -stop_loss:
                should_exit = True
            elif take_profit > 0.0 and gross_return >= take_profit:
                should_exit = True
            elif max_hold_bars > 0 and bars_held >= max_hold_bars:
                should_exit = True

            if should_exit:
                closed_side = position_side
                gross_exit_value = quantity * close
                exit_fee = gross_exit_value * fee
                if position_side == "long":
                    cash += gross_exit_value - exit_fee
                else:
                    accrued_borrow = investment * borrow_fee * bars_held
                    cash += investment + quantity * (entry_price - close) - exit_fee - accrued_borrow
                position_side = ""
                quantity = 0.0
                completed_buy_hold = model_type == "buy_hold"
                entry_price = 0.0
                entry_prob = 0.0
                investment = 0.0
                entry_fee = 0.0
                bars_held = 0

        can_enter = not position_side and not completed_buy_hold and not last_row and (not had_position or allow_flip_position)
        selected_side = (
            "long"
            if model_type == "buy_hold"
            else compare_choose_side(prob_up, prob_down, trade_mode, threshold, short_entry_threshold)
        )
        long_signal = selected_side == "long"
        short_signal = selected_side == "short"
        if had_position and allow_flip_position:
            long_signal = long_signal and closed_side == "short"
            short_signal = short_signal and closed_side == "long"
        if can_enter and (long_signal or short_signal):
            next_side = "long" if long_signal else "short"
            required_cash = min_invest * (1.0 + fee)
            if cash >= required_cash:
                cap_expr = max_invest if next_side == "long" else max_short_invest
                max_cap = min(parse_compare_max_invest(cap_expr, cash), cash / (1.0 + fee))
                if max_cap < min_invest:
                    continue
                planned = (
                    compare_investment_size(prob_up, threshold, min_invest, max_cap, confidence_multiplier)
                    if next_side == "long"
                    else compare_short_investment_size(
                        prob_down, short_entry_threshold, min_invest, max_cap, confidence_multiplier
                    )
                )
                investment = min(planned, cash / (1.0 + fee))
                if investment >= min_invest and close > 0.0:
                    entry_fee = investment * fee
                    cash -= investment + entry_fee
                    quantity = investment / close
                    position_side = next_side
                    entry_price = close
                    entry_prob = prob_up
                    bars_held = 0

        if position_side == "long":
            liquidation_value = cash + quantity * close * (1.0 - fee)
        elif position_side == "short":
            liquidation_value = (
                cash
                + investment
                + quantity * (entry_price - close)
                - quantity * close * fee
                - investment * borrow_fee * bars_held
            )
        else:
            liquidation_value = cash
        points.append({"t": timestamp.isoformat(), "v": float(liquidation_value)})

    return downsample_points(points)


def sim_series(
    reports_root: Path,
    rel_paths: list[str],
    *,
    normalized: bool = False,
    starting_cash: float = 100.0,
    min_invest: float = 1.0,
    max_invest: Any = "m",
    threshold: float = 0.52,
    exit_threshold: float = 0.50,
    trade_mode: str = "long_only",
    short_entry_threshold: float = 0.45,
    short_exit_threshold: float = 0.52,
    max_short_invest: Any = "m",
    borrow_fee: float = 0.0,
    allow_flip_position: bool = False,
    fee: float = 0.0001,
    confidence_multiplier: float = 1.0,
    max_hold_bars: int = 60,
    stop_loss: float = 0.002,
    take_profit: float = 0.004,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel in rel_paths[:10]:
        safe = safe_child(reports_root, rel)
        if safe is None or not safe.exists() or safe.suffix.lower() != ".csv":
            continue
        meta = sim_metadata_from_rel(rel)
        points: list[dict[str, Any]]
        curve_source = "saved_trades_csv"
        prediction_rel = prediction_rel_from_sim_rel(rel)
        if normalized and prediction_rel:
            prediction_path = safe_child(reports_root, prediction_rel)
            if prediction_path is None or not prediction_path.exists():
                continue
            points = normalized_compare_points(
                prediction_path,
                model_type=meta.get("model_type", ""),
                starting_cash=starting_cash,
                min_invest=min_invest,
                max_invest=max_invest,
                threshold=threshold,
                exit_threshold=exit_threshold,
                trade_mode=trade_mode,
                short_entry_threshold=short_entry_threshold,
                short_exit_threshold=short_exit_threshold,
                max_short_invest=max_short_invest,
                borrow_fee=borrow_fee,
                allow_flip_position=allow_flip_position,
                fee=fee,
                confidence_multiplier=confidence_multiplier,
                max_hold_bars=max_hold_bars,
                stop_loss=stop_loss,
                take_profit=take_profit,
                start=start,
                end=end,
            )
            curve_source = "normalized_predictions"
        else:
            points = []
            with safe.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    t = row.get("exit_time") or row.get("entry_time")
                    value = row.get("account_value_after_trade") or row.get("cash_after_trade")
                    if not t or value in {None, ""}:
                        continue
                    try:
                        points.append({"t": t, "v": float(value)})
                    except ValueError:
                        continue
            points = downsample_points(points)
        label = rel.replace("sim/", "").replace("/bank_trades.csv", "")
        item = {
            "path": rel,
            "label": label,
            "points": points,
            "curve_source": curve_source,
            "prediction_path": prediction_rel,
        }
        item.update(meta)
        out.append(item)
    return sorted(out, key=model_sort_key)


def list_training_series(root: Path) -> list[dict[str, str]]:
    candidates = []
    for base in [root / "models" / "nn", root / "models" / "xgb", root / "models" / "lr", root / "models" / "strategy"]:
        if base.exists():
            candidates.extend(base.rglob("train_metrics.json"))
            candidates.extend(base.rglob("preflight_train_metrics.json"))
    items = []
    for path in sorted(set(candidates)):
        rel = path.relative_to(root).as_posix()
        label = rel.replace("models/", "").replace("/train_metrics.json", "").replace("/preflight_train_metrics.json", "/preflight")
        item = {"path": rel, "label": label}
        item.update(training_metadata_from_rel(rel))
        if is_supported_model_type(str(item.get("model_type", ""))):
            items.append(item)
    return sorted(items, key=model_sort_key)


def training_series(root: Path, rel_paths: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel in rel_paths[:10]:
        safe = safe_child(root, rel)
        if safe is None or not safe.exists() or safe.suffix.lower() != ".json":
            continue
        try:
            data = json.loads(safe.read_text(encoding="utf-8"))
        except Exception:
            continue
        training = data.get("model_training", {})
        history = training.get("loss_history") if isinstance(training, dict) else None
        if not isinstance(history, list):
            final = training.get("final_loss") if isinstance(training, dict) else None
            history = [final] if isinstance(final, (int, float)) else []
        points = [{"t": idx + 1, "v": float(value)} for idx, value in enumerate(history) if isinstance(value, (int, float))]
        label = rel.replace("models/", "").replace("/train_metrics.json", "").replace("/preflight_train_metrics.json", "/preflight")
        item = {"path": rel, "label": label, "points": points}
        item.update(training_metadata_from_rel(rel))
        out.append(item)
    return out


def safe_child(root: Path, rel: str) -> Path | None:
    rel = unquote(rel).lstrip("/")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def live_logs() -> tuple[int, dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["docker", "compose", "logs", "--tail=200", "--no-color"],
            cwd=ROOT / "live_sim",
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        return 200, {"returncode": proc.returncode, "logs": (proc.stdout + proc.stderr)[-30000:]}
    except Exception as exc:  # noqa: BLE001
        return 500, {"error": str(exc), "logs": ""}


def sample_aligned_payload(data: dict[str, Any], time_key: str, limit: int) -> dict[str, Any]:
    times = data.get(time_key)
    if not isinstance(times, list) or len(times) <= limit:
        return data
    step = max(1, math.ceil(len(times) / limit))
    indexes = list(range(0, len(times), step))
    if indexes[-1] != len(times) - 1:
        indexes.append(len(times) - 1)
    def sample_value(value: Any) -> Any:
        if isinstance(value, list) and len(value) == len(times):
            return [value[index] for index in indexes]
        if isinstance(value, dict):
            return {key: sample_value(item) for key, item in value.items()}
        return value

    return {key: sample_value(value) for key, value in data.items()}


def compact_report_payload(report_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    if report_type == "simulation":
        payload["candles"] = sample_aligned_payload(payload.get("candles", {}), "t", 4_000)
        payload["comparison"] = sample_aligned_payload(
            payload.get("comparison", {}), "t", 4_000
        )
        return payload

    if report_type == "model":
        payload["candles"] = {
            key: sample_aligned_payload(frame, "x", 1_800)
            for key, frame in payload.get("candles", {}).items()
        }
        payload["predictions"] = sample_aligned_payload(
            payload.get("predictions", {}), "x", 4_000
        )
        payload["equity"] = sample_aligned_payload(payload.get("equity", {}), "x", 4_000)
        payload["markers"] = {
            key: sample_aligned_payload(group, "x", 800)
            for key, group in payload.get("markers", {}).items()
        }
    return payload


def extract_report_payload(
    ctx: DashboardContext, url: str, *, compact: bool = False
) -> dict[str, Any]:
    parsed = urlparse(url)
    rel = parsed.path or url
    file_path = safe_child(ctx.reports_root, rel)
    if file_path is None or not file_path.exists() or file_path.suffix.lower() != ".html":
        raise FileNotFoundError(f"Report HTML not found under data/reports: {rel}")

    cache_key = (str(file_path), file_path.stat().st_mtime_ns)
    if compact and cache_key in REPORT_PAYLOAD_CACHE:
        return REPORT_PAYLOAD_CACHE[cache_key]

    text = file_path.read_text(encoding="utf-8")
    marker = "const report ="
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"No embedded report payload found in {rel}")

    json_start = start + len(marker)
    payload, _ = json.JSONDecoder().raw_decode(text[json_start:].lstrip())
    if not isinstance(payload, dict):
        raise ValueError(f"Embedded report payload is not an object in {rel}")

    if "trades" in payload and "comparison" in payload:
        report_type = "simulation"
    elif "predictions" in payload and "markers" in payload:
        report_type = "model"
    else:
        report_type = "unknown"

    if compact:
        payload = compact_report_payload(report_type, payload)

    result = {
        "type": report_type,
        "url": parsed.path or rel,
        "payload": payload,
    }
    if compact:
        for key in list(REPORT_PAYLOAD_CACHE):
            if key[0] == str(file_path) and key != cache_key:
                del REPORT_PAYLOAD_CACHE[key]
        REPORT_PAYLOAD_CACHE[cache_key] = result
    return result


def make_handler(ctx: DashboardContext) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "CryptoPredDashboard/1.0"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = parse_qs(parsed.query)
            try:
                if path in {"/", "/index.html", "/models", "/compare", "/reports", "/live"}:
                    self.send_html(DASHBOARD_HTML)
                elif path == "/assets/cryptopred.css":
                    self.send_asset(REPORT_STYLE_PATH, "text/css; charset=utf-8")
                elif path == "/api/dashboard/status":
                    self.send_json({"ok": True, "host": ctx.host, "port": ctx.port})
                elif path == "/api/config":
                    self.send_json({"reports_host": ctx.host, "reports_port": ctx.port, "live_public_url": ctx.live_public_url})
                elif path == "/api/model-registry":
                    self.send_json(ctx.registry)
                elif path == "/api/settings":
                    self.send_json(load_dashboard_settings(ctx))
                elif path == "/api/jobs":
                    self.send_json(ctx.runner.list_jobs())
                elif path.startswith("/api/jobs/"):
                    job = ctx.runner.get(path.split("/", 3)[-1])
                    self.send_json(job if job else {"error": "not found"}, HTTPStatus.OK if job else HTTPStatus.NOT_FOUND)
                elif path == "/api/reports":
                    self.send_json(list_report_files(ctx.reports_root))
                elif path == "/api/runs":
                    self.send_json(list_runs(ctx.root))
                elif path == "/api/report-payload":
                    url_values = query.get("url", [])
                    if not url_values:
                        self.send_json({"error": "Missing url"}, HTTPStatus.BAD_REQUEST)
                    else:
                        self.send_json(
                            extract_report_payload(
                                ctx,
                                url_values[0],
                                compact=query_bool(query, "compact"),
                            )
                        )
                elif path == "/api/sim-series/list":
                    self.send_json(list_sim_series(ctx.reports_root))
                elif path == "/api/sim-series":
                    rels = query.get("path", [])
                    self.send_json(
                        sim_series(
                            ctx.reports_root,
                            rels,
                            normalized=query_bool(query, "normalized"),
                            starting_cash=query_float(query, "starting_cash", 100.0),
                            min_invest=query_float(query, "min_invest", 1.0),
                            max_invest=query_text(query, "max_invest", "m"),
                            threshold=query_float(query, "threshold", 0.52),
                            exit_threshold=query_float(query, "exit_threshold", 0.50),
                            trade_mode=query_text(query, "trade_mode", "long_only"),
                            short_entry_threshold=query_float(query, "short_entry_threshold", 0.45),
                            short_exit_threshold=query_float(query, "short_exit_threshold", 0.52),
                            max_short_invest=query_text(query, "max_short_invest", "m"),
                            borrow_fee=query_float(query, "borrow_fee", 0.0),
                            allow_flip_position=query_bool(query, "allow_flip_position"),
                            fee=query_float(query, "fee", 0.0001),
                            confidence_multiplier=query_float(query, "confidence_multiplier", 1.0),
                            max_hold_bars=query_int(query, "max_hold_bars", 60),
                            stop_loss=query_float(query, "stop_loss", 0.002),
                            take_profit=query_float(query, "take_profit", 0.004),
                            start=parse_optional_utc_timestamp(query.get("start", [""])[0]),
                            end=parse_optional_utc_timestamp(query.get("end", [""])[0], end_of_day=True),
                        )
                    )
                elif path == "/api/training-series/list":
                    self.send_json(list_training_series(ctx.root))
                elif path == "/api/training-series":
                    rels = query.get("path", [])
                    self.send_json(training_series(ctx.root, rels))
                elif path == "/api/live/logs":
                    status, payload = live_logs()
                    self.send_json(payload, HTTPStatus(status))
                elif path.startswith("/api/live/"):
                    suffix = path.removeprefix("/api/live")
                    status, payload = proxy_json(ctx.live_url + "/api" + suffix + ("?" + parsed.query if parsed.query else ""))
                    if isinstance(payload, dict):
                        payload.setdefault("running", status < 500)
                        payload.setdefault("public_url", ctx.live_public_url)
                    # Keep the dashboard page usable when Docker is stopped.
                    self.send_json(payload, HTTPStatus.OK)
                else:
                    self.serve_static(path)
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            try:
                payload = read_json_body(self)
                if path == "/api/jobs":
                    job = build_model_job(ctx, payload)
                    self.send_json(job.to_dict(), HTTPStatus.ACCEPTED)
                elif path == "/api/settings":
                    self.send_json(save_model_settings(ctx, payload))
                elif path == "/api/live/settings":
                    raw_params = payload.get("params", {}) if isinstance(payload.get("params"), dict) else {}
                    self.send_json(save_live_settings(ctx, clean_live_params(raw_params)))
                elif path == "/api/live/action":
                    raw_params = payload.get("params", {}) if isinstance(payload.get("params"), dict) else {}
                    job = build_live_job(ctx, str(payload.get("action", "")), raw_params)
                    self.send_json(job.to_dict(), HTTPStatus.ACCEPTED)
                else:
                    self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except RuntimeError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def serve_static(self, path: str) -> None:
            file_path = safe_child(ctx.reports_root, path)
            if file_path is None or not file_path.exists() or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            data = file_path.read_bytes()
            content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_asset(self, path: Path, content_type: str) -> None:
            if not path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, default=str, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_html(self, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}", flush=True)

    return Handler


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


DASHBOARD_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CryptoPred Dashboard</title>
  <link rel="stylesheet" href="/assets/cryptopred.css">
  <style>
    :root { --bg:#f4efe4; --paper:#fffaf0; --ink:#15120d; --muted:#6b6256; --line:#d8c8ad; --accent:#164f45; --orange:#c77519; --red:#b23834; --blue:#245ea8; --green:#16864e; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; color:var(--ink); background:radial-gradient(circle at 5% 0%, #fff2bd 0, var(--bg) 32%, #e8ded0 100%); font-family: Georgia, 'Times New Roman', serif; }
    header { position:sticky; top:0; z-index:5; display:flex; align-items:center; justify-content:space-between; gap:18px; padding:16px 22px; border-bottom:1px solid var(--line); background:rgba(255,250,240,.94); backdrop-filter:blur(8px); }
    h1 { margin:0; font-size:26px; letter-spacing:-.04em; }
    nav { display:flex; flex-wrap:wrap; gap:8px; }
    nav a, button, .button { border:1px solid var(--line); background:#fffdf8; color:var(--ink); padding:8px 12px; border-radius:999px; text-decoration:none; cursor:pointer; font:13px 'Courier New', monospace; }
    nav a.active, button.primary, .button.primary { background:var(--accent); color:white; border-color:var(--accent); }
    button.danger { background:var(--red); color:white; border-color:var(--red); }
    main { padding:22px; max-width:1500px; margin:0 auto; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }
    .card, .panel { border:1px solid var(--line); background:rgba(255,250,240,.9); box-shadow:0 12px 30px rgba(45,38,24,.08); border-radius:18px; padding:16px; }
    .card h2, .panel h2, .panel h3 { margin:0 0 10px; letter-spacing:-.03em; }
    .muted { color:var(--muted); font:13px 'Courier New', monospace; line-height:1.45; }
    .tabs { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; }
    .model-groups { display:grid; gap:10px; margin-bottom:16px; }
    .model-group { display:grid; grid-template-columns:minmax(150px,190px) 1fr; gap:10px; align-items:start; padding:10px; border:1px solid var(--line); border-radius:14px; background:rgba(255,253,248,.65); }
    .model-group-name { padding:8px 4px; color:var(--muted); font:12px 'Courier New', monospace; text-transform:uppercase; letter-spacing:.08em; }
    @media (max-width:700px) { .model-group { grid-template-columns:1fr; } }
    .formgrid { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }
    .settings-stack { display:grid; gap:10px; margin-top:14px; }
    details.settings-section { margin:0; padding:0; overflow:hidden; background:rgba(255,253,248,.7); }
    details.settings-section summary { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 14px; color:var(--ink); font:600 13px 'Courier New', monospace; list-style:none; }
    details.settings-section summary::-webkit-details-marker { display:none; }
    details.settings-section summary::before { content:'+'; width:20px; height:20px; display:inline-grid; place-items:center; flex:0 0 auto; border:1px solid var(--line); border-radius:50%; color:var(--accent); }
    details.settings-section[open] summary::before { content:'−'; }
    .settings-section-title { display:flex; align-items:center; gap:9px; }
    .settings-count { margin-left:auto; color:var(--muted); font-weight:400; }
    .settings-section-body { padding:4px 14px 15px; border-top:1px solid var(--line); }
    label { display:block; font:12px 'Courier New', monospace; color:var(--muted); }
    input, select { width:100%; margin-top:5px; padding:9px 10px; border:1px solid var(--line); border-radius:10px; background:#fffef9; color:var(--ink); }
    .actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
    .progress { height:14px; border:1px solid var(--line); border-radius:999px; overflow:hidden; background:#eee3d2; }
    .progress span { display:block; height:100%; width:0; background:linear-gradient(90deg,var(--orange),var(--accent)); transition:width .25s ease; }
    pre { white-space:pre-wrap; overflow:auto; max-height:360px; padding:12px; border-radius:12px; background:#17140f; color:#efe6d2; font:12px 'Courier New', monospace; }
    table { border-collapse:collapse; width:100%; font:12px 'Courier New', monospace; }
    th, td { border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }
    .status-ok { color:var(--green); } .status-bad { color:var(--red); }
    iframe { width:100%; height:640px; border:1px solid var(--line); border-radius:16px; background:white; }
    details { margin-top:12px; border:1px solid var(--line); border-radius:14px; background:rgba(255,253,248,.78); padding:10px 12px; }
    details summary { cursor:pointer; font:13px 'Courier New', monospace; color:var(--muted); }
    .job-graphs { display:grid; grid-template-columns:1fr; gap:14px; margin-top:14px; }
    .retained-graphs-head { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:8px; padding:2px 2px 0; font:600 13px 'Courier New', monospace; }
    .job-graph-head { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:8px; }
    .job-graph-head h3 { margin:0; }
    .graph-controls { display:flex; flex-wrap:wrap; gap:7px; }
    .job-graph.is-collapsed .job-graph-body { display:none; }
    .job-graph.is-expanded { position:fixed; inset:12px; z-index:80; overflow:auto; margin:0; background:var(--paper); box-shadow:0 24px 80px rgba(25,20,12,.34); }
    .job-graph.is-expanded .integrated-chart { height:min(50vh,520px); }
    body.graph-overlay-open { overflow:hidden; }
    .job-graph h3 { margin:0 0 8px; }
    .integrated-summary { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 12px; }
    .integrated-summary span { border:1px solid var(--line); background:#fffdf8; padding:5px 8px; border-radius:999px; font:12px 'Courier New', monospace; color:var(--muted); }
    .integrated-chart-shell { margin-top:10px; border:1px solid var(--line); border-radius:14px; background:#fffdf7; overflow:hidden; }
    .integrated-chart-head { display:flex; align-items:center; justify-content:space-between; gap:10px; min-height:42px; padding:7px 10px; border-bottom:1px solid var(--line); background:#f8f0df; }
    .integrated-chart-head strong { font:13px 'Courier New', monospace; }
    .integrated-chart-head .graph-controls button { padding:5px 9px; }
    .integrated-chart-shell.is-collapsed .integrated-chart { display:none; }
    .integrated-chart-shell.is-expanded { position:fixed; inset:12px; z-index:90; margin:0; overflow:auto; background:var(--paper); box-shadow:0 24px 80px rgba(25,20,12,.34); }
    .integrated-chart-shell.is-expanded .integrated-chart { height:calc(100vh - 86px); }
    .integrated-chart { position:relative; height:340px; background:#fffdf7; overflow:hidden; }
    .integrated-chart.price { height:430px; }
    .integrated-point-tooltip { position:absolute; display:none; z-index:4; min-width:210px; max-width:330px; padding:9px 11px; border:1px solid var(--line); border-radius:10px; background:rgba(255,253,248,.97); box-shadow:0 10px 28px rgba(30,24,14,.18); color:var(--ink); white-space:pre-line; pointer-events:none; font:12px 'Courier New', monospace; }
    .integrated-point-tooltip.is-visible { display:block; }
    .integrated-chart-loading { position:absolute; inset:0; display:grid; place-items:center; color:var(--muted); background:linear-gradient(100deg,#fffdf7 30%,#f4ead7 50%,#fffdf7 70%); background-size:240% 100%; animation:chart-loading 1.25s linear infinite; font:12px 'Courier New', monospace; z-index:2; }
    .integrated-chart.is-ready .integrated-chart-loading { display:none; }
    @keyframes chart-loading { to { background-position:-240% 0; } }
    .integrated-chart canvas { width:100%; height:100%; border:0; border-radius:0; }
    canvas { width:100%; height:320px; border:1px solid var(--line); border-radius:14px; background:#fffdf7; }
    .compare-stack { display:grid; grid-template-columns:1fr; gap:14px; }
    .compare-primary canvas { height:480px; }
    .compare-primary canvas { cursor:grab; }
    .compare-primary canvas.dragging { cursor:grabbing; }
    .compare-card canvas { height:360px; }
    .compare-filters { align-items:end; margin:12px 0; }
    .model-filter-list { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 12px; }
    .model-filter-list label { width:auto; display:flex; align-items:center; gap:6px; padding:7px 10px; border:1px solid var(--line); border-radius:999px; background:#fffdf8; color:var(--ink); }
    .model-filter-list input { width:auto; margin:0; }
    .compare-legend { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 10px; font:12px 'Courier New', monospace; }
    .compare-legend span { display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line); border-radius:999px; background:#fffdf8; padding:5px 8px; }
    .legend-swatch { width:22px; height:4px; border-radius:999px; display:inline-block; }
    .split { display:grid; grid-template-columns:minmax(320px, .8fr) minmax(420px, 1.2fr); gap:14px; }
    @media (max-width:900px) { .split { grid-template-columns:1fr; } header { align-items:flex-start; flex-direction:column; } }
  </style>
</head>
<body>
  <header>
    <div><h1>CryptoPred</h1><div class="muted" id="subhead">local research dashboard</div></div>
    <nav>
      <a href="/" data-route="/">Home</a>
      <a href="/models" data-route="/models">Models</a>
      <a href="/compare" data-route="/compare">Compare Models</a>
      <a href="/reports" data-route="/reports">Reports</a>
      <a href="/live" data-route="/live">Live</a>
    </nav>
  </header>
  <main id="app"></main>
<script>
const state = { registry: [], activeModel: null, currentJob: null, config: null, settings: { shared: {}, models: {}, shared_keys: [] }, saveTimer: null, modelGraphs: {}, modelGraphVersions: {}, sectionOpen: {} };
const $ = (sel, root=document) => root.querySelector(sel);
const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));
async function api(path, opts={}) {
  const res = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
function route(path) { history.pushState({}, '', path); render(); }
window.addEventListener('popstate', render);
document.addEventListener('click', (ev) => { const a = ev.target.closest('a[data-route]'); if (a) { ev.preventDefault(); route(a.getAttribute('href')); } });
function setActiveNav() { $$('nav a').forEach(a => a.classList.toggle('active', a.getAttribute('href') === location.pathname)); }
function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function pct(x) { return `${Math.round((Number(x)||0)*100)}%`; }
async function init() {
  const [config, registry, settings] = await Promise.all([api('/api/config'), api('/api/model-registry'), api('/api/settings')]);
  state.config = config; state.registry = registry; state.settings = settings; state.activeModel = state.registry[0]?.id;
  try { state.sectionOpen = JSON.parse(localStorage.getItem('cryptopred-model-sections') || '{}'); } catch (_) { state.sectionOpen = {}; }
  render(); setInterval(pollJobs, 1500); setInterval(refreshLive, 5000);
}
function render() { setActiveNav(); const p = location.pathname; if (p === '/models') return renderModels(); if (p === '/compare') return renderCompare(); if (p === '/reports') return renderReports(); if (p === '/live') return renderLive(); return renderHome(); }
function renderHome() {
  $('#app').innerHTML = `<div class="grid">
    <section class="card"><h2>Models</h2><p class="muted">Train and compare rule baselines, classical ML, CNN, GRU, and LSTM models from one place.</p><p><a class="button primary" href="/models" data-route="/models">Open Models</a></p></section>
    <section class="card"><h2>Compare Models</h2><p class="muted">Overlay saved simulation equity curves and training loss curves from generated outputs.</p><p><a class="button primary" href="/compare" data-route="/compare">Compare</a></p></section>
    <section class="card"><h2>Reports</h2><p class="muted">Open generated model and simulation visualizations from data/reports.</p><p><a class="button primary" href="/reports" data-route="/reports">Open Reports</a></p></section>
    <section class="card"><h2>Live</h2><p class="muted">Check and control the Docker paper-trading server on ${esc(state.config?.live_public_url || '')}.</p><p><a class="button primary" href="/live" data-route="/live">Open Live</a></p></section>
  </div>`;
}
function modelTabsHtml() {
  const groups = [];
  state.registry.forEach(model => {
    const name = model.complexity_group || 'Other';
    let group = groups.find(item => item.name === name);
    if (!group) { group = { name, models: [] }; groups.push(group); }
    group.models.push(model);
  });
  return groups.map(group => `<div class="model-group"><div class="model-group-name">${esc(group.name)}</div><div class="tabs">${group.models.map(m => `<button class="${m.id===state.activeModel?'primary':''}" data-model-tab="${esc(m.id)}" title="Complexity rank ${esc(m.complexity_rank)}">${esc(m.label)}</button>`).join('')}</div></div>`).join('');
}
function renderModels() {
  const model = state.registry.find(m => m.id === state.activeModel) || state.registry[0];
  state.activeModel = model?.id;
  $('#app').innerHTML = `<section class="panel"><h2>Models</h2><p class="muted">Models are ordered from least complex to most complex. Add a JSON file under src/model_registry/ with a complexity rank to add another entry.</p>
    <div class="model-groups">${modelTabsHtml()}</div>
    <div id="modelForm"></div></section><section class="panel" style="margin-top:14px"><h2>Current Job</h2><div class="progress"><span id="jobBar"></span></div><p class="muted" id="jobStatus">No job running.</p><div id="jobLinks"></div><details id="jobLogDetails" open><summary id="jobLogSummary">Terminal logs</summary><pre id="jobLog"></pre></details><div id="jobGraphs" class="job-graphs"></div></section>`;
  $$('[data-model-tab]').forEach(btn => btn.onclick = () => {
    const previousModel = state.activeModel;
    const params = collectFormParams();
    clearTimeout(state.saveTimer);
    applySettingsLocally(previousModel, params);
    saveSettingsFor(previousModel, params, false);
    state.activeModel = btn.dataset.modelTab;
    renderModels();
  });
  renderModelForm(model);
  renderStoredModelGraphs(model?.id);
  pollJobs();
}
function renderModelForm(model) {
  const fields = model.fields || [];
  const sectionFields = name => fields.filter(f => (f.section || 'model') === name);
  const section = name => sectionFields(name).map(f => fieldHtml(f, savedValue(model, f.name))).join('');
  const settingsSection = (name, title, description, defaultOpen=false) => {
    const sectionKey = `${model.id}:${name}`;
    const remembered = state.sectionOpen[sectionKey];
    const open = remembered === undefined ? defaultOpen : remembered;
    const count = sectionFields(name).length;
    if (!count) return '';
    return `<details class="settings-section" data-settings-section="${esc(sectionKey)}" ${open?'open':''}><summary><span class="settings-section-title">${esc(title)}</span><span class="settings-count">${count} setting${count===1?'':'s'}</span></summary><div class="settings-section-body">${description?`<p class="muted">${esc(description)}</p>`:''}<div class="formgrid">${section(name)}</div></div></details>`;
  };
  $('#modelForm').innerHTML = `<h3>${esc(model.label)}</h3><p class="muted">${esc(model.description)}</p><p class="muted">Starting preset: <strong>${esc(model.preset || 'custom')}</strong>. These are conservative defaults, not optimized results.</p><p class="muted" id="settingsStatus">Settings save to ${esc(state.settings?.path || 'data/reports/dashboard_settings.json')}.</p>
    <div class="actions"><button type="button" data-settings-action="expand">Expand all settings</button><button type="button" data-settings-action="collapse">Collapse all settings</button></div>
    <div class="settings-stack">
      ${settingsSection('data','Shared Data Settings','Asset, interval, date range, split, and label definition.',true)}
      ${settingsSection('trading','Trading Settings','Entry, exit, fee, cash, and position-sizing controls.',false)}
      ${settingsSection('advanced','Advanced Trading Settings','Risk controls and paper-short settings. Leverage remains restricted to 1x.',false)}
      ${settingsSection('model','Model-Specific Settings','Architecture, optimizer, regularization, and training controls for this model only.',false)}
    </div>
    <p class="muted">Simulate rebuilds the model predictions first, so every current data, training, and trading setting is applied. Quick Sim reuses the existing prediction file and applies only simulation/trading settings.</p><p class="muted">Model simulations replay the same test rows twice: long only and long + short. Both equity curves appear together.</p><div class="actions">
    <button class="primary" data-action="full">Train + Sim + Visualize</button><button data-action="train">Train Only</button><button data-action="simulate">Simulate With Current Settings</button><button data-action="quick_simulate">Quick Sim Existing Predictions</button><button data-action="visualize">Visualize</button></div>`;
  $$('details.settings-section', $('#modelForm')).forEach(details => {
    details.addEventListener('toggle', () => {
      state.sectionOpen[details.dataset.settingsSection] = details.open;
      localStorage.setItem('cryptopred-model-sections', JSON.stringify(state.sectionOpen));
    });
  });
  $$('[data-settings-action]', $('#modelForm')).forEach(button => {
    button.onclick = () => {
      const open = button.dataset.settingsAction === 'expand';
      $$('details.settings-section', $('#modelForm')).forEach(details => {
        details.open = open;
        state.sectionOpen[details.dataset.settingsSection] = open;
      });
      localStorage.setItem('cryptopred-model-sections', JSON.stringify(state.sectionOpen));
    };
  });
  $$('[data-action]', $('#modelForm')).forEach(btn => btn.onclick = () => startModelJob(model.id, btn.dataset.action));
  $$('input,select', $('#modelForm')).forEach(el => {
    el.addEventListener('input', scheduleSettingsSave);
    el.addEventListener('change', scheduleSettingsSave);
  });
}
function fieldHtml(f, value) {
  if (f.type === 'select') return `<label>${esc(f.label)}<select name="${esc(f.name)}">${(f.options||[]).map(o => `<option ${o==value?'selected':''}>${esc(o)}</option>`).join('')}</select></label>`;
  return `<label>${esc(f.label)}<input name="${esc(f.name)}" type="${esc(f.type||'text')}" step="${esc(f.step||'any')}" value="${esc(value)}"></label>`;
}
function isSharedField(name) { return (state.settings?.shared_keys || []).includes(name); }
function savedValue(model, name) {
  const shared = state.settings?.shared || {};
  const modelSettings = state.settings?.models?.[model.id] || {};
  if (isSharedField(name) && shared[name] !== undefined) return shared[name];
  if (!isSharedField(name) && modelSettings[name] !== undefined) return modelSettings[name];
  if (modelSettings[name] !== undefined) return modelSettings[name];
  return (model.defaults || {})[name] ?? '';
}
function collectFormParams() {
  const params = {};
  const form = $('#modelForm');
  if (!form) return params;
  $$('input,select', form).forEach(el => params[el.name] = el.value);
  return params;
}
function applySettingsLocally(modelId, params) {
  if (!modelId || !params) return;
  state.settings.shared ||= {};
  state.settings.models ||= {};
  state.settings.models[modelId] ||= {};
  Object.entries(params).forEach(([key, value]) => {
    if (isSharedField(key)) state.settings.shared[key] = value;
    else state.settings.models[modelId][key] = value;
  });
}
async function saveSettingsFor(modelId, params, updateStatus=true) {
  if (!modelId || !params) return;
  applySettingsLocally(modelId, params);
  try {
    state.settings = await api('/api/settings', { method:'POST', body: JSON.stringify({ model: modelId, params }) });
    const status = $('#settingsStatus');
    if (updateStatus && status) status.textContent = `Saved settings to ${state.settings.path}.`;
  } catch (e) {
    const status = $('#settingsStatus');
    if (updateStatus && status) status.textContent = `Settings save failed: ${e.message}`;
  }
}
function scheduleSettingsSave() {
  const status = $('#settingsStatus');
  if (status) status.textContent = `Unsaved changes...`;
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => saveCurrentSettings(), 500);
}
async function saveCurrentSettings() {
  if (!state.activeModel || !$('#modelForm')) return;
  await saveSettingsFor(state.activeModel, collectFormParams(), true);
}
async function startModelJob(model, action) {
  const params = collectFormParams();
  await saveCurrentSettings();
  const job = await api('/api/jobs', { method:'POST', body: JSON.stringify({ model, action, params }) }); state.currentJob = job.id; pollJobs();
}
async function pollJobs() {
  if (!$('#jobBar')) return;
  try {
    const jobs = await api('/api/jobs'); const job = jobs.find(j => ['queued','running'].includes(j.status)) || (state.currentJob && jobs.find(j => j.id === state.currentJob)) || jobs[0];
    if (!job) return;
    state.currentJob = job.id; $('#jobBar').style.width = pct(job.progress); $('#jobStatus').textContent = `${job.label} | ${job.status} | ${pct(job.progress)}`;
    const logEl = $('#jobLog');
    if (logEl) logEl.textContent = (job.logs || []).slice(-180).join('\n');
    const details = $('#jobLogDetails');
    const summary = $('#jobLogSummary');
    if (details && summary) {
      summary.textContent = `${job.status === 'running' ? 'Live' : 'Terminal'} logs (${(job.logs || []).length} lines)`;
      if (['queued','running'].includes(job.status)) {
        details.open = true;
        delete details.dataset.collapsedJobId;
      } else if (details.dataset.collapsedJobId !== job.id) {
        details.open = false;
        details.dataset.collapsedJobId = job.id;
      }
    }
    const linksEl = $('#jobLinks');
    if (linksEl) linksEl.innerHTML = Object.entries(job.links||{}).map(([k,v]) => `<a class="button" href="${esc(v)}" target="_blank">Open ${esc(k)}</a>`).join(' ');
    mergeCompletedJobGraphs(jobs);
    renderStoredModelGraphs(state.activeModel);
  } catch (e) { if ($('#jobStatus')) $('#jobStatus').textContent = e.message; }
}
function mergeCompletedJobGraphs(jobs) {
  for (const completed of [...jobs].reverse()) {
    if (completed.status !== 'completed' || !completed.model_id) continue;
    const graphs = Object.fromEntries(Object.entries(completed.auto_graphs || {}).filter(([, url]) => Boolean(url)));
    if (!Object.keys(graphs).length) continue;
    state.modelGraphs[completed.model_id] ||= {};
    state.modelGraphVersions[completed.model_id] ||= {};
    Object.assign(state.modelGraphs[completed.model_id], graphs);
    Object.keys(graphs).forEach(name => state.modelGraphVersions[completed.model_id][name] = completed.id);
  }
}
function renderStoredModelGraphs(modelId) {
  const graphsEl = $('#jobGraphs');
  if (!graphsEl || !modelId) return;
  const graphs = state.modelGraphs[modelId] || {};
  const signature = JSON.stringify([graphs, state.modelGraphVersions[modelId] || {}]);
  if (graphsEl.dataset.graphSignature === signature && graphsEl.dataset.graphModel === modelId) return;
  graphsEl.dataset.graphSignature = signature;
  graphsEl.dataset.graphModel = modelId;
  graphsEl.innerHTML = Object.keys(graphs).length
    ? `<div class="retained-graphs-head"><span>Saved results for ${esc(modelDisplayName(modelId))}</span><span class="muted">Separate actions keep previously generated graph types.</span></div>${jobGraphsHtml(graphs)}`
    : '<p class="muted">No completed model or simulation graphs for this model yet.</p>';
  if (Object.keys(graphs).length) hydrateJobGraphs(graphsEl);
}
function modelDisplayName(modelId) {
  return state.registry.find(model => model.id === modelId)?.label || modelId;
}
function jobGraphsHtml(graphs) {
  const entries = Object.entries(graphs || {}).filter(([, url]) => Boolean(url));
  if (!entries.length) return '';
  const chart = (key, label, extra='') => `<section class="integrated-chart-shell" data-chart-shell="${key}"><div class="integrated-chart-head"><strong>${label}</strong><div class="graph-controls"><button type="button" data-chart-action="collapse">Collapse</button><button type="button" data-chart-action="expand">Expand</button></div></div><div class="integrated-chart ${extra}"><div class="integrated-chart-loading">Loading ${label.toLowerCase()}...</div><canvas data-chart="${key}"></canvas><div class="integrated-point-tooltip"></div></div></section>`;
  return entries.map(([name, url]) => `<section class="job-graph panel" data-report-url="${esc(url)}" data-report-name="${esc(name)}"><div class="job-graph-head"><h3>${esc(name)} graph</h3><div class="graph-controls"><button type="button" data-graph-action="collapse">Collapse all</button><button type="button" data-graph-action="expand">Expand all</button><a class="button" href="${esc(url)}" target="_blank">Open full page</a></div></div><div class="job-graph-body"><p class="muted graph-status">Loading compact graph data...</p><div class="integrated-summary"></div>${chart('price','Price + trades','price')}${chart('middle','Model activity')}${chart('equity','Long-only vs long + short')}${chart('baseline','Models vs market baselines')}</div></section>`).join('');
}
async function hydrateJobGraphs(root) {
  for (const section of $$('.job-graph[data-report-url]', root)) {
    bindJobGraphControls(section);
    bindIndividualChartControls(section);
    const status = $('.graph-status', section);
    try {
      const url = section.dataset.reportUrl;
      const data = await api('/api/report-payload?compact=1&url=' + encodeURIComponent(url));
      section._graphState = buildIntegratedGraphState(data.type, data.payload);
      if (data.type === 'simulation') {
        $('strong', $('[data-chart-shell="middle"]', section)).textContent = 'Active invested dollars';
        $('strong', $('[data-chart-shell="equity"]', section)).textContent = 'Long-only vs long + short equity';
        $('[data-chart-shell="baseline"]', section).hidden = false;
      } else {
        $('strong', $('[data-chart-shell="middle"]', section)).textContent = 'Model probability';
        $('strong', $('[data-chart-shell="equity"]', section)).textContent = 'Model vs baseline equity';
        $('[data-chart-shell="baseline"]', section).hidden = true;
      }
      status.textContent = `${data.type} data loaded. Charts render independently; wheel to zoom, drag to pan, double-click to reset.`;
      drawIntegratedGraphProgressive(section);
      attachIntegratedGraphEvents(section);
    } catch (e) {
      status.textContent = `Could not load graph: ${e.message}`;
    }
  }
}
function bindIndividualChartControls(section) {
  for (const shell of $$('.integrated-chart-shell', section)) {
    const collapse = $('[data-chart-action="collapse"]', shell);
    const expand = $('[data-chart-action="expand"]', shell);
    collapse.onclick = () => {
      if (shell.classList.contains('is-expanded')) {
        shell.classList.remove('is-expanded');
        document.body.classList.remove('graph-overlay-open');
        expand.textContent = 'Expand';
      }
      shell.classList.toggle('is-collapsed');
      collapse.textContent = shell.classList.contains('is-collapsed') ? 'Show' : 'Collapse';
      if (!shell.classList.contains('is-collapsed')) requestAnimationFrame(() => drawIntegratedGraph(section, shell.dataset.chartShell));
    };
    expand.onclick = () => {
      shell.classList.remove('is-collapsed');
      collapse.textContent = 'Collapse';
      const expanded = shell.classList.toggle('is-expanded');
      document.body.classList.toggle('graph-overlay-open', expanded);
      expand.textContent = expanded ? 'Unexpand' : 'Expand';
      requestAnimationFrame(() => drawIntegratedGraph(section, shell.dataset.chartShell));
    };
  }
}
function bindJobGraphControls(section) {
  const collapse = $('[data-graph-action="collapse"]', section);
  const expand = $('[data-graph-action="expand"]', section);
  collapse.onclick = () => {
    if (section.classList.contains('is-expanded')) {
      section.classList.remove('is-expanded');
      document.body.classList.remove('graph-overlay-open');
      expand.textContent = 'Expand';
    }
    section.classList.toggle('is-collapsed');
    collapse.textContent = section.classList.contains('is-collapsed') ? 'Show' : 'Collapse';
    if (!section.classList.contains('is-collapsed')) requestAnimationFrame(() => scheduleIntegratedGraph(section));
  };
  expand.onclick = () => {
    section.classList.remove('is-collapsed');
    collapse.textContent = 'Collapse';
    const expanded = section.classList.toggle('is-expanded');
    document.body.classList.toggle('graph-overlay-open', expanded);
    expand.textContent = expanded ? 'Unexpand' : 'Expand';
    requestAnimationFrame(() => scheduleIntegratedGraph(section));
  };
  if (!document.body.dataset.graphEscapeBound) {
    document.body.dataset.graphEscapeBound = '1';
    document.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      const expandedChart = $('.integrated-chart-shell.is-expanded');
      if (expandedChart) {
        expandedChart.classList.remove('is-expanded');
        $('[data-chart-action="expand"]', expandedChart).textContent = 'Expand';
        document.body.classList.remove('graph-overlay-open');
        const section = expandedChart.closest('.job-graph');
        requestAnimationFrame(() => drawIntegratedGraph(section, expandedChart.dataset.chartShell));
        return;
      }
      const expanded = $('.job-graph.is-expanded');
      if (!expanded) return;
      expanded.classList.remove('is-expanded');
      $('[data-graph-action="expand"]', expanded).textContent = 'Expand';
      document.body.classList.remove('graph-overlay-open');
      requestAnimationFrame(() => scheduleIntegratedGraph(expanded));
    });
  }
}
function buildIntegratedGraphState(type, payload) {
  const state = { type, payload, xRange: [0, 1], fullRange: [0, 1] };
  if (type === 'model') {
    const baseFrame = payload.base_frame || Object.keys(payload.candles || {})[0];
    for (const frame of Object.values(payload.candles || {})) frame.t = (frame.x || []).map(Date.parse);
    payload.predictions.t = (payload.predictions.x || []).map(Date.parse);
    payload.equity.t = (payload.equity.x || []).map(Date.parse);
    for (const group of Object.values(payload.markers || {})) group.t = (group.x || []).map(Date.parse);
    payload.test_start_t = payload.test_start ? Date.parse(payload.test_start) : null;
    const t = payload.candles[baseFrame]?.t || payload.predictions.t || [];
    state.baseFrame = baseFrame;
    state.fullRange = [t[0] || 0, t[t.length - 1] || 1];
  } else {
    payload.candles.t = (payload.candles.t || []).map(Date.parse);
    payload.trades.entry_t = (payload.trades.entry_t || []).map(Date.parse);
    payload.trades.exit_t = (payload.trades.exit_t || []).map(Date.parse);
    payload.comparison.t = (payload.comparison.t || []).map(Date.parse);
    const t = payload.candles.t || [];
    state.fullRange = [t[0] || 0, t[t.length - 1] || 1];
  }
  state.xRange = [...state.fullRange];
  return state;
}
function attachIntegratedGraphEvents(section) {
  if (section.dataset.bound === '1') return;
  section.dataset.bound = '1';
  let drag = null;
  for (const canvas of $$('canvas', section)) {
    canvas.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      const st = section._graphState;
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (ev.clientX - rect.left) / Math.max(1, rect.width)));
      const anchor = st.xRange[0] + ratio * (st.xRange[1] - st.xRange[0]);
      const factor = ev.deltaY < 0 ? 0.75 : 1.35;
      const left = anchor - (anchor - st.xRange[0]) * factor;
      const right = anchor + (st.xRange[1] - anchor) * factor;
      setIntegratedRange(st, left, right);
      scheduleIntegratedGraph(section);
    }, { passive: false });
    canvas.addEventListener('mousedown', (ev) => { drag = { x: ev.clientX, range: [...section._graphState.xRange] }; });
    canvas.addEventListener('click', (ev) => showIntegratedPointTooltip(section, canvas, ev));
    canvas.addEventListener('dblclick', () => { section._graphState.xRange = [...section._graphState.fullRange]; scheduleIntegratedGraph(section); });
  }
  window.addEventListener('mouseup', () => { drag = null; });
  window.addEventListener('mousemove', (ev) => {
    if (!drag) return;
    const st = section._graphState;
    const rect = section.getBoundingClientRect();
    const span = drag.range[1] - drag.range[0];
    const shift = -((ev.clientX - drag.x) / Math.max(1, rect.width)) * span;
    setIntegratedRange(st, drag.range[0] + shift, drag.range[1] + shift);
    scheduleIntegratedGraph(section);
  });
  window.addEventListener('resize', () => scheduleIntegratedGraph(section));
}
function nearestIntegratedIndex(times, target) {
  const index = lowerBound(times, target);
  if (index <= 0) return 0;
  if (index >= times.length) return times.length - 1;
  return Math.abs(times[index] - target) < Math.abs(times[index - 1] - target) ? index : index - 1;
}
function integratedPointRows(st, chartName, target) {
  const r = st.payload;
  const money = value => '$' + Number(value || 0).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2});
  const number = value => Number(value || 0).toLocaleString(undefined, {maximumFractionDigits:6});
  let times = [];
  let rows = [];
  if (st.type === 'simulation') {
    if (chartName === 'price') {
      times = r.candles.t; const i = nearestIntegratedIndex(times, target);
      rows = [`time: ${new Date(times[i]).toISOString().slice(0,16).replace('T',' ')} UTC`, `open: ${number(r.candles.open[i])}`, `high: ${number(r.candles.high[i])}`, `low: ${number(r.candles.low[i])}`, `close: ${number(r.candles.close[i])}`];
    } else {
      times = r.comparison.t; const i = nearestIntegratedIndex(times, target);
      const source = chartName === 'middle' ? r.comparison.active : r.comparison.equity;
      const keys = chartName === 'middle'
        ? ['model','model_long_short_long','model_long_short_short'].filter(key => source?.[key])
        : chartName === 'equity'
          ? ['model','model_long_short'].filter(key => source?.[key])
          : preferredSimulationKeys(source);
      const startingCash = Number(r.comparison.starting_cash || r.summary.starting_cash || 0);
      rows = [`time: ${new Date(times[i]).toISOString().slice(0,16).replace('T',' ')} UTC`];
      for (const key of keys) {
        const value = Number(source[key][i] || 0);
        const label = r.comparison.labels?.[key] || key;
        rows.push(`${label}: ${money(value)}`);
        if (chartName !== 'middle' && key.startsWith('model')) rows.push(`  P/L: ${money(value - startingCash)}`);
      }
    }
  } else if (chartName === 'price') {
    const frame = chooseIntegratedCandleFrame(st, {width:500});
    times = frame.t; const i = nearestIntegratedIndex(times, target);
    rows = [`time: ${new Date(times[i]).toISOString().slice(0,16).replace('T',' ')} UTC`, `open: ${number(frame.open[i])}`, `high: ${number(frame.high[i])}`, `low: ${number(frame.low[i])}`, `close: ${number(frame.close[i])}`];
  } else if (chartName === 'middle') {
    times = r.predictions.t; const i = nearestIntegratedIndex(times, target);
    rows = [`time: ${new Date(times[i]).toISOString().slice(0,16).replace('T',' ')} UTC`, `probability up: ${Number(r.predictions.prob_up[i] || 0).toFixed(6)}`];
  } else {
    times = r.equity.t; const i = nearestIntegratedIndex(times, target);
    rows = [`time: ${new Date(times[i]).toISOString().slice(0,16).replace('T',' ')} UTC`, `model: ${money(r.equity.model[i])}`, `buy and hold: ${money(r.equity.buy_hold[i])}`];
    if (r.equity.ma_baseline) rows.push(`MA baseline: ${money(r.equity.ma_baseline[i])}`);
  }
  return rows;
}
function showIntegratedPointTooltip(section, canvas, event) {
  const chart = canvas.closest('.integrated-chart');
  const tooltip = $('.integrated-point-tooltip', chart);
  const rect = canvas.getBoundingClientRect();
  const st = section._graphState;
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width)));
  const target = st.xRange[0] + ratio * (st.xRange[1] - st.xRange[0]);
  const rows = integratedPointRows(st, canvas.dataset.chart, target);
  tooltip.textContent = rows.join('\\n');
  tooltip.style.left = Math.max(8, Math.min(rect.width - 230, event.clientX - rect.left + 12)) + 'px';
  tooltip.style.top = Math.max(8, event.clientY - rect.top + 12) + 'px';
  tooltip.classList.add('is-visible');
}
function scheduleIntegratedGraph(section) {
  const st = section._graphState;
  if (!st || st.redrawPending) return;
  st.redrawPending = true;
  requestAnimationFrame(() => {
    st.redrawPending = false;
    drawIntegratedGraphProgressive(section);
  });
}
function setIntegratedRange(st, left, right) {
  const full = st.fullRange;
  const minSpan = 60 * 1000;
  let span = Math.max(minSpan, right - left);
  if (span > full[1] - full[0]) span = full[1] - full[0];
  if (left < full[0]) { left = full[0]; right = left + span; }
  if (right > full[1]) { right = full[1]; left = right - span; }
  st.xRange = [left, right];
}
function drawIntegratedGraphProgressive(section) {
  const st = section._graphState;
  if (!st) return;
  st.drawGeneration = (st.drawGeneration || 0) + 1;
  const generation = st.drawGeneration;
  const stages = st.type === 'simulation'
    ? ['price', 'middle', 'equity', 'baseline']
    : ['price', 'middle', 'equity'];
  const drawStage = (index) => {
    if (generation !== st.drawGeneration || index >= stages.length) return;
    drawIntegratedGraph(section, stages[index]);
    requestAnimationFrame(() => drawStage(index + 1));
  };
  drawStage(0);
}
function drawIntegratedGraph(section, chartName=null) {
  const st = section._graphState;
  if (!st) return;
  if (st.type === 'model') drawIntegratedModel(section, st, chartName);
  else drawIntegratedSimulation(section, st, chartName);
  if (chartName) {
    const chart = $(`[data-chart="${chartName}"]`, section)?.closest('.integrated-chart');
    if (chart) chart.classList.add('is-ready');
  }
}
function canvasCtx(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: rect.width, height: rect.height, plot: { left: 58, top: 28, width: rect.width - 82, height: rect.height - 70 } };
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
function idxVisible(times, range) {
  if (!times.length) return [];
  const start = Math.max(0, lowerBound(times, range[0]) - 2);
  const end = Math.min(times.length, upperBound(times, range[1]) + 2);
  const out = [];
  for (let i = start; i < end; i++) out.push(i);
  return out;
}
function xScale(t, plot, range) { return plot.left + ((t - range[0]) / Math.max(1, range[1] - range[0])) * plot.width; }
function yScale(v, lo, hi, plot) { return plot.top + plot.height - ((v - lo) / Math.max(1e-12, hi - lo)) * plot.height; }
function chartAxes(ctx, plot, lo, hi, title, range) {
  ctx.clearRect(0, 0, plot.left + plot.width + 24, plot.top + plot.height + 44);
  ctx.strokeStyle = '#e6ddcb'; ctx.fillStyle = '#6b6256'; ctx.lineWidth = 1; ctx.font = '12px Courier New';
  for (let i = 0; i <= 4; i++) {
    const y = plot.top + plot.height * i / 4;
    ctx.beginPath(); ctx.moveTo(plot.left, y); ctx.lineTo(plot.left + plot.width, y); ctx.stroke();
    ctx.fillText((hi - (hi - lo) * i / 4).toFixed(hi <= 1.5 ? 2 : 2), 6, y + 4);
  }
  for (let i = 0; i <= 5; i++) {
    const x = plot.left + plot.width * i / 5;
    const t = new Date(range[0] + (range[1] - range[0]) * i / 5).toISOString().slice(5, 16).replace('T', ' ');
    ctx.beginPath(); ctx.moveTo(x, plot.top); ctx.lineTo(x, plot.top + plot.height); ctx.stroke();
    ctx.fillText(t, x - 26, plot.top + plot.height + 18);
  }
  ctx.fillStyle = '#15120d'; ctx.font = '14px Georgia'; ctx.fillText(title, plot.left, 18);
}
function drawStartingCashReference(ctx, plot, lo, hi, startingCash) {
  if (!Number.isFinite(startingCash) || startingCash < lo || startingCash > hi) return;
  const y = yScale(startingCash, lo, hi, plot);
  ctx.save();
  ctx.strokeStyle = '#7d5a17'; ctx.fillStyle = '#7d5a17'; ctx.lineWidth = 1.2; ctx.setLineDash([5, 5]);
  ctx.beginPath(); ctx.moveTo(plot.left, y); ctx.lineTo(plot.left + plot.width, y); ctx.stroke();
  ctx.setLineDash([]); ctx.font = '11px Courier New'; ctx.fillText(`start $${startingCash.toFixed(2)}`, plot.left + plot.width - 94, y - 5);
  ctx.restore();
}
function integratedDetailTarget(plot, range, times) {
  const visibleSpan = Math.max(1, range[1] - range[0]);
  const fullSpan = Math.max(visibleSpan, (times[times.length - 1] || range[1]) - (times[0] || range[0]));
  const zoom = Math.max(1, fullSpan / visibleSpan);
  const zoomBoost = Math.pow(Math.min(512, zoom), 0.45);
  const base = Math.max(70, Math.round(plot.width * 0.12));
  return Math.max(60, Math.min(2500, Math.round(base * zoomBoost)));
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
    let minIdx = indexes[start], maxIdx = indexes[start];
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
function drawLineSeries(ctx, times, values, indexes, plot, lo, hi, range, color, width=2, dash=[], stepped=false) {
  if (!indexes.length) return;
  const sampled = decimateIndexesByMinMax(indexes, values, integratedDetailTarget(plot, range, times));
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.beginPath();
  let started = false;
  for (let sampleIndex = 0; sampleIndex < sampled.length; sampleIndex++) {
    const i = sampled[sampleIndex];
    const v = Number(values[i]);
    if (!Number.isFinite(v)) continue;
    const x = xScale(times[i], plot, range); const y = yScale(v, lo, hi, plot);
    if (!started) { ctx.moveTo(x, y); started = true; }
    else if (stepped) {
      const previous = sampled[sampleIndex - 1];
      const previousY = yScale(Number(values[previous]), lo, hi, plot);
      ctx.lineTo(x, previousY);
      ctx.lineTo(x, y);
    } else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}
function yRange(valuesBySeries, indexes, includeZero=false) {
  let lo = includeZero ? 0 : Infinity, hi = includeZero ? 1 : -Infinity;
  for (const values of valuesBySeries) for (const i of indexes) {
    const v = Number(values[i]); if (!Number.isFinite(v)) continue;
    lo = Math.min(lo, v); hi = Math.max(hi, v);
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) { lo = 0; hi = 1; }
  const pad = (hi - lo) * 0.08 || 1;
  return [includeZero ? 0 : lo - pad, hi + pad];
}
function drawLegendMini(ctx, items, x, y) {
  ctx.font = '12px Courier New'; ctx.textBaseline = 'top';
  let offset = 0;
  for (const item of items) {
    ctx.fillStyle = item.color; ctx.fillRect(x + offset, y, 10, 10);
    ctx.fillStyle = '#15120d'; ctx.fillText(item.label, x + offset + 14, y - 1);
    offset += item.width || 100;
  }
}
function drawCandles(ctx, c, indexes, plot, lo, hi, range) {
  const width = Math.max(1, Math.min(9, plot.width / Math.max(1, indexes.length) * 0.7));
  if (width < 2.2) return drawLineSeries(ctx, c.t, c.close, indexes, plot, lo, hi, range, '#15120d', 1.5);
  for (const i of indexes) {
    const x = xScale(c.t[i], plot, range);
    const yo = yScale(c.open[i], lo, hi, plot), yc = yScale(c.close[i], lo, hi, plot);
    const yh = yScale(c.high[i], lo, hi, plot), yl = yScale(c.low[i], lo, hi, plot);
    const up = c.close[i] >= c.open[i];
    ctx.strokeStyle = up ? '#16864e' : '#b23834'; ctx.fillStyle = up ? '#d9f0df' : '#f5d9d5';
    ctx.beginPath(); ctx.moveTo(x, yh); ctx.lineTo(x, yl); ctx.stroke();
    ctx.fillRect(x - width / 2, Math.min(yo, yc), width, Math.max(1, Math.abs(yc - yo)));
    ctx.strokeRect(x - width / 2, Math.min(yo, yc), width, Math.max(1, Math.abs(yc - yo)));
  }
}
function chooseIntegratedCandleFrame(st, plot) {
  const targetBars = Math.max(120, Math.min(500, Math.floor(plot.width / 2.5)));
  const frames = Object.entries(st.payload.candles || {})
    .filter(([, frame]) => frame.t?.length)
    .sort((a, b) => {
      const aSpan = a[1].t.length > 1 ? a[1].t[a[1].t.length - 1] - a[1].t[0] : Infinity;
      const bSpan = b[1].t.length > 1 ? b[1].t[b[1].t.length - 1] - b[1].t[0] : Infinity;
      return (aSpan / Math.max(1, a[1].t.length - 1)) - (bSpan / Math.max(1, b[1].t.length - 1));
    });
  let selected = frames[frames.length - 1]?.[1] || st.payload.candles[st.baseFrame];
  for (const [, frame] of frames) {
    const visible = upperBound(frame.t, st.xRange[1]) - lowerBound(frame.t, st.xRange[0]);
    if (visible <= targetBars) {
      selected = frame;
      break;
    }
  }
  return selected;
}
function drawIntegratedTestBoundary(ctx, plot, range, testStart) {
  if (!Number.isFinite(testStart) || testStart < range[0] || testStart > range[1]) return;
  const x = xScale(testStart, plot, range);
  ctx.save();
  ctx.strokeStyle = '#7d5a17';
  ctx.fillStyle = '#7d5a17';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([7, 5]);
  ctx.beginPath(); ctx.moveTo(x, plot.top); ctx.lineTo(x, plot.top + plot.height); ctx.stroke();
  ctx.setLineDash([]);
  ctx.font = '11px Courier New';
  ctx.textAlign = x > plot.left + plot.width - 90 ? 'right' : 'left';
  ctx.fillText('test starts', x + (ctx.textAlign === 'right' ? -5 : 5), plot.top + plot.height - 6);
  ctx.restore();
}
function drawModelMarkers(ctx, markers, plot, lo, hi, range) {
  const groups = [
    ['buy_win', '#16864e', 'buy win'], ['buy_loss', '#b23834', 'buy loss'],
    ['sell_good', '#245ea8', 'sell good'], ['sell_bad', '#d64a88', 'sell bad'], ['no_loss', '#c77519', 'flat']
  ];
  const fullSpan = Math.max(1, Math.max(...groups.map(([key]) => {
    const t = markers[key]?.t || [];
    return t.length > 1 ? t[t.length - 1] - t[0] : 1;
  })));
  const zoom = Math.max(1, fullSpan / Math.max(1, range[1] - range[0]));
  const bucketPx = zoom < 2 ? 28 : zoom < 8 ? 20 : zoom < 32 ? 14 : 9;
  for (const [key, color] of groups) {
    const g = markers[key]; if (!g?.t?.length) continue;
    const ids = idxVisible(g.t, range);
    const spacing = plot.width / Math.max(1, ids.length);
    ctx.fillStyle = color; ctx.globalAlpha = 0.82;
    if (ids.length <= 400 && spacing >= 8) {
      for (const i of ids) {
        const x = xScale(g.t[i], plot, range); const y = yScale(g.close[i], lo, hi, plot);
        ctx.beginPath(); ctx.arc(x, y, 3.2, 0, Math.PI * 2); ctx.fill();
      }
    } else {
      const clusters = new Map();
      for (const i of ids) {
        const x = xScale(g.t[i], plot, range);
        const bucket = Math.floor((x - plot.left) / bucketPx);
        const current = clusters.get(bucket) || {x:0, y:0, count:0};
        current.x += x; current.y += yScale(g.close[i], lo, hi, plot); current.count += 1;
        clusters.set(bucket, current);
      }
      for (const cluster of clusters.values()) {
        const radius = Math.min(9, 3.5 + Math.log2(cluster.count + 1));
        ctx.beginPath(); ctx.arc(cluster.x / cluster.count, cluster.y / cluster.count, radius, 0, Math.PI * 2); ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }
  drawLegendMini(ctx, groups.map(([_, color, label]) => ({ color, label, width: 88 })), plot.left + 4, plot.top + 8);
}
function drawIntegratedModel(section, st, chartName=null) {
  const r = st.payload;
  $('.integrated-summary', section).innerHTML = [
    `timeline=full train+test`, `train=${r.summary.train_prediction_rows || 0}`, `test=${r.summary.test_prediction_rows || 0}`,
    `rows=${r.summary.prediction_rows}`, `buys=${r.summary.buy_predictions}`, `sells=${r.summary.sell_predictions}`,
    `net_profit=$${Number(r.summary.net_profit || 0).toFixed(2)}`, `threshold=${r.threshold}`, `exit=${r.exit_threshold}`
  ].map(x => `<span>${esc(x)}</span>`).join('');
  const price = $('[data-chart="price"]', section); const mid = $('[data-chart="middle"]', section); const eq = $('[data-chart="equity"]', section);
  if (!chartName || chartName === 'price') {
    let box = canvasCtx(price);
    const base = chooseIntegratedCandleFrame(st, box.plot);
    const ids = idxVisible(base.t, st.xRange);
    const [lo, hi] = yRange([base.low, base.high], ids);
    chartAxes(box.ctx, box.plot, lo, hi, 'Price + model predictions', st.xRange); drawCandles(box.ctx, base, ids, box.plot, lo, hi, st.xRange); drawModelMarkers(box.ctx, r.markers, box.plot, lo, hi, st.xRange); drawIntegratedTestBoundary(box.ctx, box.plot, st.xRange, r.test_start_t);
  }
  if (!chartName || chartName === 'middle') {
    const box = canvasCtx(mid); const ids = idxVisible(r.predictions.t, st.xRange);
    chartAxes(box.ctx, box.plot, 0, 1, 'Model probability', st.xRange); drawIntegratedTestBoundary(box.ctx, box.plot, st.xRange, r.test_start_t); drawLineSeries(box.ctx, r.predictions.t, r.predictions.prob_up, ids, box.plot, 0, 1, st.xRange, '#245ea8', 1.8);
    box.ctx.strokeStyle = '#164f45'; for (const v of [r.threshold, r.exit_threshold]) { const y = yScale(v, 0, 1, box.plot); box.ctx.beginPath(); box.ctx.moveTo(box.plot.left, y); box.ctx.lineTo(box.plot.left + box.plot.width, y); box.ctx.stroke(); }
  }
  if (!chartName || chartName === 'equity') {
    const box = canvasCtx(eq); const ids = idxVisible(r.equity.t, st.xRange);
    const eqSeries = [r.equity.model, r.equity.buy_hold]; if (r.equity.ma_baseline) eqSeries.push(r.equity.ma_baseline);
    const [lo, hi] = yRange(eqSeries, ids); chartAxes(box.ctx, box.plot, lo, hi, 'Full-timeline equity: train (in-sample) + test', st.xRange);
    drawLineSeries(box.ctx, r.equity.t, r.equity.model, ids, box.plot, lo, hi, st.xRange, '#164f45', 2.2);
    drawLineSeries(box.ctx, r.equity.t, r.equity.buy_hold, ids, box.plot, lo, hi, st.xRange, '#c77519', 2);
    if (r.equity.ma_baseline) drawLineSeries(box.ctx, r.equity.t, r.equity.ma_baseline, ids, box.plot, lo, hi, st.xRange, '#245ea8', 1.8);
    drawIntegratedTestBoundary(box.ctx, box.plot, st.xRange, r.test_start_t);
    drawLegendMini(box.ctx, [{color:'#164f45',label:'model'},{color:'#c77519',label:'buy hold'},{color:'#245ea8',label:'MA'}], box.plot.left + 4, box.plot.top + 8);
  }
}
function drawSimTrades(ctx, trades, plot, lo, hi, range) {
  const ids = [];
  for (let i = 0; i < trades.entry_t.length; i++) if (trades.exit_t[i] >= range[0] && trades.entry_t[i] <= range[1]) ids.push(i);
  const target = Math.min(500, integratedDetailTarget(plot, range, trades.entry_t || []));
  const step = Math.max(1, Math.ceil(ids.length / target));
  for (let n = 0; n < ids.length; n += step) {
    const i = ids[n]; const win = Number(trades.net_profit[i]) > 0; const isShort = (trades.side?.[i] || 'long') === 'short';
    const radius = step > 1 ? 5 : 3.5;
    ctx.fillStyle = isShort ? (win ? '#7b4ab3' : '#e0ad16') : (win ? '#16864e' : '#b23834');
    let x = xScale(trades.entry_t[i], plot, range), y = yScale(trades.entry_price[i], lo, hi, plot);
    ctx.beginPath();
    if (isShort) { ctx.moveTo(x, y + radius); ctx.lineTo(x + radius, y - radius); ctx.lineTo(x - radius, y - radius); }
    else { ctx.moveTo(x, y - radius); ctx.lineTo(x + radius, y + radius); ctx.lineTo(x - radius, y + radius); }
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = isShort ? (win ? '#7b4ab3' : '#e0ad16') : (win ? '#245ea8' : '#d64a88');
    x = xScale(trades.exit_t[i], plot, range); y = yScale(trades.exit_price[i], lo, hi, plot);
    ctx.beginPath();
    if (isShort) { ctx.moveTo(x, y - radius); ctx.lineTo(x + radius, y + radius); ctx.lineTo(x - radius, y + radius); }
    else { ctx.moveTo(x, y + radius); ctx.lineTo(x + radius, y - radius); ctx.lineTo(x - radius, y - radius); }
    ctx.closePath(); ctx.fill();
  }
  drawLegendMini(ctx, [{color:'#16864e',label:'long win'},{color:'#b23834',label:'long loss'},{color:'#7b4ab3',label:'short win'},{color:'#e0ad16',label:'short loss'}], plot.left + 4, plot.top + 8);
}
function preferredSimulationKeys(series) {
  const preferred = ['model', 'model_long_short', 'buy_hold'];
  return [...preferred.filter(key => series?.[key]), ...Object.keys(series || {}).filter(key => !preferred.includes(key))];
}
function drawIntegratedSimulation(section, st, chartName=null) {
  const r = st.payload; const c = r.candles;
  const longOnlyEquity = r.comparison.equity?.model || [];
  const longShortEquity = r.comparison.equity?.model_long_short || [];
  const finalLongOnly = Number(longOnlyEquity.at(-1) || 0);
  const finalLongShort = Number(longShortEquity.at(-1) || 0);
  const comparisonCounts = r.comparison.comparison_trade_counts || {};
  const shortExposure = r.comparison.active?.model_long_short_short || [];
  const shortExposureRate = shortExposure.length
    ? shortExposure.filter(value => Number(value) > 0).length / shortExposure.length
    : 0;
  $('.integrated-summary', section).innerHTML = [
    `long_only_trades=${r.summary.trade_count}`,
    `long_short_trades=${comparisonCounts.total || 0}`,
    `long_short_longs=${comparisonCounts.long || 0}`,
    `long_short_shorts=${comparisonCounts.short || 0}`,
    `short_exposure_time=${(shortExposureRate * 100).toFixed(1)}%`,
    `long_only_end=$${finalLongOnly.toFixed(2)}`,
    `long+short_end=$${finalLongShort.toFixed(2)}`,
    `difference=$${(finalLongShort - finalLongOnly).toFixed(2)}`
  ].map(x => `<span>${esc(x)}</span>`).join('');
  const price = $('[data-chart="price"]', section); const mid = $('[data-chart="middle"]', section); const eq = $('[data-chart="equity"]', section); const baseline = $('[data-chart="baseline"]', section);
  if (!chartName || chartName === 'price') {
    const box = canvasCtx(price); const ids = idxVisible(c.t, st.xRange);
    const [lo, hi] = yRange([c.low, c.high], ids);
    chartAxes(box.ctx, box.plot, lo, hi, 'Price + long-only simulation trades', st.xRange); drawCandles(box.ctx, c, ids, box.plot, lo, hi, st.xRange); drawSimTrades(box.ctx, r.trades, box.plot, lo, hi, st.xRange);
  }
  if (!chartName || chartName === 'middle') {
    const box = canvasCtx(mid); const ids = idxVisible(r.comparison.t, st.xRange);
    const active = r.comparison.active; const activeKeys = ['model', 'model_long_short_long', 'model_long_short_short'].filter(key => active?.[key]);
    const [lo, hi] = yRange(activeKeys.map(k => active[k]), ids, true); chartAxes(box.ctx, box.plot, lo, hi, 'Active invested dollars by position side', st.xRange);
    const drawKeys = [...activeKeys.filter(key => key !== 'model'), ...activeKeys.filter(key => key === 'model')];
    drawKeys.forEach((k, n) => drawLineSeries(box.ctx, r.comparison.t, active[k], ids, box.plot, lo, hi, st.xRange, simulationSeriesColor(k, n), k === 'model' ? 2.6 : 2.2, k === 'model' ? [8, 4] : [], true));
    drawLegendMini(box.ctx, activeKeys.map((k, n) => ({ color:simulationSeriesColor(k, n), label:(r.comparison.labels?.[k] || k) + (k === 'model' ? ' (dashed)' : ''), width:175 })), box.plot.left + 4, box.plot.top + 8);
  }
  if (!chartName || chartName === 'equity') {
    const box = canvasCtx(eq); const ids = idxVisible(r.comparison.t, st.xRange);
    const equity = r.comparison.equity; const eqKeys = ['model', 'model_long_short'].filter(key => equity?.[key]);
    const startingCash = Number(r.comparison.starting_cash || r.summary.starting_cash || 0);
    const [lo, hi] = yRange(eqKeys.map(k => equity[k]), ids); chartAxes(box.ctx, box.plot, lo, hi, `Model equity; final P/L long ${formatSignedMoney(finalLongOnly - startingCash)}, long+short ${formatSignedMoney(finalLongShort - startingCash)}`, st.xRange);
    drawStartingCashReference(box.ctx, box.plot, lo, hi, startingCash);
    const drawKeys = [...eqKeys.filter(key => key !== 'model'), ...eqKeys.filter(key => key === 'model')];
    drawKeys.forEach((k, n) => drawLineSeries(box.ctx, r.comparison.t, equity[k], ids, box.plot, lo, hi, st.xRange, simulationSeriesColor(k, n), k === 'model' ? 2.8 : 2, k === 'model' ? [8, 4] : []));
    drawLegendMini(box.ctx, eqKeys.map((k, n) => ({ color:simulationSeriesColor(k, n), label:(r.comparison.labels?.[k] || k) + (k === 'model' ? ' (dashed)' : ''), width:175 })), box.plot.left + 4, box.plot.top + 8);
  }
  if (!chartName || chartName === 'baseline') {
    const box = canvasCtx(baseline); const ids = idxVisible(r.comparison.t, st.xRange);
    const equity = r.comparison.equity; const eqKeys = preferredSimulationKeys(equity).slice(0, 6);
    const [lo, hi] = yRange(eqKeys.map(k => equity[k]), ids); chartAxes(box.ctx, box.plot, lo, hi, 'Models vs buy-and-hold and strategy baselines', st.xRange);
    drawStartingCashReference(box.ctx, box.plot, lo, hi, Number(r.comparison.starting_cash || r.summary.starting_cash || 0));
    const drawKeys = [...eqKeys.filter(key => key !== 'model'), ...eqKeys.filter(key => key === 'model')];
    drawKeys.forEach((k, n) => drawLineSeries(box.ctx, r.comparison.t, equity[k], ids, box.plot, lo, hi, st.xRange, simulationSeriesColor(k, n), k === 'model' ? 2.8 : 2, k === 'model' ? [8, 4] : []));
    drawLegendMini(box.ctx, eqKeys.map((k, n) => ({ color:simulationSeriesColor(k, n), label:(r.comparison.labels?.[k] || k) + (k === 'model' ? ' (dashed)' : ''), width:175 })), box.plot.left + 4, box.plot.top + 8);
  }
}
function formatSignedMoney(value) {
  const number = Number(value || 0);
  return `${number >= 0 ? '+' : '-'}$${Math.abs(number).toFixed(2)}`;
}
function simulationSeriesColor(key, index) {
  if (key === 'model') return '#164f45';
  if (key === 'model_long_short') return '#7b4ab3';
  if (key === 'model_long_short_long') return '#7b4ab3';
  if (key === 'model_long_short_short') return '#b23834';
  if (key === 'buy_hold') return '#c77519';
  return ['#245ea8','#b23834','#7a5224','#15120d'][index % 4];
}
async function renderReports() {
  $('#app').innerHTML = `<section class="panel"><h2>Reports</h2><p class="muted">Generated HTML files in data/reports.</p><div id="reports" class="grid"></div></section>`;
  const reports = await api('/api/reports');
  $('#reports').innerHTML = reports.map(r => `<a class="card" href="${esc(r.url)}" target="_blank"><h3>${esc(r.label)}</h3><p class="muted">${esc(r.complexity_group || 'Other')} · open generated visualization</p></a>`).join('') || '<p class="muted">No report HTML files found.</p>';
}
function liveSaved(name, fallback='') {
  const live = state.settings?.live || {};
  return live[name] !== undefined ? live[name] : fallback;
}
function liveField(name, label, fallback='', type='text', step='any') {
  return `<label>${esc(label)}<input name="${esc(name)}" type="${esc(type)}" step="${esc(step)}" value="${esc(liveSaved(name, fallback))}"></label>`;
}
function liveSelect(name, label, options, fallback='') {
  const value = liveSaved(name, fallback || options[0] || '');
  return `<label>${esc(label)}<select name="${esc(name)}">${options.map(o => `<option value="${esc(o)}" ${o===value?'selected':''}>${esc(o)}</option>`).join('')}</select></label>`;
}
function liveSection(title, fields) {
  return `<h3>${esc(title)}</h3><div class="formgrid">${fields.join('')}</div>`;
}
function setLiveValue(name, value) {
  const el = $(`#liveConfigForm [name="${name}"]`);
  if (el) el.value = value;
}
function applyLiveModelPreset(modelType) {
  const model = String(modelType || '').toLowerCase();
  setLiveValue('LIVE_TRAIN_MODEL_TYPE', model);
  if (model === 'lstm') {
    setLiveValue('TRAINER_ENV', 'env/trainers/lstm_torch.env');
    setLiveValue('LIVE_TRAIN_BACKEND', 'torch');
    setLiveValue('LIVE_TRAIN_LOOKBACK', '70');
    setLiveValue('LIVE_TRAIN_SEQUENCE_FEATURE_SET', 'technical');
    setLiveValue('NN_HIDDEN_LAYERS', '32');
  } else if (model === 'transformer') {
    setLiveValue('TRAINER_ENV', 'env/trainers/transformer_torch.env');
    setLiveValue('LIVE_TRAIN_BACKEND', 'torch');
    setLiveValue('LIVE_TRAIN_LOOKBACK', '70');
    setLiveValue('LIVE_TRAIN_SEQUENCE_FEATURE_SET', 'technical');
    setLiveValue('NN_TRANSFORMER_D_MODEL', '64');
    setLiveValue('NN_TRANSFORMER_HEADS', '4');
    setLiveValue('NN_TRANSFORMER_LAYERS', '2');
    setLiveValue('NN_TRANSFORMER_FF_DIM', '128');
    setLiveValue('NN_TRANSFORMER_DROPOUT', '0.1');
    setLiveValue('NN_HIDDEN_LAYERS', '32');
  } else if (model === 'gru') {
    setLiveValue('TRAINER_ENV', 'env/trainers/gru_torch.env');
    setLiveValue('LIVE_TRAIN_BACKEND', 'torch');
    setLiveValue('LIVE_TRAIN_LOOKBACK', '70');
    setLiveValue('LIVE_TRAIN_SEQUENCE_FEATURE_SET', 'technical');
    setLiveValue('NN_HIDDEN_LAYERS', '32');
  } else if (model === 'mlp') {
    setLiveValue('TRAINER_ENV', 'env/trainers/mlp_torch.env');
    setLiveValue('LIVE_TRAIN_BACKEND', 'torch');
    setLiveValue('LIVE_TRAIN_LOOKBACK', '50');
    setLiveValue('LIVE_TRAIN_SEQUENCE_FEATURE_SET', 'basic');
    setLiveValue('NN_HIDDEN_LAYERS', '64,32');
  } else if (model === 'cnn') {
    setLiveValue('TRAINER_ENV', 'env/trainers/cnn_torch.env');
    setLiveValue('LIVE_TRAIN_BACKEND', 'torch');
    setLiveValue('LIVE_TRAIN_LOOKBACK', '50');
    setLiveValue('LIVE_TRAIN_SEQUENCE_FEATURE_SET', 'basic');
    setLiveValue('NN_HIDDEN_LAYERS', '32,16');
  }
  if ($('#liveSettingsStatus')) $('#liveSettingsStatus').textContent = 'Unsaved live model selection...';
}
function liveFormHtml() {
  return `<form id="liveConfigForm">
    ${liveSection('Asset + active model', [
      liveSelect('DATA_SOURCE', 'Data Source', ['binance'], 'binance'),
      liveField('SYMBOL', 'Symbol', 'SOLUSDT'),
      liveField('INTERVAL', 'Interval', '3m'),
      liveSelect('LIVE_MODEL_TYPE', 'Model Used By Live Sim', ['mlp','cnn','gru','lstm','transformer'], 'cnn'),
      liveSelect('TRAINER_ENV', 'Trainer Preset', ['env/trainers/mlp_torch.env','env/trainers/cnn_torch.env','env/trainers/gru_torch.env','env/trainers/lstm_torch.env','env/trainers/transformer_torch.env'], 'env/trainers/cnn_torch.env')
    ])}
    ${liveSection('Retrain schedule + window', [
      liveField('LIVE_RETRAIN_FREQUENCY', 'Update Frequency', '1d'),
      liveField('LIVE_RETRAIN_TRAIN_START', 'Training Window Start', '2023-11-18T00:00:00Z'),
      liveField('LIVE_RETRAIN_TRAIN_END', 'Training Window End', '2026-05-18T00:00:00Z'),
      liveField('LIVE_RETRAIN_LOOKBACK_DAYS', 'Fallback Lookback Days', '913', 'number', '1'),
      liveSelect('LIVE_TRAIN_USE_FULL_WINDOW', 'Train On Full Window', ['true','false'], 'true')
    ])}
    ${liveSection('Training model settings', [
      liveSelect('LIVE_TRAIN_MODEL_TYPE', 'Retrain Model Type', ['mlp','cnn','gru','lstm','transformer'], 'cnn'),
      liveSelect('LIVE_TRAIN_BACKEND', 'Backend', ['torch'], 'torch'),
      liveSelect('LIVE_TRAIN_DEVICE', 'Device', ['cuda','cuda:0','auto'], 'cuda'),
      liveField('LIVE_TRAIN_LOOKBACK', 'Lookback Candles', '50', 'number', '1'),
      liveSelect('LIVE_TRAIN_SEQUENCE_FEATURE_SET', 'Feature Set', ['basic','technical'], 'basic'),
      liveField('LIVE_TRAIN_EDGE', 'Target Edge', '0.0003', 'number', '0.0001'),
      liveField('NN_EPOCHS', 'Epochs', '180', 'number', '1'),
      liveField('NN_BATCH_SIZE', 'Batch Size', '2048', 'number', '1'),
      liveField('NN_LR', 'Learning Rate', '0.0008', 'number', '0.0001'),
      liveField('NN_L2', 'L2', '0.0003', 'number', '0.0001')
    ])}
    ${liveSection('Sequence model shape', [
      liveField('NN_CNN_FILTERS', 'CNN Filters', '16,32'),
      liveField('NN_CNN_KERNEL_SIZES', 'CNN Kernel Sizes', '5,3'),
      liveField('NN_HIDDEN_LAYERS', 'Dense Hidden Layers', '32,16'),
      liveField('NN_LSTM_HIDDEN_SIZE', 'LSTM Hidden Size', '64', 'number', '1'),
      liveField('NN_LSTM_LAYERS', 'LSTM Layers', '1', 'number', '1'),
      liveField('NN_LSTM_DROPOUT', 'LSTM Dropout', '0.0', 'number', '0.01'),
      liveField('NN_GRU_HIDDEN_SIZE', 'GRU Hidden Size', '64', 'number', '1'),
      liveField('NN_GRU_LAYERS', 'GRU Layers', '1', 'number', '1'),
      liveField('NN_GRU_DROPOUT', 'GRU Dropout', '0.0', 'number', '0.01'),
      liveField('NN_TRANSFORMER_D_MODEL', 'Transformer Width', '64', 'number', '1'),
      liveField('NN_TRANSFORMER_HEADS', 'Attention Heads', '4', 'number', '1'),
      liveField('NN_TRANSFORMER_LAYERS', 'Encoder Layers', '2', 'number', '1'),
      liveField('NN_TRANSFORMER_FF_DIM', 'Feed-forward Width', '128', 'number', '1'),
      liveField('NN_TRANSFORMER_DROPOUT', 'Transformer Dropout', '0.1', 'number', '0.01')
    ])}
    ${liveSection('Paper-trading settings', [
      liveSelect('TRADE_MODE', 'Trade Mode', ['long_only','short_only','long_short'], 'long_only'),
      liveField('LIVE_STARTING_CASH', 'Starting Cash', '100', 'number', '1'),
      liveField('LIVE_MIN_INVEST', 'Min Invest', '1', 'number', '0.01'),
      liveField('LIVE_MAX_INVEST', 'Max Long Size', 'm'),
      liveField('LIVE_MAX_SHORT_INVEST', 'Max Short Size', 'm'),
      liveField('LIVE_CONFIDENCE_MULTIPLIER', 'Confidence Multiplier', '1.0', 'number', '0.1'),
      liveField('THRESHOLD', 'Long Entry Threshold', '0.55', 'number', '0.01'),
      liveField('EXIT_THRESHOLD', 'Long Exit Threshold', '0.48', 'number', '0.01'),
      liveField('SHORT_ENTRY_THRESHOLD', 'Short Entry Threshold', '0.55', 'number', '0.01'),
      liveField('SHORT_EXIT_THRESHOLD', 'Short Exit / Cover Threshold', '0.48', 'number', '0.01'),
      liveField('FEE', 'Fee Per Side', '0.0001', 'number', '0.00001'),
    ])}
    <p class="muted"><strong>Shorting Mode:</strong> Paper Simulation Only. No margin, futures, or real orders are used.</p>
    <details class="settings-details"><summary>Advanced Trading Settings</summary><div class="formgrid">${[
      liveField('LIVE_SLIPPAGE', 'Slippage Per Side', '0', 'number', '0.00001'),
      liveField('BORROW_FEE', 'Borrow / Funding Fee Per Bar', '0', 'number', '0.000001'),
      liveSelect('ALLOW_FLIP_POSITION', 'Allow Flip Long / Short', ['0','1'], '0'),
      liveField('STOP_LOSS', 'Stop Loss', '0.002', 'number', '0.0001'),
      liveField('TAKE_PROFIT', 'Take Profit', '0.004', 'number', '0.0001'),
      liveField('MAX_HOLD_BARS', 'Max Hold Bars', '60', 'number', '1'),
      liveField('LEVERAGE', 'Leverage (1x Only)', '1', 'number', '1'),
      liveSelect('LIQUIDATION_SIMULATION', 'Liquidation Simulation', ['off','basic'], 'off')
    ].join('')}</div></details>
  </form>`;
}
function collectLiveParams() {
  const params = {};
  const form = $('#liveConfigForm');
  if (!form) return params;
  $$('input,select', form).forEach(el => params[el.name] = el.value);
  if (params.LIVE_MODEL_TYPE) params.NN_MODEL_TYPE = params.LIVE_MODEL_TYPE;
  if (params.LIVE_TRAIN_LOOKBACK) params.NN_LOOKBACK = params.LIVE_TRAIN_LOOKBACK;
  if (params.LIVE_TRAIN_SEQUENCE_FEATURE_SET) params.NN_SEQUENCE_FEATURE_SET = params.LIVE_TRAIN_SEQUENCE_FEATURE_SET;
  if (params.LIVE_TRAIN_BACKEND) params.NN_BACKEND = params.LIVE_TRAIN_BACKEND;
  if (params.LIVE_TRAIN_DEVICE) params.NN_DEVICE = params.LIVE_TRAIN_DEVICE;
  return params;
}
async function saveLiveSettings(updateStatus=true) {
  const params = collectLiveParams();
  state.settings.live = { ...(state.settings.live || {}), ...params };
  try {
    state.settings = await api('/api/live/settings', { method:'POST', body:JSON.stringify({params}) });
    if (updateStatus && $('#liveSettingsStatus')) $('#liveSettingsStatus').textContent = `Saved live settings to ${state.settings.path}.`;
  } catch (e) {
    if (updateStatus && $('#liveSettingsStatus')) $('#liveSettingsStatus').textContent = `Live settings save failed: ${e.message}`;
  }
}
async function startLiveAction(action) {
  const params = collectLiveParams();
  await saveLiveSettings(false);
  const job = await api('/api/live/action', {method:'POST', body:JSON.stringify({action, params})});
  state.currentJob = job.id;
  pollJobs();
}
async function renderLive() {
  $('#app').innerHTML = `<div class="split"><section class="panel"><h2>Live Simulation</h2><p class="muted">Docker live sim stays on port 8080. This page writes the selected model config to live_sim/env/active.env and snapshots it under models/live_env_snapshots.</p><div id="liveStatus" class="grid"></div><div class="actions"><button class="primary" data-live="start">Start Live Sim</button><button data-live="sync">Switch Model Env</button><button class="danger" data-live="stop">Stop Live Sim</button><button data-live="update_model">Update Model</button><button id="saveLiveSettingsBtn">Save Page Settings</button><button id="liveLogsBtn">Logs</button><a class="button" href="${esc(state.config?.live_public_url || '')}" target="_blank">Open Full Live Dashboard</a></div><p class="muted" id="liveSettingsStatus">Live settings save to ${esc(state.settings?.path || 'data/reports/dashboard_settings.json')}.</p>${liveFormHtml()}<div class="progress"><span id="jobBar"></span></div><p class="muted" id="jobStatus">No live action running.</p><details id="jobLogDetails" open><summary id="jobLogSummary">Terminal logs</summary><pre id="jobLog"></pre></details><pre id="liveLogs"></pre></section><section class="panel"><h2>Embedded Live Dashboard</h2><iframe id="liveFrame" src="${esc(state.config?.live_public_url || '')}"></iframe></section></div>`;
  $$('[data-live]').forEach(btn => btn.onclick = async () => startLiveAction(btn.dataset.live));
  $('#saveLiveSettingsBtn').onclick = async () => saveLiveSettings(true);
  $$('input,select', $('#liveConfigForm')).forEach(el => {
    el.addEventListener('input', () => { if ($('#liveSettingsStatus')) $('#liveSettingsStatus').textContent = 'Unsaved live settings...'; });
    el.addEventListener('change', () => { if ($('#liveSettingsStatus')) $('#liveSettingsStatus').textContent = 'Unsaved live settings...'; });
  });
  const liveModelSelect = $('#liveConfigForm [name="LIVE_MODEL_TYPE"]');
  if (liveModelSelect) liveModelSelect.addEventListener('change', () => applyLiveModelPreset(liveModelSelect.value));
  $('#liveLogsBtn').onclick = async () => { const data = await api('/api/live/logs'); $('#liveLogs').textContent = data.logs || data.error || ''; };
  refreshLive();
}
async function refreshLive() {
  if (!$('#liveStatus')) return;
  try {
    const s = await api('/api/live/status');
    const account = s.account || {}; const cfg = s.config || {}; const dec = s.latest_decision || {}; const ticker = s.latest_ticker || {};
    $('#liveStatus').innerHTML = [
      ['Status', s.running === false ? 'not running' : (s.status || 'unknown')], ['Symbol', cfg.symbol || ''], ['Interval', cfg.interval || ''], ['Train Model', cfg.train_model_type || ''], ['Retrain Every', cfg.retrain_frequency || ''], ['Equity', account.equity ?? account.account_value ?? ''], ['Cash', account.cash ?? ''], ['Position', s.position ? 'open' : 'none'], ['Prob Up', dec.prob_up ?? ''], ['Bid / Ask', `${ticker.bid_price ?? ''} / ${ticker.ask_price ?? ''}`]
    ].map(([k,v]) => `<div class="card"><h3>${esc(k)}</h3><p class="muted ${String(v).includes('not running')?'status-bad':'status-ok'}">${esc(v)}</p></div>`).join('');
  } catch (e) { $('#liveStatus').innerHTML = `<div class="card"><h3>Status</h3><p class="muted status-bad">${esc(e.message)}</p></div>`; }
}
async function renderCompare() {
  $('#app').innerHTML = `<div class="compare-stack">
    <section class="panel compare-card compare-primary">
      <h2>Simulation Equity</h2>
      <p class="muted">Filter by asset first, then candle interval, then model types. Date range limits how much of the simulation graph is shown.</p>
      <div class="formgrid compare-filters">
        <label>Stock / Crypto<select id="compareAsset"></select></label>
        <label>Time Period<select id="compareInterval"></select></label>
        <label>Graph Start<input id="compareStart" type="datetime-local"></label>
        <label>Graph End<input id="compareEnd" type="datetime-local"></label>
      </div>
      <div class="formgrid compare-filters">
        <label>Starting Cash<input id="compareStartingCash" type="number" step="1"></label>
        <label>Fee Per Side<input id="compareFee" type="number" step="0.00001"></label>
        <label>Trade Mode<select id="compareTradeMode"><option value="long_only">Long Only</option><option value="short_only">Short Only</option><option value="long_short">Long + Short</option></select></label>
        <label>Long Entry Threshold<input id="compareThreshold" type="number" step="0.01"></label>
        <label>Long Exit Threshold<input id="compareExitThreshold" type="number" step="0.01"></label>
        <label>Short Entry Threshold<input id="compareShortEntryThreshold" type="number" step="0.01"></label>
        <label>Short Exit Threshold<input id="compareShortExitThreshold" type="number" step="0.01"></label>
        <label>Min Invest<input id="compareMinInvest" type="number" step="1"></label>
        <label>Max Long Size<input id="compareMaxInvest" type="text"></label>
        <label>Max Short Size<input id="compareMaxShortInvest" type="text"></label>
        <label>Size Multiplier<input id="compareConfidenceMultiplier" type="number" step="0.1"></label>
        <label>Borrow Fee Per Bar<input id="compareBorrowFee" type="number" step="0.000001"></label>
        <label>Allow Flip Position<select id="compareAllowFlip"><option value="0">No</option><option value="1">Yes</option></select></label>
        <label>Max Hold Bars<input id="compareMaxHoldBars" type="number" step="1"></label>
        <label>Stop Loss<input id="compareStopLoss" type="number" step="0.0001"></label>
        <label>Take Profit<input id="compareTakeProfit" type="number" step="0.0001"></label>
      </div>
      <p class="muted">Compare curves are rebuilt from prediction files with these shared settings, so each selected model starts from the same cash baseline. Mouse wheel zooms, drag pans, double-click resets.</p>
      <div><div class="muted">Model Types</div><div id="compareModelFilters" class="model-filter-list"></div></div>
      <div id="simLegend" class="compare-legend"></div>
      <canvas id="simCanvas" width="1500" height="520"></canvas>
      <p class="muted" id="simCount"></p>
    </section>
    <section class="panel compare-card">
      <h2>Training Loss</h2>
      <p class="muted">Uses the same asset, interval, and model-type filters. Training loss is plotted by epoch, so the graph date range does not apply here.</p>
      <div id="trainLegend" class="compare-legend"></div>
      <canvas id="trainCanvas" width="1500" height="390"></canvas>
      <p class="muted" id="trainCount"></p>
    </section>
    <section class="panel"><h2>Saved Runs</h2><div id="runs"></div></section>
  </div>`;
  const [sims, trains, runs] = await Promise.all([api('/api/sim-series/list'), api('/api/training-series/list'), api('/api/runs')]);
  const previousCompare = state.compare || {};
  state.compare = {
    sims,
    trains,
    runs,
    asset: previousCompare.asset,
    interval: previousCompare.interval,
    start: previousCompare.start,
    end: previousCompare.end,
    selectedModels: new Set(previousCompare.selectedModels || []),
    simSettings: previousCompare.simSettings || defaultCompareSimSettings(),
    simRange: previousCompare.simRange || null,
    simFullRange: previousCompare.simFullRange || null,
    latestSimSeries: previousCompare.latestSimSeries || [],
    simSignature: previousCompare.simSignature || '',
  };
  initCompareFilters();
  renderRunsTable(runs);
  await updateCompareGraphs();
}
function renderRunsTable(runs) {
  $('#runs').innerHTML = `<table><thead><tr><th>Run</th><th>Model</th><th>Asset</th><th>Created</th></tr></thead><tbody>${[...(runs.current||[]), ...(runs.runs||[])].slice(0,80).map(r => `<tr><td>${esc(r.run_id)}</td><td>${esc([r.family,r.model_type,r.backend].filter(Boolean).join('/'))}</td><td>${esc([r.data_source,r.symbol,r.interval].filter(Boolean).join('/'))}</td><td>${esc(r.created_at_utc || '')}</td></tr>`).join('')}</tbody></table>`;
}
function sharedSetting(name, fallback) {
  const value = state.settings?.shared?.[name];
  return value === undefined || value === null || value === '' ? fallback : value;
}
function defaultCompareSimSettings() {
  return {
    starting_cash: sharedSetting('SIM_STARTING_CASH', '100'),
    fee: sharedSetting('FEE', '0.0001'),
    threshold: sharedSetting('THRESHOLD', sharedSetting('DECISION_THRESHOLD', '0.52')),
    exit_threshold: sharedSetting('EXIT_THRESHOLD', '0.50'),
    trade_mode: sharedSetting('TRADE_MODE', 'long_only'),
    short_entry_threshold: sharedSetting('SHORT_ENTRY_THRESHOLD', '0.55'),
    short_exit_threshold: sharedSetting('SHORT_EXIT_THRESHOLD', '0.48'),
    min_invest: sharedSetting('SIM_MIN_INVEST', '1'),
    max_invest: sharedSetting('SIM_MAX_INVEST', 'm'),
    max_short_invest: sharedSetting('SIM_MAX_SHORT_INVEST', sharedSetting('SIM_MAX_INVEST', 'm')),
    confidence_multiplier: sharedSetting('SIM_CONFIDENCE_MULTIPLIER', '1'),
    borrow_fee: sharedSetting('BORROW_FEE', '0'),
    allow_flip_position: sharedSetting('ALLOW_FLIP_POSITION', '0'),
    max_hold_bars: sharedSetting('MAX_HOLD_BARS', '60'),
    stop_loss: sharedSetting('STOP_LOSS', '0.002'),
    take_profit: sharedSetting('TAKE_PROFIT', '0.004'),
  };
}
function initCompareSimControls() {
  const cmp = state.compare;
  cmp.simSettings = { ...defaultCompareSimSettings(), ...(cmp.simSettings || {}) };
  const bindings = [
    ['compareStartingCash', 'starting_cash'],
    ['compareFee', 'fee'],
    ['compareTradeMode', 'trade_mode'],
    ['compareThreshold', 'threshold'],
    ['compareExitThreshold', 'exit_threshold'],
    ['compareShortEntryThreshold', 'short_entry_threshold'],
    ['compareShortExitThreshold', 'short_exit_threshold'],
    ['compareMinInvest', 'min_invest'],
    ['compareMaxInvest', 'max_invest'],
    ['compareMaxShortInvest', 'max_short_invest'],
    ['compareConfidenceMultiplier', 'confidence_multiplier'],
    ['compareBorrowFee', 'borrow_fee'],
    ['compareAllowFlip', 'allow_flip_position'],
    ['compareMaxHoldBars', 'max_hold_bars'],
    ['compareStopLoss', 'stop_loss'],
    ['compareTakeProfit', 'take_profit'],
  ];
  bindings.forEach(([id, key]) => {
    const el = $('#' + id);
    if (!el) return;
    el.value = cmp.simSettings[key] ?? '';
    el.onchange = () => {
      cmp.simSettings[key] = el.value;
      cmp.simRange = null;
      scheduleCompareGraphsUpdate();
    };
  });
}
function compareSimQueryParams(items) {
  const cmp = state.compare;
  const settings = { ...defaultCompareSimSettings(), ...(cmp.simSettings || {}) };
  const params = new URLSearchParams();
  items.forEach(i => params.append('path', i.path));
  params.set('normalized', '1');
  Object.entries(settings).forEach(([key, value]) => params.set(key, value));
  if (cmp.start) params.set('start', cmp.start);
  if (cmp.end) params.set('end', cmp.end);
  return params;
}
function scheduleCompareGraphsUpdate(delay=150) {
  const cmp = state.compare;
  if (!cmp) return;
  clearTimeout(cmp.updateTimer);
  cmp.updateTimer = setTimeout(() => updateCompareGraphs(), delay);
}
function initCompareFilters() {
  const cmp = state.compare;
  const all = [...(cmp.sims || []), ...(cmp.trains || [])].filter(s => s.asset_key && s.interval);
  const assets = uniqueSorted(all.map(s => s.asset_key));
  const preferredAsset = [state.settings?.shared?.DATA_SOURCE, state.settings?.shared?.SYMBOL].filter(Boolean).join('/');
  cmp.asset = assets.includes(cmp.asset) ? cmp.asset : (assets.includes(preferredAsset) ? preferredAsset : assets[0] || '');
  $('#compareAsset').innerHTML = assets.map(a => `<option value="${esc(a)}" ${a===cmp.asset?'selected':''}>${esc(a)}</option>`).join('');
  $('#compareAsset').onchange = () => {
    cmp.asset = $('#compareAsset').value;
    cmp.selectedModels = new Set();
    cmp.simRange = null;
    refreshCompareDependentFilters();
    scheduleCompareGraphsUpdate();
  };
  $('#compareInterval').onchange = () => {
    cmp.interval = $('#compareInterval').value;
    cmp.selectedModels = new Set();
    cmp.simRange = null;
    refreshCompareDependentFilters();
    scheduleCompareGraphsUpdate();
  };
  $('#compareStart').value = cmp.start || '';
  $('#compareEnd').value = cmp.end || '';
  $('#compareStart').onchange = () => { cmp.start = $('#compareStart').value; cmp.simRange = null; scheduleCompareGraphsUpdate(); };
  $('#compareEnd').onchange = () => { cmp.end = $('#compareEnd').value; cmp.simRange = null; scheduleCompareGraphsUpdate(); };
  initCompareSimControls();
  refreshCompareDependentFilters();
}
function refreshCompareDependentFilters() {
  const cmp = state.compare;
  const all = [...(cmp.sims || []), ...(cmp.trains || [])].filter(s => s.asset_key === cmp.asset);
  const intervals = uniqueSorted(all.map(s => s.interval));
  const preferredInterval = state.settings?.shared?.INTERVAL || '';
  cmp.interval = intervals.includes(cmp.interval) ? cmp.interval : (intervals.includes(preferredInterval) ? preferredInterval : intervals[0] || '');
  $('#compareInterval').innerHTML = intervals.map(i => `<option value="${esc(i)}" ${i===cmp.interval?'selected':''}>${esc(i)}</option>`).join('');
  const models = uniqueSeriesModels([...cmp.sims, ...cmp.trains].filter(s => s.asset_key === cmp.asset && s.interval === cmp.interval));
  if (!models.some(m => cmp.selectedModels.has(m.key))) {
    cmp.selectedModels = new Set(models.slice(0, 6).map(m => m.key));
  }
  $('#compareModelFilters').innerHTML = models.map(m => `<label title="${esc(m.group)}"><input type="checkbox" value="${esc(m.key)}" ${cmp.selectedModels.has(m.key)?'checked':''}> ${esc(m.group)} · ${esc(m.label)}</label>`).join('') || '<p class="muted">No model outputs found for this asset and interval.</p>';
  $$('input[type="checkbox"]', $('#compareModelFilters')).forEach(input => {
    input.onchange = () => {
      cmp.selectedModels = new Set($$('input:checked', $('#compareModelFilters')).map(el => el.value));
      cmp.simRange = null;
      scheduleCompareGraphsUpdate();
    };
  });
}
function uniqueSorted(values) { return Array.from(new Set(values.filter(Boolean))).sort((a,b) => a.localeCompare(b, undefined, {numeric:true})); }
function uniqueSeriesModels(items) {
  const map = new Map();
  items.forEach(item => {
    const key = item.model_key || item.label;
    if (!key) return;
    if (!map.has(key)) map.set(key, { key, label: item.model_label || item.label || key, rank: Number(item.complexity_rank ?? 999), group: item.complexity_group || 'Other' });
  });
  return Array.from(map.values()).sort((a,b) => (a.rank - b.rank) || a.label.localeCompare(b.label));
}
function selectedCompareItems(items) {
  const cmp = state.compare;
  return (items || []).filter(item => item.asset_key === cmp.asset && item.interval === cmp.interval && cmp.selectedModels.has(item.model_key || item.label));
}
async function updateCompareGraphs() {
  if (!state.compare || !$('#simCanvas')) return;
  const cmp = state.compare;
  cmp.updateToken = (cmp.updateToken || 0) + 1;
  const token = cmp.updateToken;
  const simItems = selectedCompareItems(cmp.sims);
  const trainItems = selectedCompareItems(cmp.trains);
  const simParams = compareSimQueryParams(simItems);
  const signature = simParams.toString();
  if (cmp.simSignature !== signature) {
    cmp.simSignature = signature;
    cmp.simRange = null;
  }
  if ($('#simCount')) $('#simCount').textContent = 'Loading normalized simulation curves...';
  const simSeries = simItems.length ? await api('/api/sim-series?' + signature) : [];
  const trainSeries = trainItems.length ? await api('/api/training-series?' + trainItems.map(i => 'path=' + encodeURIComponent(i.path)).join('&')) : [];
  if (token !== cmp.updateToken) return;
  const rangedSim = simSeries;
  cmp.latestSimSeries = rangedSim;
  cmp.simFullRange = fullTimeRange(rangedSim);
  if (!rangeIsValid(cmp.simRange, cmp.simFullRange)) {
    cmp.simRange = cmp.simFullRange ? [...cmp.simFullRange] : null;
  }
  drawCompareChart($('#simCanvas'), rangedSim, { xType:'time', yLabel:'Account equity ($)', xLabel:'Time', legendEl:$('#simLegend'), xRange:cmp.simRange, fullXRange:cmp.simFullRange });
  drawCompareChart($('#trainCanvas'), trainSeries, { xType:'index', yLabel:'Training loss', xLabel:'Epoch', legendEl:$('#trainLegend') });
  setupCompareZoom($('#simCanvas'));
  const settings = { ...defaultCompareSimSettings(), ...(cmp.simSettings || {}) };
  $('#simCount').textContent = `${rangedSim.filter(s => (s.points||[]).length).length} normalized simulation curve(s), ${rangedSim.reduce((n,s)=>n+(s.points||[]).length,0)} plotted points. Start cash=$${Number(settings.starting_cash || 0).toFixed(2)}, fee=${settings.fee}, buy=${settings.threshold}, exit=${settings.exit_threshold}.`;
  $('#trainCount').textContent = `${trainSeries.filter(s => (s.points||[]).length).length} training loss curve(s).`;
}
function fullTimeRange(series) {
  const xs = [];
  (series || []).forEach(s => (s.points || []).forEach(p => {
    const t = Date.parse(p.t);
    if (Number.isFinite(t)) xs.push(t);
  }));
  return xs.length ? [Math.min(...xs), Math.max(...xs)] : null;
}
function rangeIsValid(range, full) {
  return Array.isArray(range) && Array.isArray(full) && Number.isFinite(range[0]) && Number.isFinite(range[1]) && range[1] > range[0] && range[0] >= full[0] && range[1] <= full[1];
}
function clampRange(range, full) {
  if (!full) return range;
  const fullSpan = Math.max(1, full[1] - full[0]);
  let span = Math.min(Math.max(1, range[1] - range[0]), fullSpan);
  let start = range[0];
  let end = start + span;
  if (start < full[0]) { start = full[0]; end = start + span; }
  if (end > full[1]) { end = full[1]; start = end - span; }
  return [start, end];
}
function redrawCompareSimOnly() {
  const cmp = state.compare;
  if (!cmp || !$('#simCanvas')) return;
  drawCompareChart($('#simCanvas'), cmp.latestSimSeries || [], { xType:'time', yLabel:'Account equity ($)', xLabel:'Time', legendEl:$('#simLegend'), xRange:cmp.simRange, fullXRange:cmp.simFullRange });
}
function scheduleCompareSimRedraw() {
  const cmp = state.compare;
  if (!cmp || cmp.redrawPending) return;
  cmp.redrawPending = true;
  requestAnimationFrame(() => {
    cmp.redrawPending = false;
    redrawCompareSimOnly();
  });
}
function setupCompareZoom(canvas) {
  if (!canvas || canvas.dataset.zoomBound) return;
  canvas.dataset.zoomBound = '1';
  let drag = null;
  const plotLeft = 76, plotRightPad = 24;
  const pointerFrac = ev => {
    const rect = canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const plotWidth = Math.max(1, rect.width - plotLeft - plotRightPad);
    return Math.min(1, Math.max(0, (x - plotLeft) / plotWidth));
  };
  canvas.addEventListener('wheel', ev => {
    const cmp = state.compare;
    if (!cmp?.simFullRange || !cmp?.simRange) return;
    ev.preventDefault();
    const full = cmp.simFullRange;
    const current = cmp.simRange;
    const fullSpan = Math.max(1, full[1] - full[0]);
    const span = Math.max(1, current[1] - current[0]);
    const factor = ev.deltaY > 0 ? 1.25 : 0.8;
    const minSpan = Math.min(fullSpan, 1000 * 60 * 15);
    const newSpan = Math.min(fullSpan, Math.max(minSpan, span * factor));
    const frac = pointerFrac(ev);
    const anchor = current[0] + span * frac;
    cmp.simRange = clampRange([anchor - newSpan * frac, anchor + newSpan * (1 - frac)], full);
    scheduleCompareSimRedraw();
  }, { passive:false });
  canvas.addEventListener('dblclick', () => {
    const cmp = state.compare;
    if (!cmp?.simFullRange) return;
    cmp.simRange = [...cmp.simFullRange];
    scheduleCompareSimRedraw();
  });
  canvas.addEventListener('mousedown', ev => {
    const cmp = state.compare;
    if (!cmp?.simFullRange || !cmp?.simRange) return;
    drag = { x: ev.clientX, range: [...cmp.simRange] };
    canvas.classList.add('dragging');
  });
  window.addEventListener('mousemove', ev => {
    const cmp = state.compare;
    if (!drag || !cmp?.simFullRange) return;
    const rect = canvas.getBoundingClientRect();
    const plotWidth = Math.max(1, rect.width - plotLeft - plotRightPad);
    const span = drag.range[1] - drag.range[0];
    const dx = ev.clientX - drag.x;
    const shift = -dx / plotWidth * span;
    cmp.simRange = clampRange([drag.range[0] + shift, drag.range[1] + shift], cmp.simFullRange);
    scheduleCompareSimRedraw();
  });
  window.addEventListener('mouseup', () => {
    drag = null;
    canvas.classList.remove('dragging');
  });
}
function parseDateFilter(value, endOfDay) {
  if (!value) return null;
  const text = String(value).trim();
  const parsed = Date.parse(text.length === 10 ? `${text}T${endOfDay ? '23:59:59' : '00:00:00'}` : text);
  return Number.isFinite(parsed) ? parsed : null;
}
function drawCompareChart(canvas, series, opts) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const cssWidth = Math.max(700, Math.floor(rect.width || canvas.width));
  const cssHeight = Math.max(300, Math.floor(rect.height || canvas.height));
  if (canvas.width !== Math.floor(cssWidth * dpr) || canvas.height !== Math.floor(cssHeight * dpr)) {
    canvas.width = Math.floor(cssWidth * dpr);
    canvas.height = Math.floor(cssHeight * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.fillStyle = '#fffdf7';
  ctx.fillRect(0, 0, cssWidth, cssHeight);
  const colors = ['#164f45','#c77519','#245ea8','#b23834','#16864e','#7b4ab3','#8a5b20','#2b7a78','#9d4edd','#5f0f40'];
  const normalized = (series || []).map((s, idx) => ({
    ...s,
    color: colors[idx % colors.length],
    displayLabel: compactSeriesLabel(s),
    points: (s.points || []).map(p => ({ x: opts.xType === 'time' ? Date.parse(p.t) : Number(p.t), y: Number(p.v), raw: p.t })).filter(p => Number.isFinite(p.x) && Number.isFinite(p.y)),
  })).filter(s => s.points.length);
  renderCompareLegend(opts.legendEl, normalized);
  const xRange = Array.isArray(opts.xRange) ? opts.xRange : null;
  const visibleSeries = normalized.map(s => ({
    ...s,
    points: xRange ? s.points.filter(p => p.x >= xRange[0] && p.x <= xRange[1]) : s.points,
  })).filter(s => s.points.length);
  const rawVisible = visibleSeries.flatMap(s => s.points);
  if (!rawVisible.length) {
    ctx.fillStyle = '#6b6256';
    ctx.font = '14px Courier New';
    ctx.fillText('No selected series data for these filters.', 28, 44);
    return;
  }
  const left = 76, right = 24, top = 26, bottom = 62;
  const plot = { left, right: cssWidth - right, top, bottom: cssHeight - bottom, width: cssWidth - left - right, height: cssHeight - top - bottom };
  const minX = xRange ? xRange[0] : Math.min(...rawVisible.map(p => p.x));
  const maxX = xRange ? xRange[1] : Math.max(...rawVisible.map(p => p.x));
  const targetPoints = compareDetailTarget(plot.width, minX, maxX, opts.fullXRange);
  const drawableSeries = visibleSeries.map(s => ({ ...s, points: decimateComparePoints(s.points, targetPoints) })).filter(s => s.points.length);
  const all = drawableSeries.flatMap(s => s.points);
  let minY = Math.min(...all.map(p => p.y));
  let maxY = Math.max(...all.map(p => p.y));
  if (Math.abs(maxY - minY) < 1e-12) { minY -= 1; maxY += 1; }
  const yPad = (maxY - minY) * 0.08;
  minY -= yPad; maxY += yPad;
  const sx = x => plot.left + (x - minX) / Math.max(1, maxX - minX) * plot.width;
  const sy = y => plot.bottom - (y - minY) / Math.max(1e-12, maxY - minY) * plot.height;
  drawCompareAxes(ctx, plot, { minX, maxX, minY, maxY, ...opts });
  drawableSeries.forEach(s => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    s.points.forEach((p, i) => {
      const x = sx(p.x), y = sy(p.y);
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.stroke();
  });
}
function compareDetailTarget(plotWidth, minX, maxX, fullRange) {
  const visibleSpan = Math.max(1, maxX - minX);
  const fullSpan = Array.isArray(fullRange) ? Math.max(visibleSpan, fullRange[1] - fullRange[0]) : visibleSpan;
  const zoom = Math.max(1, fullSpan / visibleSpan);
  const zoomBoost = Math.sqrt(Math.min(64, zoom));
  const base = Math.max(140, Math.round(plotWidth * 0.35));
  return Math.max(80, Math.min(5000, Math.round(base * zoomBoost)));
}
function decimateComparePoints(points, target) {
  if (!points || points.length <= target) return points || [];
  if (target <= 3) return [points[0], points[points.length - 1]];
  const bucketCount = Math.max(1, Math.floor((target - 2) / 2));
  const bucketSize = (points.length - 2) / bucketCount;
  const out = [points[0]];
  for (let b = 0; b < bucketCount; b++) {
    const start = 1 + Math.floor(b * bucketSize);
    const end = Math.min(points.length - 1, 1 + Math.floor((b + 1) * bucketSize));
    if (start >= end) continue;
    let min = points[start], max = points[start];
    for (let i = start + 1; i < end; i++) {
      const p = points[i];
      if (p.y < min.y) min = p;
      if (p.y > max.y) max = p;
    }
    if (min === max) out.push(min);
    else if (min.x < max.x) out.push(min, max);
    else out.push(max, min);
  }
  out.push(points[points.length - 1]);
  return out;
}
function compactSeriesLabel(s) {
  const asset = [s.source, s.symbol, s.interval].filter(Boolean).join('/');
  const model = s.model_label || s.label || s.path;
  return asset ? `${model} | ${asset}` : model;
}
function renderCompareLegend(el, series) {
  if (!el) return;
  el.innerHTML = series.length ? series.map(s => `<span><i class="legend-swatch" style="background:${esc(s.color)}"></i>${esc(s.displayLabel)}</span>`).join('') : '<span>No curve selected</span>';
}
function drawCompareAxes(ctx, plot, opts) {
  ctx.strokeStyle = '#d8c8ad';
  ctx.lineWidth = 1;
  ctx.font = '12px Courier New';
  ctx.fillStyle = '#6b6256';
  ctx.beginPath();
  ctx.moveTo(plot.left, plot.top);
  ctx.lineTo(plot.left, plot.bottom);
  ctx.lineTo(plot.right, plot.bottom);
  ctx.stroke();
  for (let i = 0; i <= 4; i++) {
    const y = plot.top + plot.height * i / 4;
    const value = opts.maxY - (opts.maxY - opts.minY) * i / 4;
    ctx.strokeStyle = i === 4 ? '#d8c8ad' : 'rgba(216,200,173,.55)';
    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(plot.right, y);
    ctx.stroke();
    ctx.fillStyle = '#6b6256';
    ctx.textAlign = 'right';
    ctx.fillText(formatY(value, opts.yLabel), plot.left - 8, y + 4);
  }
  for (let i = 0; i <= 5; i++) {
    const x = plot.left + plot.width * i / 5;
    const value = opts.minX + (opts.maxX - opts.minX) * i / 5;
    ctx.strokeStyle = 'rgba(216,200,173,.45)';
    ctx.beginPath();
    ctx.moveTo(x, plot.top);
    ctx.lineTo(x, plot.bottom);
    ctx.stroke();
    ctx.fillStyle = '#6b6256';
    ctx.textAlign = i === 0 ? 'left' : i === 5 ? 'right' : 'center';
    ctx.fillText(opts.xType === 'time' ? formatTimeTick(value, opts.maxX - opts.minX) : String(Math.round(value)), x, plot.bottom + 22);
  }
  ctx.save();
  ctx.translate(18, plot.top + plot.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillStyle = '#15120d';
  ctx.font = '13px Courier New';
  ctx.fillText(opts.yLabel || 'Value', 0, 0);
  ctx.restore();
  ctx.textAlign = 'center';
  ctx.fillStyle = '#15120d';
  ctx.font = '13px Courier New';
  ctx.fillText(opts.xLabel || 'X', plot.left + plot.width / 2, plot.bottom + 48);
}
function formatY(value, label) {
  if ((label || '').includes('$')) return `$${Number(value).toFixed(value >= 1000 ? 0 : 2)}`;
  if (Math.abs(value) >= 10) return Number(value).toFixed(2);
  return Number(value).toPrecision(4);
}
function formatTimeTick(ms, span) {
  const d = new Date(ms);
  if (!Number.isFinite(d.getTime())) return '';
  if (span > 1000 * 60 * 60 * 24 * 365) return d.toLocaleDateString(undefined, { year:'2-digit', month:'short' });
  if (span > 1000 * 60 * 60 * 24 * 45) return d.toLocaleDateString(undefined, { month:'short', day:'numeric' });
  if (span > 1000 * 60 * 60 * 24 * 2) return d.toLocaleDateString(undefined, { month:'short', day:'numeric' });
  return d.toLocaleString(undefined, { hour:'2-digit', minute:'2-digit', hour12:false });
}
init().catch(err => { document.body.innerHTML = `<pre>${esc(err.stack || err.message)}</pre>`; });
</script>
</body>
</html>'''


def main() -> None:
    args = parse_args()
    ctx = DashboardContext(args)
    ctx.reports_root.mkdir(parents=True, exist_ok=True)
    handler = make_handler(ctx)
    try:
        server = ReusableThreadingHTTPServer((args.host, args.port), handler)
    except OSError as exc:
        raise SystemExit(
            f"Could not bind dashboard to {args.host}:{args.port}: {exc}. "
            "Another server is probably already using the port. Run `make stop`, or change REPORTS_PORT."
        ) from exc
    print(f"CryptoPred dashboard: http://{args.host}:{args.port}/", flush=True)
    print(f"Live sim target: {args.live_public_url}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Stopping dashboard.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
