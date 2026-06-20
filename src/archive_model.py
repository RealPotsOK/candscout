#!/usr/bin/env python3
"""Archive trained models with reproducibility metadata and current-run pointers."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ARTIFACT_DEST_NAMES = {
    "model": "model.npz",
    "train_metrics": "train_metrics.json",
    "backtest_report": "backtest_report.json",
    "predictions": "predictions.parquet",
    "sim_report": "sim_report.json",
    "sim_trades": "sim_trades.csv",
    "visualization": "visualization.html",
    "sim_visualization": "sim_visualization.html",
}


def parse_key_value(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive a model artifact and run configuration.")
    parser.add_argument("--archive-root", default="models/archive", help="Legacy archive root directory")
    parser.add_argument("--run-store", default="", help="Canonical run-store root, for example models/runs")
    parser.add_argument("--current-root", default="models/current", help="Current pointer root")
    parser.add_argument("--write-current", action="store_true", help="Write models/current pointer for this run")
    parser.add_argument("--list-runs", action="store_true", help="List recent canonical runs and exit")
    parser.add_argument("--show-current", action="store_true", help="Show current model pointers and exit")
    parser.add_argument("--name", default="", help="Snapshot/run name; default uses generated UTC name")
    parser.add_argument("--model", default="", help="Model artifact path")
    parser.add_argument("--train-metrics", default="", help="Training metrics JSON path")
    parser.add_argument("--backtest-report", default="", help="Backtest report JSON path")
    parser.add_argument("--predictions", default="", help="Predictions parquet path")
    parser.add_argument("--sim-report", default="", help="Bank simulation report JSON path")
    parser.add_argument("--sim-trades", default="", help="Bank simulation trades CSV path")
    parser.add_argument("--visualization", default="", help="Model visualization HTML path")
    parser.add_argument("--sim-visualization", default="", help="Simulation visualization HTML path")
    parser.add_argument("--env-file", action="append", default=[], help="Env file to copy into snapshot")
    parser.add_argument("--param", action="append", default=[], help="Resolved KEY=VALUE parameter")
    parser.add_argument("--family", default="", help="Model family, for example nn or lr")
    parser.add_argument("--model-type", default="", help="Model type, for example cnn or logreg")
    parser.add_argument("--backend", default="", help="Training backend, for example torch or numpy")
    parser.add_argument("--asset-env", default="", help="Selected asset preset env path")
    parser.add_argument("--trainer-env", default="", help="Selected trainer preset env path")
    parser.add_argument("--include-diff", action="store_true", help="Include current git diff patch")
    parser.add_argument("--limit", type=int, default=20, help="Rows for --list-runs/--show-current")
    return parser.parse_args()


def run_git(args: list[str]) -> str:
    try:
        result = subprocess.run(["git", *args], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return ""
    return result.stdout.strip()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return json_safe(value.item())
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return str(value)


def load_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def model_metadata(path: Path) -> dict[str, Any]:
    if not path.exists() or path.suffix != ".npz":
        return {}

    metadata: dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as artifact:
        for key in artifact.files:
            value = artifact[key]
            if key.startswith(("conv_W_", "conv_b_", "dense_W_", "dense_b_", "state_value_", "W_", "b_")) or key in {
                "weights",
                "mean",
                "std",
            }:
                metadata[key] = {"shape": list(value.shape)}
            else:
                metadata[key] = json_safe(value)
    return metadata


def slug(value: str, fallback: str) -> str:
    raw = str(value or fallback).strip()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-_.")
    return safe or fallback


def resolve_identity(args: argparse.Namespace, params: dict[str, str], timestamp_compact: str) -> dict[str, str]:
    family = args.family or ("nn" if params.get("NN_MODEL_TYPE") else "lr")
    model_type = args.model_type or params.get("NN_MODEL_TYPE") or params.get("MODEL_TYPE") or "logreg"
    backend = args.backend or params.get("NN_BACKEND") or ("numpy" if family == "lr" else "unknown")
    source = params.get("DATA_SOURCE", "unknown")
    symbol = params.get("SYMBOL", "unknown")
    interval = params.get("INTERVAL", "unknown")
    run_id = args.name.strip() or "_".join(
        [
            timestamp_compact,
            slug(source, "source"),
            slug(symbol, "symbol"),
            slug(interval, "interval"),
            slug(family, "family"),
            slug(model_type, "model"),
            slug(backend, "backend"),
        ]
    )
    return {
        "run_id": slug(run_id, timestamp_compact),
        "data_source": source,
        "symbol": symbol,
        "interval": interval,
        "family": family,
        "model_type": model_type,
        "backend": backend,
        "asset_env": args.asset_env or params.get("ASSET_ENV", ""),
        "trainer_env": args.trainer_env or params.get("TRAINER_ENV", ""),
    }


def unique_archive_dir(root: Path, run_id: str, allow_suffix: bool) -> tuple[Path, str]:
    candidate = root / run_id
    if not candidate.exists():
        return candidate, run_id
    if not allow_suffix:
        raise FileExistsError(f"Archive already exists: {candidate}")
    for idx in range(2, 1000):
        suffixed_id = f"{run_id}-{idx:02d}"
        candidate = root / suffixed_id
        if not candidate.exists():
            return candidate, suffixed_id
    raise FileExistsError(f"Could not find unique run directory under {root} for {run_id}")


def copy_artifact(src: str, dest_dir: Path, artifacts: dict[str, dict[str, str]], logical_name: str, canonical: bool) -> None:
    if not src:
        artifacts[logical_name] = {"status": "not_provided", "source": "", "path": ""}
        return

    src_path = Path(src)
    if not src_path.exists():
        artifacts[logical_name] = {"status": "missing", "source": src, "path": ""}
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    if canonical and logical_name == "model":
        dest_name = "model" + (src_path.suffix or ".artifact")
    else:
        dest_name = ARTIFACT_DEST_NAMES.get(logical_name) if canonical else None
    dest_path = dest_dir / (dest_name or src_path.name)
    shutil.copy2(src_path, dest_path)
    artifacts[logical_name] = {"status": "copied", "source": src, "path": str(dest_path)}


def copy_env_files(env_files: list[str], dest_dir: Path) -> dict[str, dict[str, str]]:
    copied: dict[str, dict[str, str]] = {}
    for env_file in env_files:
        if not env_file:
            continue
        src_path = Path(env_file)
        key = str(src_path)
        if not src_path.exists():
            copied[key] = {"status": "missing", "source": key, "path": ""}
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name
        shutil.copy2(src_path, dest_path)
        copied[key] = {"status": "copied", "source": key, "path": str(dest_path)}
    return copied


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + ("\n" if content and not content.endswith("\n") else ""), encoding="utf-8")


def pick_model_metric_key(train_metrics: dict[str, Any], backtest_report: dict[str, Any], identity: dict[str, str]) -> str:
    if train_metrics.get("model_type"):
        return str(train_metrics["model_type"])
    if backtest_report.get("assumptions", {}).get("model_type"):
        return str(backtest_report["assumptions"]["model_type"])
    if identity["family"] == "lr":
        return "logistic_regression"
    if identity["family"] == "nn" and identity["model_type"] in {"mlp", "cnn", "gru", "lstm", "transformer"}:
        return f"sequence_{identity['model_type']}"
    return identity["model_type"]


def extract_summary_metrics(train_metrics: dict[str, Any], backtest_report: dict[str, Any], identity: dict[str, str]) -> dict[str, Any]:
    model_key = pick_model_metric_key(train_metrics, backtest_report, identity)
    training = train_metrics.get("model_training", {})
    train_comparison = train_metrics.get("baseline_vs_model", {})
    backtests = backtest_report.get("backtest", {})
    return {
        "model_metric_key": model_key,
        "dataset": train_metrics.get("dataset", {}),
        "class_balance_test": train_metrics.get("class_balance", {}).get("test", {}),
        "train_metrics": train_comparison.get(model_key, {}),
        "best_threshold": train_metrics.get("threshold_sweep", {}).get("best", {}),
        "final_loss": training.get("final_loss"),
        "backend": training.get("backend", identity.get("backend")),
        "device": training.get("device"),
        "backtest": backtests.get(model_key, {}),
    }


def write_current_pointer(current_root: Path, manifest: dict[str, Any]) -> Path:
    identity = manifest["run_identity"]
    pointer_dir = current_root / slug(identity["data_source"], "source") / slug(identity["symbol"], "symbol") / slug(identity["interval"], "interval")
    pointer_name = "_".join(
        [
            slug(identity["family"], "family"),
            slug(identity["model_type"], "model"),
            slug(identity["backend"], "backend"),
        ]
    ) + ".json"
    pointer_path = pointer_dir / pointer_name
    artifacts = manifest["artifacts"]
    pointer = {
        "run_id": identity["run_id"],
        "run_dir": manifest["archive_dir"],
        "created_at_utc": manifest["created_at_utc"],
        "data_source": identity["data_source"],
        "symbol": identity["symbol"],
        "interval": identity["interval"],
        "family": identity["family"],
        "model_type": identity["model_type"],
        "backend": identity["backend"],
        "model_path": artifacts.get("model", {}).get("path", ""),
        "metrics_path": artifacts.get("train_metrics", {}).get("path", ""),
        "backtest_path": artifacts.get("backtest_report", {}).get("path", ""),
        "predictions_path": artifacts.get("predictions", {}).get("path", ""),
        "summary_metrics": manifest.get("summary_metrics", {}),
    }
    write_text(pointer_path, json.dumps(pointer, indent=2, sort_keys=True))
    return pointer_path


def write_readme(archive_dir: Path, manifest: dict[str, Any], args: argparse.Namespace) -> None:
    identity = manifest["run_identity"]
    summary = manifest.get("summary_metrics", {})
    backtest = summary.get("backtest", {}) or {}
    train_metrics = summary.get("train_metrics", {}) or {}
    lines = [
        f"# Model Run: {identity['run_id']}",
        "",
        f"- Created UTC: `{manifest['created_at_utc']}`",
        f"- Source/Symbol/Interval: `{identity['data_source']} / {identity['symbol']} / {identity['interval']}`",
        f"- Family: `{identity['family']}`",
        f"- Model type: `{identity['model_type']}`",
        f"- Backend: `{identity['backend']}`",
        f"- Asset env: `{identity.get('asset_env', '')}`",
        f"- Trainer env: `{identity.get('trainer_env', '')}`",
        f"- Model source: `{args.model}`",
        "",
        "Key metrics:",
        f"- Train accuracy: `{train_metrics.get('accuracy', '')}`",
        f"- Train precision_y1: `{train_metrics.get('precision_y1', '')}`",
        f"- Train recall_y1: `{train_metrics.get('recall_y1', '')}`",
        f"- Train f1_y1: `{train_metrics.get('f1_y1', '')}`",
        f"- Backtest total_return: `{backtest.get('total_return', '')}`",
        f"- Backtest trade_count: `{backtest.get('trade_count', '')}`",
        f"- Backtest avg_net_return_per_trade: `{backtest.get('avg_net_return_per_trade', '')}`",
        "",
        "Important files:",
        "- `manifest.json`: resolved parameters, copied paths, git state, summary metrics, and model metadata.",
        "- `env/`: copied environment defaults used by Makefile.",
        "- `artifacts/`: model, metrics, backtest report, predictions, and optional reports that existed when saved.",
    ]
    write_text(archive_dir / "README.md", "\n".join(lines))


def list_runs(run_store: Path, limit: int) -> None:
    manifests = []
    for path in run_store.glob("*/manifest.json"):
        data = load_json(str(path))
        if data:
            manifests.append(data)
    manifests.sort(key=lambda item: item.get("created_at_utc", ""), reverse=True)
    if not manifests:
        print(f"No runs found under {run_store}")
        return
    for data in manifests[:limit]:
        identity = data.get("run_identity", {})
        summary = data.get("summary_metrics", {})
        backtest = summary.get("backtest", {}) or {}
        print(
            f"{data.get('created_at_utc', '')}  "
            f"{identity.get('run_id', '')}  "
            f"{identity.get('data_source', '')}/{identity.get('symbol', '')}/{identity.get('interval', '')}  "
            f"{identity.get('family', '')}/{identity.get('model_type', '')}/{identity.get('backend', '')}  "
            f"trades={backtest.get('trade_count', '')} return={backtest.get('total_return', '')}"
        )


def show_current(current_root: Path, limit: int) -> None:
    pointers = []
    for path in current_root.glob("**/*.json"):
        data = load_json(str(path))
        if data:
            pointers.append((path, data))
    pointers.sort(key=lambda item: item[1].get("created_at_utc", ""), reverse=True)
    if not pointers:
        print(f"No current model pointers found under {current_root}")
        return
    for path, data in pointers[:limit]:
        print(
            f"{path}: {data.get('created_at_utc', '')}  "
            f"{data.get('data_source', '')}/{data.get('symbol', '')}/{data.get('interval', '')}  "
            f"{data.get('family', '')}/{data.get('model_type', '')}/{data.get('backend', '')}  "
            f"run={data.get('run_id', '')}"
        )


def main() -> None:
    args = parse_args()
    if args.list_runs:
        list_runs(Path(args.run_store or "models/runs"), args.limit)
        return
    if args.show_current:
        show_current(Path(args.current_root), args.limit)
        return
    if not args.model:
        raise ValueError("--model is required unless --list-runs or --show-current is used")
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Required model artifact not found: {args.model}")
    if args.run_store:
        required_for_run = {
            "train metrics": args.train_metrics,
            "backtest report": args.backtest_report,
            "predictions": args.predictions,
        }
        missing_required = [f"{label}: {path}" for label, path in required_for_run.items() if not path or not Path(path).exists()]
        if missing_required:
            raise FileNotFoundError(
                "Canonical run saving requires completed train/backtest artifacts:\n"
                + "\n".join(missing_required)
            )

    now = datetime.now(timezone.utc)
    created_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    timestamp_compact = now.strftime("%Y%m%dT%H%M%SZ")
    params = parse_key_value(args.param)
    identity = resolve_identity(args, params, timestamp_compact)

    if args.run_store:
        archive_root = Path(args.run_store)
        archive_dir, run_id = unique_archive_dir(archive_root, identity["run_id"], allow_suffix=True)
        identity["run_id"] = run_id
        canonical_artifacts = True
    else:
        name = args.name.strip() or timestamp_compact
        archive_dir, run_id = unique_archive_dir(Path(args.archive_root), name, allow_suffix=False)
        identity["run_id"] = run_id
        canonical_artifacts = False

    artifacts: dict[str, dict[str, str]] = {}
    copy_artifact(args.model, archive_dir / "artifacts", artifacts, "model", canonical_artifacts)
    copy_artifact(args.train_metrics, archive_dir / "artifacts", artifacts, "train_metrics", canonical_artifacts)
    copy_artifact(args.backtest_report, archive_dir / "artifacts", artifacts, "backtest_report", canonical_artifacts)
    copy_artifact(args.predictions, archive_dir / "artifacts", artifacts, "predictions", canonical_artifacts)
    copy_artifact(args.sim_report, archive_dir / "artifacts", artifacts, "sim_report", canonical_artifacts)
    copy_artifact(args.sim_trades, archive_dir / "artifacts", artifacts, "sim_trades", canonical_artifacts)
    copy_artifact(args.visualization, archive_dir / "artifacts", artifacts, "visualization", canonical_artifacts)
    copy_artifact(args.sim_visualization, archive_dir / "artifacts", artifacts, "sim_visualization", canonical_artifacts)

    copied_env = copy_env_files(args.env_file, archive_dir / "env")
    git_info = {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "status_short": run_git(["status", "--short"]),
    }

    if args.include_diff:
        write_text(archive_dir / "git_diff.patch", run_git(["diff"]))

    train_metrics = load_json(args.train_metrics)
    backtest_report = load_json(args.backtest_report)
    manifest = {
        "created_at_utc": created_at,
        "archive_name": identity["run_id"],
        "archive_dir": str(archive_dir),
        "run_identity": identity,
        "params": params,
        "artifacts": artifacts,
        "env_files": copied_env,
        "summary_metrics": extract_summary_metrics(train_metrics, backtest_report, identity),
        "model_metadata": model_metadata(Path(args.model)),
        "git": git_info,
    }

    current_pointer = ""
    if args.write_current:
        current_pointer = str(write_current_pointer(Path(args.current_root), manifest))
        manifest["current_pointer"] = current_pointer

    write_text(archive_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    write_readme(archive_dir, manifest, args)

    print(f"Saved model archive to {archive_dir}")
    if current_pointer:
        print(f"Updated current pointer: {current_pointer}")


if __name__ == "__main__":
    main()
