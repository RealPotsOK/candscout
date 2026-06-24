"""Daily rolling-window retraining scheduler for the live simulator."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sequence_nn import load_sequence_model

from .config import Config, parse_interval_seconds, parse_retrain_frequency
from .store import Store


@dataclass
class RetrainStatus:
    enabled: bool
    running: bool = False
    last_status: str | None = None
    last_started_at: str | None = None
    last_finished_at: str | None = None
    last_message: str | None = None
    last_run_dir: str | None = None
    next_run_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RetrainScheduler:
    def __init__(self, cfg: Config, store: Store) -> None:
        self.cfg = cfg
        self.store = store
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.status = RetrainStatus(enabled=cfg.retrain_enabled)
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.cfg.retrain_enabled:
            self._set_next_run(None)
            return
        self.thread = threading.Thread(target=self._loop, name="retrain-scheduler", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def status_dict(self) -> dict[str, Any]:
        with self.lock:
            return self.status.to_dict()

    def run_now_async(self, reason: str = "manual") -> None:
        if not self.cfg.retrain_enabled:
            raise RuntimeError("Retraining is disabled")
        with self.lock:
            if self.status.running:
                raise RuntimeError("Retraining is already running")
        threading.Thread(target=self._run_once_guarded, args=(reason,), name="manual-retrain", daemon=True).start()

    def run_now_sync(self, reason: str = "manual") -> None:
        if not self.cfg.retrain_enabled:
            raise RuntimeError("Retraining is disabled")
        with self.lock:
            if self.status.running:
                raise RuntimeError("Retraining is already running")
        self._run_once_guarded(reason, raise_errors=True)

    def _loop(self) -> None:
        if self.cfg.retrain_on_start:
            self._run_once_guarded("startup")

        while not self.stop_event.is_set():
            next_run = next_retrain_run_utc(self.cfg, self.store)
            self._set_next_run(iso(next_run))
            wait_seconds = max(1.0, (next_run - datetime.now(timezone.utc)).total_seconds())
            if self.stop_event.wait(wait_seconds):
                break
            self._run_once_guarded("scheduled")

    def _set_next_run(self, value: str | None) -> None:
        with self.lock:
            self.status.next_run_at = value

    def _run_once_guarded(self, reason: str, *, raise_errors: bool = False) -> None:
        with self.lock:
            if self.status.running:
                return
            self.status.running = True
            self.status.last_status = "running"
            self.status.last_started_at = now_iso()
            self.status.last_finished_at = None
            self.status.last_message = f"started: {reason}"
        started_at = now_iso()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(self.cfg.training_runs_dir) / run_id
        lock_path = retrain_lock_path(Path(self.cfg.training_runs_dir))
        self.store.insert_training_run(
            run_id=run_id,
            started_at=started_at,
            status="running",
            run_dir=str(run_dir),
            message=f"started: {reason}",
            active_model_updated=False,
        )
        self.store.insert_event(started_at, "info", f"retrain started reason={reason} run_id={run_id}")
        try:
            lock_fd = acquire_retrain_lock(lock_path)
            summary = self._run_training(run_id, run_dir)
            finished_at = now_iso()
            message = f"trained and activated model: {summary['model_path']}"
            with self.lock:
                self.status.running = False
                self.status.last_status = "success"
                self.status.last_finished_at = finished_at
                self.status.last_message = message
                self.status.last_run_dir = str(run_dir)
            self.store.finish_training_run(
                run_id=run_id,
                ended_at=finished_at,
                status="success",
                message=message,
                active_model_updated=True,
                metadata=json.dumps(summary, indent=2, sort_keys=True),
            )
            self.store.insert_event(finished_at, "info", f"retrain success run_id={run_id}")
            cleanup_old_runs(Path(self.cfg.training_runs_dir), self.cfg.retrain_keep_runs)
        except Exception as exc:  # noqa: BLE001 - scheduler must keep running after failures.
            finished_at = now_iso()
            message = str(exc)
            with self.lock:
                self.status.running = False
                self.status.last_status = "failed"
                self.status.last_finished_at = finished_at
                self.status.last_message = message
                self.status.last_run_dir = str(run_dir)
            self.store.finish_training_run(
                run_id=run_id,
                ended_at=finished_at,
                status="failed",
                message=message,
                active_model_updated=False,
                metadata=None,
            )
            self.store.insert_event(finished_at, "error", f"retrain failed run_id={run_id}: {message}")
            print(json.dumps({"event": "retrain_failed", "run_id": run_id, "error": message}), flush=True)
            if raise_errors:
                raise
        finally:
            if "lock_fd" in locals():
                release_retrain_lock(lock_fd, lock_path)

    def _run_training(self, run_id: str, run_dir: Path) -> dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        start_dt, end_dt, window_summary = resolve_training_window(self.cfg)
        raw_path = run_dir / "candles.parquet"
        cache_path = live_candle_cache_path(self.cfg)
        model_path = run_dir / "model.npz"
        metrics_path = run_dir / "train_metrics.json"
        manifest_path = run_dir / "manifest.json"
        env = os.environ.copy()
        env["PYTHONPATH"] = "/app/src" if Path("/app/src").exists() else str(Path.cwd() / "src")
        prepare_torch_runtime_env(env)
        seed_live_candle_cache(cache_path, Path(self.cfg.training_runs_dir), self.cfg.symbol, self.cfg.interval)

        download_cmd = [
            sys.executable,
            src_script("download.py"),
            "--source",
            "binance",
            "--symbol",
            self.cfg.symbol,
            "--interval",
            self.cfg.interval,
            "--start",
            iso(start_dt),
            "--end",
            iso(end_dt),
            "--out",
            str(raw_path),
            "--cache-file",
            str(cache_path),
        ]
        train_cmd = [
            sys.executable,
            src_script("train_sequence_nn.py"),
            "--raw-data",
            str(raw_path),
            "--model-out",
            str(model_path),
            "--metrics-out",
            str(metrics_path),
            "--model-type",
            self.cfg.train_model_type,
            "--backend",
            self.cfg.train_backend,
            "--device",
            self.cfg.train_device,
            "--lookback",
            str(self.cfg.train_lookback),
            "--sequence-feature-set",
            self.cfg.train_sequence_feature_set,
            "--edge",
            str(self.cfg.train_edge),
            "--split",
            str(self.cfg.train_split),
            "--cnn-filters",
            self.cfg.train_cnn_filters,
            "--cnn-kernel-sizes",
            self.cfg.train_cnn_kernel_sizes,
            "--lstm-hidden-size",
            str(self.cfg.train_lstm_hidden_size),
            "--lstm-layers",
            str(self.cfg.train_lstm_layers),
            "--lstm-dropout",
            str(self.cfg.train_lstm_dropout),
            "--gru-hidden-size",
            str(self.cfg.train_gru_hidden_size),
            "--gru-layers",
            str(self.cfg.train_gru_layers),
            "--gru-dropout",
            str(self.cfg.train_gru_dropout),
            "--transformer-d-model",
            str(self.cfg.train_transformer_d_model),
            "--transformer-heads",
            str(self.cfg.train_transformer_heads),
            "--transformer-layers",
            str(self.cfg.train_transformer_layers),
            "--transformer-ff-dim",
            str(self.cfg.train_transformer_ff_dim),
            "--transformer-dropout",
            str(self.cfg.train_transformer_dropout),
            "--hidden-layers",
            self.cfg.train_hidden_layers,
            "--lr",
            str(self.cfg.train_lr),
            "--epochs",
            str(self.cfg.train_epochs),
            "--batch-size",
            str(self.cfg.train_batch_size),
            "--l2",
            str(self.cfg.train_l2),
            "--decision-threshold",
            str(self.cfg.train_decision_threshold),
            "--threshold-grid",
            self.cfg.train_threshold_grid,
            "--optimize-metric",
            self.cfg.train_optimize_metric,
            "--class-weight-mode",
            self.cfg.train_class_weight_mode,
            "--seed",
            str(self.cfg.train_seed),
        ]
        if self.cfg.train_use_full_window:
            train_cmd.append("--train-on-all")

        run_command(download_cmd, run_dir / "download.log", env)
        run_command(train_cmd, run_dir / "train.log", env)
        validate_model(model_path)
        activate_model(model_path, Path(self.cfg.model_path))

        summary = {
            "run_id": run_id,
            "symbol": self.cfg.symbol,
            "interval": self.cfg.interval,
            "start": iso(start_dt),
            "end": iso(end_dt),
            "lookback_days": self.cfg.retrain_lookback_days,
            "training_window": window_summary,
            "retrain_frequency": self.cfg.retrain_frequency,
            "raw_path": str(raw_path),
            "candle_cache_path": str(cache_path),
            "model_path": str(model_path),
            "active_model_path": self.cfg.model_path,
            "metrics_path": str(metrics_path),
            "train_params": {
                "model_type": self.cfg.train_model_type,
                "backend": self.cfg.train_backend,
                "device": self.cfg.train_device,
                "lookback": self.cfg.train_lookback,
                "sequence_feature_set": self.cfg.train_sequence_feature_set,
                "edge": self.cfg.train_edge,
                "split": self.cfg.train_split,
                "train_use_full_window": self.cfg.train_use_full_window,
                "cnn_filters": self.cfg.train_cnn_filters,
                "cnn_kernel_sizes": self.cfg.train_cnn_kernel_sizes,
                "lstm_hidden_size": self.cfg.train_lstm_hidden_size,
                "lstm_layers": self.cfg.train_lstm_layers,
                "lstm_dropout": self.cfg.train_lstm_dropout,
                "gru_hidden_size": self.cfg.train_gru_hidden_size,
                "gru_layers": self.cfg.train_gru_layers,
                "gru_dropout": self.cfg.train_gru_dropout,
                "transformer_d_model": self.cfg.train_transformer_d_model,
                "transformer_heads": self.cfg.train_transformer_heads,
                "transformer_layers": self.cfg.train_transformer_layers,
                "transformer_ff_dim": self.cfg.train_transformer_ff_dim,
                "transformer_dropout": self.cfg.train_transformer_dropout,
                "hidden_layers": self.cfg.train_hidden_layers,
                "lr": self.cfg.train_lr,
                "epochs": self.cfg.train_epochs,
                "batch_size": self.cfg.train_batch_size,
                "l2": self.cfg.train_l2,
                "class_weight_mode": self.cfg.train_class_weight_mode,
                "seed": self.cfg.train_seed,
            },
        }
        manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary


def live_candle_cache_path(cfg: Config) -> Path:
    symbol = cfg.symbol.upper().replace("/", "_")
    interval = cfg.interval
    return Path(cfg.retrain_cache_dir) / "binance" / symbol / interval / "cache.parquet"


def seed_live_candle_cache(cache_path: Path, training_runs_dir: Path, symbol: str, interval: str) -> None:
    """Seed the persistent cache from the newest old run, if one exists.

    Older versions wrote cache.parquet beside each run's candles.parquet, so a
    fresh persistent cache should not force another full historical download.
    """
    if cache_path.exists() or not training_runs_dir.exists():
        return

    symbol = symbol.upper()
    candidates: list[Path] = []
    for run_dir in sorted((p for p in training_runs_dir.iterdir() if p.is_dir()), reverse=True):
        candidates.extend([run_dir / "cache.parquet", run_dir / "candles.parquet"])

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            if parquet_matches_symbol_interval(candidate, symbol, interval):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, cache_path)
                print(f"Seeded live candle cache from {candidate} -> {cache_path}", flush=True)
                return
        except Exception as exc:  # noqa: BLE001 - corrupted old caches should not block retraining.
            print(f"Skipped old candle cache seed {candidate}: {exc}", flush=True)


def parquet_matches_symbol_interval(path: Path, symbol: str, interval: str) -> bool:
    import pandas as pd

    frame = pd.read_parquet(path, columns=["open_time", "symbol"])
    if frame.empty:
        return False
    if "symbol" not in frame.columns:
        return False
    symbols = {str(value).upper() for value in frame["symbol"].dropna().unique()}
    if symbol not in symbols:
        return False
    times = pd.to_datetime(frame["open_time"], utc=True).sort_values()
    diffs = times.diff().dropna().dt.total_seconds()
    if diffs.empty:
        return False
    expected_seconds = parse_interval_seconds(interval)
    median_seconds = float(diffs.median())
    return abs(median_seconds - expected_seconds) < 1e-6


def src_script(name: str) -> str:
    docker_path = Path("/app/src") / name
    if docker_path.exists():
        return str(docker_path)
    return str(Path(__file__).resolve().parents[2] / "src" / name)


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}; see {log_path}")


def prepare_torch_runtime_env(env: dict[str, str]) -> None:
    """Keep Torch from resolving a missing Docker passwd entry for HOST_UID."""
    state_dir = Path("/app/state") if Path("/app/state").exists() else Path.cwd() / "state"
    cache_dir = state_dir / ".cache"
    for path in [
        cache_dir,
        cache_dir / "torch",
        cache_dir / "torchinductor",
        cache_dir / "triton",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    env.setdefault("HOME", str(state_dir))
    env.setdefault("USER", "candscout")
    env.setdefault("LOGNAME", "candscout")
    env.setdefault("XDG_CACHE_HOME", str(cache_dir))
    env.setdefault("TORCH_HOME", str(cache_dir / "torch"))
    env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir / "torchinductor"))
    env.setdefault("TRITON_CACHE_DIR", str(cache_dir / "triton"))
    # This project does not use torch.compile; disabling dynamo avoids cache setup
    # paths that can fail for bind-mounted host UIDs inside minimal containers.
    env.setdefault("TORCHDYNAMO_DISABLE", "1")


def validate_model(path: Path) -> None:
    model = load_sequence_model(path)
    if int(model["lookback"]) < 2:
        raise RuntimeError(f"Invalid trained model lookback in {path}")


def activate_model(trained_model_path: Path, active_model_path: Path) -> None:
    active_model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = active_model_path.with_suffix(active_model_path.suffix + ".tmp")
    shutil.copy2(trained_model_path, tmp_path)
    os.replace(tmp_path, active_model_path)


def cleanup_old_runs(runs_dir: Path, keep: int) -> None:
    if keep <= 0 or not runs_dir.exists():
        return
    runs = sorted([p for p in runs_dir.iterdir() if p.is_dir()], reverse=True)
    for old_run in runs[keep:]:
        shutil.rmtree(old_run, ignore_errors=True)


def retrain_lock_path(training_runs_dir: Path) -> Path:
    return training_runs_dir.parent / "retrain.lock"


def acquire_retrain_lock(lock_path: Path) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        age_seconds = time.time() - lock_path.stat().st_mtime
        if age_seconds > 12 * 60 * 60:
            lock_path.unlink(missing_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Retraining is already running; lock exists at {lock_path}") from exc
    payload = {"pid": os.getpid(), "started_at": now_iso()}
    os.write(fd, json.dumps(payload).encode("utf-8"))
    return fd


def release_retrain_lock(lock_fd: int, lock_path: Path) -> None:
    os.close(lock_fd)
    lock_path.unlink(missing_ok=True)


def next_daily_run_utc(raw_time: str) -> datetime:
    hour, minute = parse_hhmm(raw_time)
    now = datetime.now(timezone.utc)
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def next_retrain_run_utc(cfg: Config, store: Store) -> datetime:
    amount, unit = parse_retrain_frequency(cfg.retrain_frequency)
    if unit == "d" and amount == 1:
        return next_daily_run_utc(cfg.retrain_time_utc)

    now = datetime.now(timezone.utc)
    latest = store.latest_training_run()
    base = None
    if latest:
        for key in ("ended_at", "started_at"):
            if latest.get(key):
                try:
                    base = parse_utc_datetime(str(latest[key]))
                    break
                except ValueError:
                    continue
    if base is None:
        base = now

    candidate = add_retrain_frequency(base, amount, unit)
    while candidate <= now:
        candidate = add_retrain_frequency(candidate, amount, unit)
    return candidate


def add_retrain_frequency(base: datetime, amount: int, unit: str) -> datetime:
    base = base.astimezone(timezone.utc)
    if unit == "h":
        return base + timedelta(hours=amount)
    if unit == "d":
        return base + timedelta(days=amount)
    if unit == "w":
        return base + timedelta(weeks=amount)
    if unit == "m":
        return add_months(base, amount)
    raise ValueError(f"Unsupported retrain frequency unit: {unit}")


def add_months(base: datetime, months: int) -> datetime:
    month_index = base.month - 1 + months
    year = base.year + month_index // 12
    month = month_index % 12 + 1
    day = min(base.day, days_in_month(year, month))
    return base.replace(year=year, month=month, day=day)


def days_in_month(year: int, month: int) -> int:
    if month == 12:
        next_month = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        next_month = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    this_month = datetime(year, month, 1, tzinfo=timezone.utc)
    return (next_month - this_month).days


def resolve_training_window(cfg: Config) -> tuple[datetime, datetime, dict[str, Any]]:
    end_dt = floor_to_interval(datetime.now(timezone.utc), cfg.interval)
    if cfg.retrain_train_start and cfg.retrain_train_end:
        configured_start = parse_utc_datetime(cfg.retrain_train_start)
        configured_end = parse_utc_datetime(cfg.retrain_train_end)
        duration = configured_end - configured_start
        if duration.total_seconds() <= 0:
            raise ValueError("RETRAIN_TRAIN_END must be after RETRAIN_TRAIN_START")
        start_dt = end_dt - duration
        return start_dt, end_dt, {
            "mode": "rolling_duration_from_configured_dates",
            "configured_start": iso(configured_start),
            "configured_end": iso(configured_end),
            "duration_seconds": int(duration.total_seconds()),
            "duration_days": duration.total_seconds() / 86400.0,
            "end_aligned_to_interval": cfg.interval,
        }

    start_dt = end_dt - timedelta(days=cfg.retrain_lookback_days)
    return start_dt, end_dt, {
        "mode": "rolling_lookback_days",
        "lookback_days": cfg.retrain_lookback_days,
        "duration_seconds": int((end_dt - start_dt).total_seconds()),
        "duration_days": cfg.retrain_lookback_days,
        "end_aligned_to_interval": cfg.interval,
    }


def parse_hhmm(raw: str) -> tuple[int, int]:
    parts = raw.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM time, got {raw!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"Invalid HH:MM time: {raw!r}")
    return hour, minute


def parse_utc_datetime(raw: str) -> datetime:
    text = raw.strip()
    if not text:
        raise ValueError("UTC datetime cannot be blank")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"Invalid UTC datetime: {raw!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def floor_to_interval(dt: datetime, interval: str) -> datetime:
    dt = dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
    seconds = parse_interval_seconds(interval)
    timestamp = int(dt.timestamp())
    return datetime.fromtimestamp(timestamp - (timestamp % seconds), tz=timezone.utc)


def now_iso() -> str:
    return iso(datetime.now(timezone.utc))


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
