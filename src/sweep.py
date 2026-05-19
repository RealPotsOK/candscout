#!/usr/bin/env python3
"""Run grid sweeps over label edge, candle-window features, and decision thresholds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

FEATURE_PRESETS = {
    "default": {
        "return_windows": "1,3,5,15,30,60",
        "vol_windows": "5,15,30,60",
        "sma_short_window": 5,
        "sma_long_window": 20,
        "extra_sma_windows": "50,100",
        "volume_z_window": 20,
        "volume_ratio_windows": "20,60",
        "include_time_features": True,
    },
    "fast": {
        "return_windows": "1,2,3,5,10,20",
        "vol_windows": "3,5,10,20",
        "sma_short_window": 3,
        "sma_long_window": 12,
        "extra_sma_windows": "24,48",
        "volume_z_window": 10,
        "volume_ratio_windows": "10,30",
        "include_time_features": True,
    },
    "slow": {
        "return_windows": "1,5,15,30,60,120",
        "vol_windows": "10,30,60,120",
        "sma_short_window": 10,
        "sma_long_window": 40,
        "extra_sma_windows": "100,200",
        "volume_z_window": 30,
        "volume_ratio_windows": "30,120",
        "include_time_features": True,
    },
}


def parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def parse_string_list(raw: str) -> list[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one string value")
    return values


def run_command(cmd: list[str]) -> None:
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{process.stdout}\n"
            f"STDERR:\n{process.stderr}"
        )


def tag_for_run(preset: str, edge: float) -> str:
    edge_tag = str(edge).replace(".", "p")
    return f"{preset}_edge_{edge_tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep feature presets and label edges.")
    parser.add_argument("--raw-input", required=True, help="Raw candle Parquet path")
    parser.add_argument("--interval", default="5m", help="Candle interval label forwarded to features metadata")
    parser.add_argument("--output-dir", default="models/sweeps_5m", help="Sweep output directory")
    parser.add_argument("--edges", default="0.0015,0.002,0.0025", help="Comma-separated edge values")
    parser.add_argument(
        "--feature-presets",
        default="default,fast,slow",
        help="Comma-separated preset names (default,fast,slow)",
    )
    parser.add_argument(
        "--threshold-grid",
        default="0.30,0.40,0.50,0.55,0.60,0.70",
        help="Threshold grid forwarded to train.py",
    )
    parser.add_argument(
        "--optimize-metric",
        choices=["f1_y1", "recall_y1", "precision_y1", "accuracy"],
        default="f1_y1",
        help="Metric used for best-threshold selection",
    )
    parser.add_argument("--split", type=float, default=0.8, help="Chronological train split fraction")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization")
    parser.add_argument(
        "--class-weight-mode",
        choices=["none", "balanced", "manual"],
        default="balanced",
        help="Class weighting mode for train.py",
    )
    parser.add_argument("--pos-weight", type=float, default=None, help="Manual pos weight when mode=manual")
    parser.add_argument("--fee", type=float, default=0.001, help="Per-side fee for backtest")
    parser.add_argument(
        "--rank-by",
        choices=["best_f1_y1", "best_recall_y1", "best_precision_y1", "best_accuracy"],
        default="best_f1_y1",
        help="Primary ranking metric for sweep summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    edges = parse_float_list(args.edges)
    preset_names = parse_string_list(args.feature_presets)

    unknown = [name for name in preset_names if name not in FEATURE_PRESETS]
    if unknown:
        raise ValueError(f"Unknown feature presets: {unknown}. Known: {sorted(FEATURE_PRESETS.keys())}")

    output_dir = Path(args.output_dir)
    features_dir = output_dir / "features"
    models_dir = output_dir / "models"
    metrics_dir = output_dir / "metrics"
    backtests_dir = output_dir / "backtests"

    for path in [features_dir, models_dir, metrics_dir, backtests_dir]:
        path.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    python_bin = sys.executable

    summary_rows: list[dict] = []

    for preset_name in preset_names:
        preset = FEATURE_PRESETS[preset_name]

        for edge in edges:
            run_tag = tag_for_run(preset_name, edge)
            feature_path = features_dir / f"{run_tag}.parquet"
            feature_meta_path = features_dir / f"{run_tag}.meta.json"
            model_path = models_dir / f"{run_tag}.npz"
            metrics_path = metrics_dir / f"{run_tag}.json"
            backtest_path = backtests_dir / f"{run_tag}.json"

            features_cmd = [
                python_bin,
                str(script_dir / "features.py"),
                "--input",
                args.raw_input,
                "--output",
                str(feature_path),
                "--meta-out",
                str(feature_meta_path),
                "--edge",
                str(edge),
                "--interval",
                args.interval,
                "--return-windows",
                preset["return_windows"],
                "--vol-windows",
                preset["vol_windows"],
                "--sma-short-window",
                str(preset["sma_short_window"]),
                "--sma-long-window",
                str(preset["sma_long_window"]),
                "--extra-sma-windows",
                preset["extra_sma_windows"],
                "--volume-z-window",
                str(preset["volume_z_window"]),
                "--volume-ratio-windows",
                preset["volume_ratio_windows"],
            ]
            if preset["include_time_features"]:
                features_cmd.append("--include-time-features")
            else:
                features_cmd.append("--no-include-time-features")
            run_command(features_cmd)

            train_cmd = [
                python_bin,
                str(script_dir / "train.py"),
                "--features",
                str(feature_path),
                "--feature-meta",
                str(feature_meta_path),
                "--model-out",
                str(model_path),
                "--metrics-out",
                str(metrics_path),
                "--split",
                str(args.split),
                "--lr",
                str(args.lr),
                "--epochs",
                str(args.epochs),
                "--l2",
                str(args.l2),
                "--threshold-grid",
                args.threshold_grid,
                "--optimize-metric",
                args.optimize_metric,
                "--class-weight-mode",
                args.class_weight_mode,
            ]
            if args.pos_weight is not None:
                train_cmd.extend(["--pos-weight", str(args.pos_weight)])
            run_command(train_cmd)

            metrics_payload = json.loads(metrics_path.read_text())
            best_threshold = float(metrics_payload["threshold_sweep"]["best"]["threshold"])

            backtest_cmd = [
                python_bin,
                str(script_dir / "backtest.py"),
                "--features",
                str(feature_path),
                "--model",
                str(model_path),
                "--fee",
                str(args.fee),
                "--threshold",
                str(best_threshold),
                "--report-out",
                str(backtest_path),
            ]
            run_command(backtest_cmd)

            backtest_payload = json.loads(backtest_path.read_text())
            best = metrics_payload["threshold_sweep"]["best"]
            backtest_logreg = backtest_payload["backtest"]["logistic_regression"]

            summary_rows.append(
                {
                    "run_tag": run_tag,
                    "preset": preset_name,
                    "edge": float(edge),
                    "feature_count": len(metrics_payload["feature_columns"]),
                    "test_positive_rate": float(metrics_payload["class_balance"]["test"]["positive_rate"]),
                    "pos_weight_used": float(metrics_payload["model_training"]["pos_weight_used"]),
                    "best_threshold": best_threshold,
                    "best_accuracy": float(best["accuracy"]),
                    "best_precision_y1": float(best["precision_y1"]),
                    "best_recall_y1": float(best["recall_y1"]),
                    "best_f1_y1": float(best["f1_y1"]),
                    "backtest_trade_count": int(backtest_logreg["trade_count"]),
                    "backtest_total_return": float(backtest_logreg["total_return"]),
                    "backtest_hit_rate": float(backtest_logreg["hit_rate"]),
                    "metrics_path": str(metrics_path),
                    "backtest_path": str(backtest_path),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        by=[args.rank_by, "best_recall_y1", "best_precision_y1", "best_accuracy"],
        ascending=False,
    ).reset_index(drop=True)

    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(summary_df.to_json(orient="records", indent=2))

    print(f"Sweep completed with {len(summary_df)} runs")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print("Top 5 runs:")
    print(summary_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
