#!/usr/bin/env python3
"""Create rule-based strategy model artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from strategy_model.strategies import (  # noqa: E402
    STRATEGY_LABELS,
    load_raw_candles,
    split_frame,
    strategy_scores,
    write_model,
)
from train import class_balance, classification_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/configure a rule-based strategy model.")
    parser.add_argument("--raw-data", required=True, help="Raw candle Parquet path")
    parser.add_argument(
        "--model-type",
        choices=["buy_hold", "prev_movement", "ma"],
        required=True,
        help="Strategy model type",
    )
    parser.add_argument("--model-out", required=True, help="Strategy model JSON output")
    parser.add_argument("--metrics-out", required=True, help="Training metrics JSON output")
    parser.add_argument("--split", type=float, default=0.95, help="Chronological train split")
    parser.add_argument("--edge", type=float, default=0.0003, help="Label edge for target metrics")
    parser.add_argument("--ma-window", type=int, default=20, help="Moving-average window for the MA strategy")
    parser.add_argument("--threshold", type=float, default=0.52, help="Entry score threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_data)
    df = load_raw_candles(raw_path, edge=args.edge)
    if len(df) < 50:
        raise ValueError("Need at least 50 rows for strategy training/evaluation")

    train_df, test_df, split_idx = split_frame(df, args.split)
    selected_window = int(args.ma_window)

    scores = strategy_scores(df, args.model_type, selected_window, 0.0)
    test_scores = scores[split_idx:]
    y_test = test_df["target"].to_numpy(dtype=np.int64)
    y_pred = (test_scores >= args.threshold).astype(int)
    comparison = {
        "always_positive": classification_metrics(y_test, np.ones_like(y_test)),
        "always_negative": classification_metrics(y_test, np.zeros_like(y_test)),
        args.model_type: classification_metrics(y_test, y_pred),
    }

    model = {
        "model_family": "strategy",
        "model_type": args.model_type,
        "label": STRATEGY_LABELS.get(args.model_type, args.model_type),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raw_data": str(raw_path),
        "edge": float(args.edge),
        "split": float(args.split),
        "selected_window": int(selected_window),
        "ma_window": int(args.ma_window),
        "score_meaning": {
            "1.0": "buy-and-hold long",
            "0.8": "long / buy zone",
            "0.5": "neutral / hold current state",
            "0.2": "cash / sell zone",
        },
    }
    write_model(Path(args.model_out), model)

    metrics = {
        "model_type": f"strategy_{args.model_type}",
        "strategy": model,
        "primary_objective": "rule_strategy_baseline_or_train_split_window_selection",
        "dataset": {
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "split_fraction": float(args.split),
            "first_open_time": str(df["open_time"].iloc[0]),
            "last_open_time": str(df["open_time"].iloc[-1]),
        },
        "class_balance": {
            "overall": class_balance(df["target"].to_numpy(dtype=np.int64)),
            "train": class_balance(train_df["target"].to_numpy(dtype=np.int64)),
            "test": class_balance(y_test),
        },
        "model_training": {
            "backend": "rule_based",
            "loss_history": [],
            "final_loss": None,
            "selected_window": int(selected_window),
        },
        "baseline_vs_model": comparison,
    }
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved strategy model to {args.model_out}")
    print(f"Saved strategy metrics to {args.metrics_out}")
    print(f"Strategy type: {args.model_type}")
    print(f"Selected MA window: {selected_window}")


if __name__ == "__main__":
    main()
