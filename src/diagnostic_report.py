#!/usr/bin/env python3
"""Generate diagnostics for class balance, probability distribution, and threshold quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one float threshold value")
    if any(v <= 0.0 or v >= 1.0 for v in values):
        raise ValueError("Threshold values must be strictly between 0 and 1")
    return values


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def resolve_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing required column for {purpose}. Tried: {candidates}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create diagnostic report for trained model outputs.")
    parser.add_argument("--features", required=True, help="Feature Parquet path")
    parser.add_argument("--model", required=True, help="Trained model .npz path")
    parser.add_argument("--split", type=float, default=0.8, help="Chronological train split fraction")
    parser.add_argument("--fee", type=float, default=0.001, help="Per-side fee rate")
    parser.add_argument(
        "--thresholds",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99",
        help="Comma-separated thresholds for diagnostic sweep",
    )
    parser.add_argument(
        "--report-out",
        default="models/diagnostic_report_5m.json",
        help="Diagnostic JSON output path",
    )
    parser.add_argument(
        "--threshold-table-out",
        default="models/diagnostic_threshold_sweep_5m.csv",
        help="Threshold sweep CSV output path",
    )
    parser.add_argument(
        "--test-predictions-out",
        default="data/reports/test_predictions_diagnostic_5m.parquet",
        help="Test-only predictions Parquet output",
    )
    return parser.parse_args()


def class_balance(y: np.ndarray) -> dict:
    if len(y) == 0:
        return {"positive_rate": 0.0, "negative_rate": 0.0, "positive_count": 0, "negative_count": 0}
    positive_count = int(np.sum(y == 1))
    negative_count = int(np.sum(y == 0))
    positive_rate = float(positive_count / len(y))
    return {
        "positive_rate": positive_rate,
        "negative_rate": float(1.0 - positive_rate),
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def prob_distribution(probs: np.ndarray) -> dict:
    series = pd.Series(probs)
    return {
        "min": float(series.min()),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
    }


def threshold_row(
    y_true: np.ndarray,
    forward_returns: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    fee: float,
) -> dict:
    signal = probs >= threshold
    signal_count = int(np.sum(signal))

    total_pos = int(np.sum(y_true == 1))

    if signal_count == 0:
        return {
            "threshold": float(threshold),
            "signal_count": 0,
            "precision": 0.0,
            "recall": 0.0,
            "label_hit_rate": 0.0,
            "profitable_trade_rate": 0.0,
            "avg_forward_return": 0.0,
            "avg_net_return_per_trade": 0.0,
            "total_compounded_return": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": total_pos,
            "tn": int(len(y_true) - total_pos),
        }

    y_sig = y_true[signal]
    fwd_sig = forward_returns[signal]
    net_sig = fwd_sig - (2.0 * fee)

    tp = int(np.sum(y_sig == 1))
    fp = int(signal_count - tp)
    fn = int(total_pos - tp)
    tn = int(len(y_true) - tp - fp - fn)

    precision = float(tp / signal_count) if signal_count else 0.0
    recall = float(tp / total_pos) if total_pos else 0.0
    label_hit_rate = precision
    profitable_trade_rate = float(np.mean(net_sig > 0.0))

    compounded = float(np.prod(1.0 + net_sig) - 1.0)

    return {
        "threshold": float(threshold),
        "signal_count": signal_count,
        "precision": precision,
        "recall": recall,
        "label_hit_rate": label_hit_rate,
        "profitable_trade_rate": profitable_trade_rate,
        "avg_forward_return": float(np.mean(fwd_sig)),
        "avg_net_return_per_trade": float(np.mean(net_sig)),
        "total_compounded_return": compounded,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    args = parse_args()

    if not (0.0 < args.split < 1.0):
        raise ValueError("--split must be between 0 and 1")

    thresholds = parse_float_list(args.thresholds)

    df = pd.read_parquet(args.features).sort_values("open_time").reset_index(drop=True)
    required_base = {"open_time", "close", "target"}
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise ValueError(f"Feature dataset missing required columns: {sorted(missing_base)}")
    forward_return_col = resolve_column(
        df,
        ["forward_return", "forward_return_1m"],
        "next-candle forward return",
    )

    model = np.load(args.model)
    feature_names = [str(x) for x in model["feature_names"]]
    missing_model_cols = set(feature_names) - set(df.columns)
    if missing_model_cols:
        raise ValueError(f"Feature dataset missing model columns: {sorted(missing_model_cols)}")

    n_rows = len(df)
    split_idx = int(n_rows * args.split)
    split_idx = max(1, min(n_rows - 1, split_idx))

    x = df[feature_names].to_numpy(dtype=np.float64)
    mean = model["mean"]
    std = model["std"]
    bias = float(model["bias"][0])
    weights = model["weights"]

    x_norm = (x - mean) / std
    probs_all = sigmoid(x_norm @ weights + bias)

    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    probs_test = probs_all[split_idx:]
    y_test = test_df["target"].to_numpy(dtype=np.int64)
    fwd_test = test_df[forward_return_col].to_numpy(dtype=np.float64)

    dataset_balance = class_balance(df["target"].to_numpy(dtype=np.int64))
    test_balance = class_balance(y_test)

    rows = [
        threshold_row(
            y_true=y_test,
            forward_returns=fwd_test,
            probs=probs_test,
            threshold=t,
            fee=args.fee,
        )
        for t in thresholds
    ]

    threshold_df = pd.DataFrame(rows)

    predictions_out = Path(args.test_predictions_out)
    predictions_out.parent.mkdir(parents=True, exist_ok=True)

    pred_frame = pd.DataFrame(
        {
            "open_time": test_df["open_time"],
            "close": test_df["close"],
            "target": test_df["target"].astype(int),
            "forward_return": test_df[forward_return_col],
            "prob_up": probs_test,
            "predicted_class_at_0_50": (probs_test >= 0.50).astype(int),
        }
    )

    for t in thresholds:
        col = f"signal_at_{str(t).replace('.', '_')}"
        pred_frame[col] = (probs_test >= t).astype(int)

    pred_frame.to_parquet(predictions_out, index=False)

    report = {
        "inputs": {
            "features": str(args.features),
            "model": str(args.model),
            "split": float(args.split),
            "fee_per_side": float(args.fee),
            "thresholds": thresholds,
            "test_rows": int(len(test_df)),
            "dataset_rows": int(len(df)),
            "train_rows": int(split_idx),
            "forward_return_column": forward_return_col,
        },
        "A_class_balance": {
            "dataset": dataset_balance,
            "test": test_balance,
        },
        "B_test_probability_distribution": prob_distribution(probs_test),
        "C_threshold_sweep": rows,
        "D_hit_rate_definition": {
            "label_hit_rate": "Among signaled trades, fraction where target==1 (classification precision on traded rows).",
            "profitable_trade_rate": "Among signaled trades, fraction where forward_return - 2*fee > 0.",
            "note": "These are different metrics and should both be tracked.",
        },
        "artifacts": {
            "threshold_table_csv": str(args.threshold_table_out),
            "test_predictions_parquet": str(predictions_out),
        },
    }

    threshold_out = Path(args.threshold_table_out)
    threshold_out.parent.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(threshold_out, index=False)

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2))

    print(f"Saved diagnostic report to {report_out}")
    print(f"Saved threshold sweep table to {threshold_out}")
    print(f"Saved test predictions to {predictions_out}")
    print(f"Dataset positive_rate: {dataset_balance['positive_rate']:.6f}")
    print(f"Test positive_rate:    {test_balance['positive_rate']:.6f}")


if __name__ == "__main__":
    main()
