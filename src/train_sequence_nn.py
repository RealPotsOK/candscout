#!/usr/bin/env python3
"""Train a from-scratch NumPy sequence neural net on raw candle sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sequence_data import build_sequence_dataset, flatten_sequences, load_raw_candles
from sequence_nn import (
    parse_hidden_layers,
    parse_int_list,
    predict_cnn_proba,
    predict_proba,
    save_cnn_sequence_model,
    save_sequence_model,
    train_cnn,
    train_mlp,
)
from train import (
    class_balance,
    classification_metrics,
    parse_float_list,
    pick_best_threshold,
    resolve_pos_weight,
    threshold_sweep_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy sequence neural net on past candle sequences.")
    parser.add_argument("--raw-data", required=True, help="Raw candle Parquet path")
    parser.add_argument("--model-out", default="models/seq_nn_5m.npz", help="Sequence model output path")
    parser.add_argument("--metrics-out", default="models/seq_nn_train_metrics_5m.json", help="Metrics JSON output")
    parser.add_argument("--model-type", choices=["cnn", "mlp"], default="cnn", help="Sequence model architecture")
    parser.add_argument("--lookback", type=int, default=50, help="Past candles per prediction")
    parser.add_argument("--edge", type=float, default=0.0015, help="Positive label edge on next-candle return")
    parser.add_argument("--split", type=float, default=0.8, help="Chronological train split fraction")
    parser.add_argument("--cnn-filters", default="16,32", help="Comma-separated Conv1D filter counts")
    parser.add_argument("--cnn-kernel-sizes", default="5,3", help="Comma-separated Conv1D kernel sizes")
    parser.add_argument("--hidden-layers", default="32,16", help="Comma-separated dense hidden layer sizes")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    parser.add_argument("--epochs", type=int, default=25, help="Full passes through the train set")
    parser.add_argument("--batch-size", type=int, default=2048, help="Mini-batch size")
    parser.add_argument("--l2", type=float, default=0.00001, help="L2 regularization")
    parser.add_argument("--decision-threshold", type=float, default=0.55, help="Probability threshold for y=1")
    parser.add_argument(
        "--threshold-grid",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99",
        help="Comma-separated threshold sweep values",
    )
    parser.add_argument(
        "--optimize-metric",
        choices=["f1_y1", "recall_y1", "precision_y1", "accuracy"],
        default="f1_y1",
        help="Metric used to choose best threshold from threshold-grid",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=["none", "balanced", "manual"],
        default="balanced",
        help="Class weighting for minority positive labels",
    )
    parser.add_argument("--pos-weight", type=float, default=None, help="Manual positive class weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def normalize_flat_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std == 0.0] = 1.0
    x_train_norm = ((x_train - mean) / std).astype(np.float32, copy=False)
    x_test_norm = ((x_test - mean) / std).astype(np.float32, copy=False)
    return x_train_norm, x_test_norm, mean, std


def normalize_sequence_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std[std == 0.0] = 1.0
    x_train_norm = ((x_train - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32, copy=False)
    x_test_norm = ((x_test - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32, copy=False)
    return x_train_norm, x_test_norm, mean, std


def main() -> None:
    args = parse_args()
    if not (0.0 < args.split < 1.0):
        raise ValueError("--split must be between 0 and 1")
    if args.lookback < 2:
        raise ValueError("--lookback must be at least 2")

    thresholds = parse_float_list(args.threshold_grid)
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    cnn_filters = parse_int_list(args.cnn_filters, "CNN filter count")
    cnn_kernel_sizes = parse_int_list(args.cnn_kernel_sizes, "CNN kernel size")
    if len(cnn_filters) != len(cnn_kernel_sizes):
        raise ValueError("--cnn-filters and --cnn-kernel-sizes must have the same number of entries")

    candles = load_raw_candles(Path(args.raw_data))
    x_seq, y, meta, channel_names = build_sequence_dataset(candles, lookback=args.lookback, edge=args.edge)

    n_rows = len(x_seq)
    if n_rows < 50:
        raise ValueError("Need at least 50 sequence rows to train and evaluate.")

    split_idx = int(n_rows * args.split)
    split_idx = max(1, min(n_rows - 1, split_idx))

    x_train_seq = x_seq[:split_idx]
    x_test_seq = x_seq[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:].astype(np.int64)
    pos_weight = resolve_pos_weight(y_train, args.class_weight_mode, args.pos_weight)
    model_key = f"sequence_{args.model_type}"

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    if args.model_type == "mlp":
        x = flatten_sequences(x_seq)
        x_train_norm, x_test_norm, mean, std = normalize_flat_train_test(x[:split_idx], x[split_idx:])
        weights, biases, loss_history = train_mlp(
            x_train=x_train_norm,
            y_train=y_train,
            hidden_layers=hidden_layers,
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            l2=args.l2,
            pos_weight=pos_weight,
            seed=args.seed,
        )
        model_probs = predict_proba(x_test_norm, weights, biases)
        save_sequence_model(
            model_out,
            weights,
            biases,
            input_mean=mean,
            input_std=std,
            lookback=args.lookback,
            channel_names=channel_names,
            hidden_layers=hidden_layers,
            edge=args.edge,
        )
        input_metadata = {
            "lookback": int(args.lookback),
            "channels": channel_names,
            "input_shape": [int(args.lookback), int(len(channel_names))],
            "input_dim": int(x.shape[1]),
            "normalization": "flattened feature mean/std from train split only",
            "definition": "Each row receives the last lookback candles as per-candle OHLC returns vs previous close plus log volume, flattened before MLP input.",
        }
        architecture_metadata = {"hidden_layers": hidden_layers}
    else:
        x_train_norm, x_test_norm, mean, std = normalize_sequence_train_test(x_train_seq, x_test_seq)
        conv_weights, conv_biases, dense_weights, dense_biases, loss_history = train_cnn(
            x_train=x_train_norm,
            y_train=y_train,
            cnn_filters=cnn_filters,
            cnn_kernel_sizes=cnn_kernel_sizes,
            hidden_layers=hidden_layers,
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            l2=args.l2,
            pos_weight=pos_weight,
            seed=args.seed,
        )
        model_probs = predict_cnn_proba(x_test_norm, conv_weights, conv_biases, dense_weights, dense_biases)
        save_cnn_sequence_model(
            model_out,
            conv_weights=conv_weights,
            conv_biases=conv_biases,
            dense_weights=dense_weights,
            dense_biases=dense_biases,
            input_mean=mean,
            input_std=std,
            lookback=args.lookback,
            channel_names=channel_names,
            cnn_filters=cnn_filters,
            cnn_kernel_sizes=cnn_kernel_sizes,
            hidden_layers=hidden_layers,
            edge=args.edge,
        )
        input_metadata = {
            "lookback": int(args.lookback),
            "channels": channel_names,
            "input_shape": [int(args.lookback), int(len(channel_names))],
            "normalization": "per-channel mean/std from train split only, preserving [lookback, channels] shape",
            "definition": "Each row receives the last lookback candles as per-candle OHLC returns vs previous close plus log volume.",
        }
        architecture_metadata = {
            "cnn_filters": cnn_filters,
            "cnn_kernel_sizes": cnn_kernel_sizes,
            "hidden_layers": hidden_layers,
        }

    model_preds = (model_probs >= args.decision_threshold).astype(int)
    test_meta = meta.iloc[split_idx:].reset_index(drop=True)

    baseline_preds = {
        "always_positive": np.ones_like(y_test, dtype=int),
        "always_negative": np.zeros_like(y_test, dtype=int),
        "prev_candle_direction": (test_meta["return_1bar"].to_numpy(dtype=np.float64) > 0.0).astype(int),
        "ma_direction": (test_meta["sma_spread"].to_numpy(dtype=np.float64) > 0.0).astype(int),
        model_key: model_preds,
    }

    comparison = {name: classification_metrics(y_test, preds) for name, preds in baseline_preds.items()}
    sweep_rows = threshold_sweep_metrics(y_test, model_probs, thresholds)
    best_row = pick_best_threshold(sweep_rows, args.optimize_metric)

    metrics = {
        "primary_objective": "predictive_validity_vs_baselines",
        "secondary_objective": "fee_aware_backtest_reality_check",
        "model_type": model_key,
        "dataset": {
            "raw_rows": int(len(candles)),
            "sequence_rows_total": int(n_rows),
            "rows_train": int(split_idx),
            "rows_test": int(n_rows - split_idx),
            "split_fraction": float(args.split),
        },
        "sequence_input": input_metadata,
        "target": {
            "edge": float(args.edge),
            "definition": "target=1 when next-candle forward_return > edge, else 0",
        },
        "class_balance": {
            "overall": class_balance(y.astype(np.int64)),
            "train": class_balance(y_train.astype(np.int64)),
            "test": class_balance(y_test),
        },
        "model_training": {
            "model_type": model_key,
            **architecture_metadata,
            "learning_rate": float(args.lr),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "l2": float(args.l2),
            "decision_threshold": float(args.decision_threshold),
            "class_weight_mode": args.class_weight_mode,
            "pos_weight_used": float(pos_weight),
            "seed": int(args.seed),
            "initial_loss": float(loss_history[0]),
            "final_loss": float(loss_history[-1]),
            "loss_history": loss_history,
        },
        "baseline_vs_model": comparison,
        "threshold_sweep": {
            "optimize_metric": args.optimize_metric,
            "rows": sweep_rows,
            "best": best_row,
        },
    }

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2))

    print(f"Saved sequence model to {model_out}")
    print(f"Saved sequence metrics to {metrics_out}")
    print(f"{model_key} test metrics at decision-threshold:")
    print(json.dumps(comparison[model_key], indent=2))
    print("Best threshold from sweep:")
    print(json.dumps(best_row, indent=2))


if __name__ == "__main__":
    main()
