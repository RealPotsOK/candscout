#!/usr/bin/env python3
"""Train MLP, CNN, GRU, LSTM, or Transformer models on candle sequences."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from sequence_data import build_sequence_dataset, flatten_sequences, load_raw_candles
from sequence_nn import (
    parse_hidden_layers,
    parse_int_list,
    predict_cnn_proba,
    predict_proba,
    save_cnn_sequence_model,
    save_gru_sequence_model,
    save_lstm_sequence_model,
    save_sequence_model,
    save_transformer_sequence_model,
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
    parser = argparse.ArgumentParser(description="Train a sequence neural net on past candle sequences.")
    parser.add_argument("--raw-data", required=True, help="Raw candle Parquet path")
    parser.add_argument("--model-out", default="models/seq_nn_5m.npz", help="Sequence model output path")
    parser.add_argument("--metrics-out", default="models/seq_nn_train_metrics_5m.json", help="Metrics JSON output")
    parser.add_argument(
        "--model-type",
        choices=["cnn", "mlp", "gru", "lstm", "transformer"],
        default="cnn",
        help="Sequence model architecture",
    )
    parser.add_argument(
        "--backend",
        choices=["numpy", "torch"],
        default="numpy",
        help="Training backend. torch can use CUDA, but still saves a compatible .npz artifact.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for --backend torch. Default is cuda; CPU is intentionally rejected.",
    )
    parser.add_argument("--lookback", type=int, default=50, help="Past candles per prediction")
    parser.add_argument(
        "--sequence-feature-set",
        choices=["basic", "technical"],
        default="basic",
        help="Sequence channels: basic OHLCV-relative channels or expanded technical indicators",
    )
    parser.add_argument("--edge", type=float, default=0.0015, help="Positive label edge on next-candle return")
    parser.add_argument("--short-edge", type=float, default=None, help="Short label edge. Defaults to --edge.")
    parser.add_argument(
        "--target-direction",
        choices=["both", "up", "down"],
        default="both",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--split", type=float, default=0.8, help="Chronological train split fraction")
    parser.add_argument(
        "--train-on-all",
        action="store_true",
        help="Train on the full dataset and report in-sample metrics; intended for live rolling retrains.",
    )
    parser.add_argument("--cnn-filters", default="16,32", help="Comma-separated Conv1D filter counts")
    parser.add_argument("--cnn-kernel-sizes", default="5,3", help="Comma-separated Conv1D kernel sizes")
    parser.add_argument("--lstm-hidden-size", type=int, default=64, help="LSTM recurrent hidden size")
    parser.add_argument("--lstm-layers", type=int, default=1, help="Number of stacked LSTM layers")
    parser.add_argument("--lstm-dropout", type=float, default=0.0, help="Dropout between LSTM layers; only active for >1 layer")
    parser.add_argument("--gru-hidden-size", type=int, default=64, help="GRU recurrent hidden size")
    parser.add_argument("--gru-layers", type=int, default=1, help="Number of stacked GRU layers")
    parser.add_argument("--gru-dropout", type=float, default=0.0, help="Dropout between GRU layers; only active for >1 layer")
    parser.add_argument("--transformer-d-model", type=int, default=64, help="Transformer embedding width")
    parser.add_argument("--transformer-heads", type=int, default=4, help="Transformer attention heads")
    parser.add_argument("--transformer-layers", type=int, default=2, help="Transformer encoder layers")
    parser.add_argument("--transformer-ff-dim", type=int, default=128, help="Transformer feed-forward width")
    parser.add_argument("--transformer-dropout", type=float, default=0.1, help="Transformer attention/head dropout")
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


def down_npz_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_down{path.suffix}")


def down_json_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_down.json")


def run_dual_training(args: argparse.Namespace) -> None:
    model_out = Path(args.model_out)
    metrics_out = Path(args.metrics_out)
    def dedupe_command(cmd: list[str]) -> list[str]:
        skip_next = False
        cleaned: list[str] = []
        replace_flags = {"--target-direction", "--model-out", "--metrics-out", "--seed"}
        for idx, part in enumerate(cmd):
            if skip_next:
                skip_next = False
                continue
            if part in replace_flags:
                skip_next = True
                continue
            cleaned.append(part)
        return cleaned

    base_args = dedupe_command(sys.argv[1:])
    up_cmd = [
        sys.executable,
        __file__,
        *base_args,
        "--target-direction",
        "up",
        "--model-out",
        str(model_out),
        "--metrics-out",
        str(metrics_out),
        "--seed",
        str(args.seed),
    ]
    down_cmd = [
        sys.executable,
        __file__,
        *base_args,
        "--target-direction",
        "down",
        "--model-out",
        str(down_npz_path(model_out)),
        "--metrics-out",
        str(down_json_path(metrics_out)),
        "--seed",
        str(args.seed + 1000),
    ]
    subprocess.run(up_cmd, check=True)
    subprocess.run(down_cmd, check=True)
    up_metrics = json.loads(metrics_out.read_text())
    down_metrics = json.loads(down_json_path(metrics_out).read_text())
    up_metrics["dual_targets"] = {
        "target_up": {
            "class_balance": up_metrics.get("class_balance", {}),
            "metrics": up_metrics.get("baseline_vs_model", {}).get(up_metrics.get("model_type", ""), {}),
            "threshold_sweep": up_metrics.get("threshold_sweep", {}),
        },
        "target_down": {
            "class_balance": down_metrics.get("class_balance", {}),
            "metrics": down_metrics.get("baseline_vs_model", {}).get(down_metrics.get("model_type", ""), {}),
            "threshold_sweep": down_metrics.get("threshold_sweep", {}),
        },
    }
    up_metrics["down_model"] = {
        "model_path": str(down_npz_path(model_out)),
        "metrics_path": str(down_json_path(metrics_out)),
        "target": down_metrics.get("target", {}),
    }
    metrics_out.write_text(json.dumps(up_metrics, indent=2))
    print(f"Saved dual sequence metrics to {metrics_out}")


def main() -> None:
    args = parse_args()
    if args.target_direction == "both":
        run_dual_training(args)
        return
    if not args.train_on_all and not (0.0 < args.split < 1.0):
        raise ValueError("--split must be between 0 and 1")
    if args.lookback < 2:
        raise ValueError("--lookback must be at least 2")

    thresholds = parse_float_list(args.threshold_grid)
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    cnn_filters = parse_int_list(args.cnn_filters, "CNN filter count")
    cnn_kernel_sizes = parse_int_list(args.cnn_kernel_sizes, "CNN kernel size")
    if len(cnn_filters) != len(cnn_kernel_sizes):
        raise ValueError("--cnn-filters and --cnn-kernel-sizes must have the same number of entries")
    if args.model_type in {"gru", "lstm", "transformer"} and args.backend != "torch":
        raise ValueError(f"--model-type {args.model_type} requires --backend torch")
    if args.lstm_hidden_size <= 0:
        raise ValueError("--lstm-hidden-size must be positive")
    if args.lstm_layers <= 0:
        raise ValueError("--lstm-layers must be positive")
    if not (0.0 <= args.lstm_dropout < 1.0):
        raise ValueError("--lstm-dropout must be >= 0 and < 1")
    if args.gru_hidden_size <= 0:
        raise ValueError("--gru-hidden-size must be positive")
    if args.gru_layers <= 0:
        raise ValueError("--gru-layers must be positive")
    if not (0.0 <= args.gru_dropout < 1.0):
        raise ValueError("--gru-dropout must be >= 0 and < 1")
    if args.transformer_d_model <= 0:
        raise ValueError("--transformer-d-model must be positive")
    if args.transformer_heads <= 0:
        raise ValueError("--transformer-heads must be positive")
    if args.transformer_d_model % args.transformer_heads != 0:
        raise ValueError("--transformer-d-model must be divisible by --transformer-heads")
    if args.transformer_layers <= 0:
        raise ValueError("--transformer-layers must be positive")
    if args.transformer_ff_dim <= 0:
        raise ValueError("--transformer-ff-dim must be positive")
    if not (0.0 <= args.transformer_dropout < 1.0):
        raise ValueError("--transformer-dropout must be >= 0 and < 1")

    candles = load_raw_candles(Path(args.raw_data))
    x_seq, y, meta, channel_names = build_sequence_dataset(
        candles,
        lookback=args.lookback,
        edge=args.edge,
        short_edge=args.short_edge,
        feature_set=args.sequence_feature_set,
    )
    target_col = "target_down" if args.target_direction == "down" else "target_up"
    y = meta[target_col].to_numpy(dtype=np.float32)

    n_rows = len(x_seq)
    if n_rows < 50:
        raise ValueError("Need at least 50 sequence rows to train and evaluate.")

    if args.train_on_all:
        split_idx = n_rows
        x_train_seq = x_seq
        x_eval_seq = x_seq
        y_train = y
        y_eval = y.astype(np.int64)
        eval_meta = meta.reset_index(drop=True)
        evaluation_label = "train_in_sample"
    else:
        split_idx = int(n_rows * args.split)
        split_idx = max(1, min(n_rows - 1, split_idx))
        x_train_seq = x_seq[:split_idx]
        x_eval_seq = x_seq[split_idx:]
        y_train = y[:split_idx]
        y_eval = y[split_idx:].astype(np.int64)
        eval_meta = meta.iloc[split_idx:].reset_index(drop=True)
        evaluation_label = "chronological_test"
    pos_weight = resolve_pos_weight(y_train, args.class_weight_mode, args.pos_weight)
    model_key = f"sequence_{args.model_type}"

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    training_device = "cpu"

    if args.model_type == "mlp":
        x = flatten_sequences(x_seq)
        x_train_flat = x if args.train_on_all else x[:split_idx]
        x_eval_flat = x if args.train_on_all else x[split_idx:]
        x_train_norm, x_eval_norm, mean, std = normalize_flat_train_test(x_train_flat, x_eval_flat)
        if args.backend == "torch":
            try:
                from torch_sequence_nn import train_torch_mlp
            except ImportError as exc:
                raise RuntimeError("NN_BACKEND=torch requires PyTorch installed in this virtualenv.") from exc

            weights, biases, loss_history, model_probs, training_device = train_torch_mlp(
                x_train=x_train_norm,
                y_train=y_train,
                x_eval=x_eval_norm,
                hidden_layers=hidden_layers,
                learning_rate=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                l2=args.l2,
                pos_weight=pos_weight,
                seed=args.seed,
                device_name=args.device,
            )
        else:
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
            model_probs = predict_proba(x_eval_norm, weights, biases)
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
            short_edge=args.short_edge,
            training_backend=args.backend,
            training_device=training_device,
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
    elif args.model_type == "cnn":
        x_train_norm, x_eval_norm, mean, std = normalize_sequence_train_test(x_train_seq, x_eval_seq)
        if args.backend == "torch":
            try:
                from torch_sequence_nn import train_torch_cnn
            except ImportError as exc:
                raise RuntimeError("NN_BACKEND=torch requires PyTorch installed in this virtualenv.") from exc

            (
                conv_weights,
                conv_biases,
                dense_weights,
                dense_biases,
                loss_history,
                model_probs,
                training_device,
            ) = train_torch_cnn(
                x_train=x_train_norm,
                y_train=y_train,
                x_eval=x_eval_norm,
                cnn_filters=cnn_filters,
                cnn_kernel_sizes=cnn_kernel_sizes,
                hidden_layers=hidden_layers,
                learning_rate=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                l2=args.l2,
                pos_weight=pos_weight,
                seed=args.seed,
                device_name=args.device,
            )
        else:
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
            model_probs = predict_cnn_proba(x_eval_norm, conv_weights, conv_biases, dense_weights, dense_biases)
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
            short_edge=args.short_edge,
            training_backend=args.backend,
            training_device=training_device,
        )
        input_metadata = {
            "lookback": int(args.lookback),
            "channels": channel_names,
            "input_shape": [int(args.lookback), int(len(channel_names))],
            "feature_set": args.sequence_feature_set,
            "normalization": "per-channel mean/std from train split only, preserving [lookback, channels] shape",
            "definition": "Each row receives the last lookback candles as sequence channels.",
        }
        architecture_metadata = {
            "cnn_filters": cnn_filters,
            "cnn_kernel_sizes": cnn_kernel_sizes,
            "hidden_layers": hidden_layers,
        }
    elif args.model_type == "lstm":
        x_train_norm, x_eval_norm, mean, std = normalize_sequence_train_test(x_train_seq, x_eval_seq)
        try:
            from torch_sequence_nn import train_torch_lstm
        except ImportError as exc:
            raise RuntimeError("NN_BACKEND=torch requires PyTorch installed in this virtualenv.") from exc

        state, loss_history, model_probs, training_device = train_torch_lstm(
            x_train=x_train_norm,
            y_train=y_train,
            x_eval=x_eval_norm,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_layers=args.lstm_layers,
            hidden_layers=hidden_layers,
            dropout=args.lstm_dropout,
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            l2=args.l2,
            pos_weight=pos_weight,
            seed=args.seed,
            device_name=args.device,
        )
        save_lstm_sequence_model(
            model_out,
            state=state,
            input_mean=mean,
            input_std=std,
            lookback=args.lookback,
            channel_names=channel_names,
            sequence_feature_set=args.sequence_feature_set,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_layers=args.lstm_layers,
            lstm_dropout=args.lstm_dropout,
            hidden_layers=hidden_layers,
            edge=args.edge,
            short_edge=args.short_edge,
            training_backend=args.backend,
            training_device=training_device,
        )
        input_metadata = {
            "lookback": int(args.lookback),
            "channels": channel_names,
            "input_shape": [int(args.lookback), int(len(channel_names))],
            "feature_set": args.sequence_feature_set,
            "normalization": "per-channel mean/std from train split only, preserving [lookback, channels] shape",
            "definition": "Each row receives the last lookback candles with OHLCV-relative and technical indicator channels.",
        }
        architecture_metadata = {
            "lstm_hidden_size": int(args.lstm_hidden_size),
            "lstm_layers": int(args.lstm_layers),
            "lstm_dropout": float(args.lstm_dropout),
            "hidden_layers": hidden_layers,
        }
    elif args.model_type == "gru":
        x_train_norm, x_eval_norm, mean, std = normalize_sequence_train_test(x_train_seq, x_eval_seq)
        try:
            from torch_sequence_nn import train_torch_gru
        except ImportError as exc:
            raise RuntimeError("NN_BACKEND=torch requires PyTorch installed in this virtualenv.") from exc

        state, loss_history, model_probs, training_device = train_torch_gru(
            x_train=x_train_norm,
            y_train=y_train,
            x_eval=x_eval_norm,
            gru_hidden_size=args.gru_hidden_size,
            gru_layers=args.gru_layers,
            hidden_layers=hidden_layers,
            dropout=args.gru_dropout,
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            l2=args.l2,
            pos_weight=pos_weight,
            seed=args.seed,
            device_name=args.device,
        )
        save_gru_sequence_model(
            model_out,
            state=state,
            input_mean=mean,
            input_std=std,
            lookback=args.lookback,
            channel_names=channel_names,
            sequence_feature_set=args.sequence_feature_set,
            gru_hidden_size=args.gru_hidden_size,
            gru_layers=args.gru_layers,
            gru_dropout=args.gru_dropout,
            hidden_layers=hidden_layers,
            edge=args.edge,
            short_edge=args.short_edge,
            training_backend=args.backend,
            training_device=training_device,
        )
        input_metadata = {
            "lookback": int(args.lookback),
            "channels": channel_names,
            "input_shape": [int(args.lookback), int(len(channel_names))],
            "feature_set": args.sequence_feature_set,
            "normalization": "per-channel mean/std from train split only, preserving [lookback, channels] shape",
            "definition": "Each row receives the last lookback candles with OHLCV-relative and technical indicator channels.",
        }
        architecture_metadata = {
            "gru_hidden_size": int(args.gru_hidden_size),
            "gru_layers": int(args.gru_layers),
            "gru_dropout": float(args.gru_dropout),
            "hidden_layers": hidden_layers,
        }
    else:
        x_train_norm, x_eval_norm, mean, std = normalize_sequence_train_test(x_train_seq, x_eval_seq)
        try:
            from torch_sequence_nn import train_torch_transformer
        except ImportError as exc:
            raise RuntimeError("NN_BACKEND=torch requires PyTorch installed in this virtualenv.") from exc

        state, loss_history, model_probs, training_device = train_torch_transformer(
            x_train=x_train_norm,
            y_train=y_train,
            x_eval=x_eval_norm,
            lookback=args.lookback,
            d_model=args.transformer_d_model,
            heads=args.transformer_heads,
            layers=args.transformer_layers,
            ff_dim=args.transformer_ff_dim,
            hidden_layers=hidden_layers,
            dropout=args.transformer_dropout,
            learning_rate=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            l2=args.l2,
            pos_weight=pos_weight,
            seed=args.seed,
            device_name=args.device,
        )
        save_transformer_sequence_model(
            model_out,
            state=state,
            input_mean=mean,
            input_std=std,
            lookback=args.lookback,
            channel_names=channel_names,
            sequence_feature_set=args.sequence_feature_set,
            transformer_d_model=args.transformer_d_model,
            transformer_heads=args.transformer_heads,
            transformer_layers=args.transformer_layers,
            transformer_ff_dim=args.transformer_ff_dim,
            transformer_dropout=args.transformer_dropout,
            hidden_layers=hidden_layers,
            edge=args.edge,
            short_edge=args.short_edge,
            training_backend=args.backend,
            training_device=training_device,
        )
        input_metadata = {
            "lookback": int(args.lookback),
            "channels": channel_names,
            "input_shape": [int(args.lookback), int(len(channel_names))],
            "feature_set": args.sequence_feature_set,
            "normalization": "per-channel mean/std from train split only, preserving [lookback, channels] shape",
            "definition": "Each row receives an ordered candle sequence with learned positional embeddings.",
        }
        architecture_metadata = {
            "transformer_d_model": int(args.transformer_d_model),
            "transformer_heads": int(args.transformer_heads),
            "transformer_layers": int(args.transformer_layers),
            "transformer_ff_dim": int(args.transformer_ff_dim),
            "transformer_dropout": float(args.transformer_dropout),
            "hidden_layers": hidden_layers,
        }

    model_preds = (model_probs >= args.decision_threshold).astype(int)

    baseline_preds = {
        "always_positive": np.ones_like(y_eval, dtype=int),
        "always_negative": np.zeros_like(y_eval, dtype=int),
        "prev_candle_direction": (eval_meta["return_1bar"].to_numpy(dtype=np.float64) > 0.0).astype(int),
        "ma_direction": (eval_meta["sma_spread"].to_numpy(dtype=np.float64) > 0.0).astype(int),
        model_key: model_preds,
    }

    comparison = {name: classification_metrics(y_eval, preds) for name, preds in baseline_preds.items()}
    sweep_rows = threshold_sweep_metrics(y_eval, model_probs, thresholds)
    best_row = pick_best_threshold(sweep_rows, args.optimize_metric)

    metrics = {
        "primary_objective": "predictive_validity_vs_baselines",
        "secondary_objective": "fee_aware_backtest_reality_check",
        "model_type": model_key,
        "dataset": {
            "raw_rows": int(len(candles)),
            "sequence_rows_total": int(n_rows),
            "rows_train": int(split_idx),
            "rows_test": int(0 if args.train_on_all else n_rows - split_idx),
            "rows_eval": int(len(y_eval)),
            "split_fraction": float(1.0 if args.train_on_all else args.split),
            "evaluation": evaluation_label,
        },
        "sequence_input": input_metadata,
        "target": {
            "direction": args.target_direction,
            "edge": float(args.edge),
            "short_edge": float(args.edge if args.short_edge is None else args.short_edge),
            "definition": (
                "target_up=1 when next-candle forward_return > edge"
                if args.target_direction == "up"
                else "target_down=1 when next-candle forward_return < -short_edge"
            ),
        },
        "class_balance": {
            "overall": class_balance(y.astype(np.int64)),
            "train": class_balance(y_train.astype(np.int64)),
            "test": class_balance(y_eval),
        },
        "model_training": {
            "model_type": model_key,
            **architecture_metadata,
            "backend": args.backend,
            "device": training_device,
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
    metric_label = "in-sample" if args.train_on_all else "test"
    print(f"{model_key} {metric_label} metrics at decision-threshold:")
    print(json.dumps(comparison[model_key], indent=2))
    print("Best threshold from sweep:")
    print(json.dumps(best_row, indent=2))


if __name__ == "__main__":
    main()
