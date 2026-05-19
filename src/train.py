#!/usr/bin/env python3
"""Train baseline rules and a from-scratch NumPy logistic regression model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FEATURE_COLUMNS = [
    "return_1bar",
    "return_3bar",
    "return_5bar",
    "return_15bar",
    "return_30bar",
    "return_60bar",
    "volatility_5bar",
    "volatility_15bar",
    "volatility_30bar",
    "volatility_60bar",
    "sma_5_ratio",
    "sma_20_ratio",
    "sma_spread",
    "sma_50_ratio",
    "sma_100_ratio",
    "volume_change_1bar",
    "volume_zscore_20",
    "volume_sma_ratio_20",
    "volume_sma_ratio_60",
    "high_low_range",
    "close_open_range",
    "candle_body_pct",
    "candle_body_abs_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_position_in_range",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

LEGACY_FEATURE_COLUMNS = [
    "return_1m",
    "return_3m",
    "return_5m",
    "return_15m",
    "volatility_5m",
    "volatility_15m",
    "sma_5_ratio",
    "sma_20_ratio",
    "sma_spread",
    "volume_change_1m",
    "volume_zscore_20",
    "high_low_range",
    "close_open_range",
]


def parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def parse_feature_columns(raw: str) -> list[str]:
    columns = [x.strip() for x in raw.split(",") if x.strip()]
    if not columns:
        raise ValueError("--feature-columns was provided but no columns were parsed")
    return columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train rule baselines and NumPy logistic regression.")
    parser.add_argument("--features", required=True, help="Feature Parquet path")
    parser.add_argument("--feature-meta", default=None, help="Optional feature metadata JSON path")
    parser.add_argument(
        "--feature-columns",
        default=None,
        help="Optional comma-separated feature columns override",
    )
    parser.add_argument("--model-out", default="models/logreg_5m.npz", help="Model output path (.npz)")
    parser.add_argument("--metrics-out", default="models/train_metrics_5m.json", help="Training metrics JSON output")
    parser.add_argument("--split", type=float, default=0.8, help="Chronological train split fraction (default: 0.8)")
    parser.add_argument("--lr", type=float, default=0.05, help="Gradient descent learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="Gradient descent epochs")
    parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization strength")
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Probability threshold for y=1")
    parser.add_argument(
        "--threshold-grid",
        default="0.30,0.40,0.50,0.55,0.60,0.70",
        help="Comma-separated thresholds used for predictive-quality sweep reporting",
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
        help="Class weighting for logistic regression training",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Positive-class weight when class-weight-mode=manual",
    )
    return parser.parse_args()


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def resolve_feature_columns(df: pd.DataFrame, features_path: Path, feature_meta_arg: str | None, feature_cols_arg: str | None) -> tuple[list[str], str]:
    if feature_cols_arg:
        cols = parse_feature_columns(feature_cols_arg)
        return cols, "cli_override"

    candidate_meta: Path | None = None
    if feature_meta_arg:
        candidate_meta = Path(feature_meta_arg)
    else:
        if features_path.suffix:
            candidate_meta = features_path.with_suffix(".meta.json")

    if candidate_meta and candidate_meta.exists():
        payload = json.loads(candidate_meta.read_text())
        feature_cols = payload.get("feature_columns")
        if isinstance(feature_cols, list) and feature_cols:
            return [str(x) for x in feature_cols], str(candidate_meta)

    # Fallback for older feature files created before metadata support.
    fallback = [col for col in DEFAULT_FEATURE_COLUMNS if col in df.columns]
    if fallback:
        return fallback, "default_fallback"

    legacy_fallback = [col for col in LEGACY_FEATURE_COLUMNS if col in df.columns]
    if legacy_fallback:
        return legacy_fallback, "legacy_1m_fallback"

    raise ValueError(
        "Could not resolve feature columns. Provide --feature-columns or ensure feature metadata JSON exists."
    )


def resolve_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing required column for {purpose}. Tried: {candidates}")


def resolve_pos_weight(y_train: np.ndarray, mode: str, manual_weight: float | None) -> float:
    positives = float(np.sum(y_train == 1.0))
    negatives = float(np.sum(y_train == 0.0))

    if mode == "none":
        return 1.0

    if mode == "manual":
        if manual_weight is None or manual_weight <= 0.0:
            raise ValueError("--pos-weight must be > 0 when --class-weight-mode=manual")
        return float(manual_weight)

    # Balanced mode.
    if positives <= 0.0:
        return 1.0
    return float(negatives / positives)


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float,
    epochs: int,
    l2: float,
    pos_weight: float,
) -> tuple[np.ndarray, float, list[float]]:
    n_samples, n_features = x_train.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0
    loss_history: list[float] = []

    eps = 1e-12
    sample_weight = np.where(y_train == 1.0, pos_weight, 1.0).astype(np.float64)
    sample_weight /= np.mean(sample_weight)
    weight_denom = float(np.sum(sample_weight))

    for _ in range(epochs):
        logits = x_train @ weights + bias
        probs = sigmoid(logits)

        bce_terms = y_train * np.log(probs + eps) + (1.0 - y_train) * np.log(1.0 - probs + eps)
        bce = -np.sum(sample_weight * bce_terms) / weight_denom
        loss = bce + 0.5 * l2 * np.sum(weights * weights)
        loss_history.append(float(loss))

        error = sample_weight * (probs - y_train)
        grad_w = (x_train.T @ error) / weight_denom + l2 * weights
        grad_b = float(np.sum(error) / weight_denom)

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, bias, loss_history


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": float(accuracy),
        "precision_y1": float(precision),
        "recall_y1": float(recall),
        "f1_y1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def class_balance(y: np.ndarray) -> dict:
    if len(y) == 0:
        return {"positive_rate": 0.0, "negative_rate": 0.0}
    positive_rate = float(np.mean(y == 1))
    return {
        "positive_rate": positive_rate,
        "negative_rate": float(1.0 - positive_rate),
    }


def ensure_required_columns(df: pd.DataFrame, feature_columns: list[str]) -> str:
    required = set(feature_columns + ["target", "open_time", "sma_spread"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in feature data: {sorted(missing)}")
    return resolve_column(df, ["return_1bar", "return_1m"], "previous-candle direction baseline")


def threshold_sweep_metrics(y_true: np.ndarray, probs: np.ndarray, thresholds: list[float]) -> list[dict]:
    rows: list[dict] = []
    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(int)
        metrics = classification_metrics(y_true, y_pred)
        rows.append({"threshold": float(threshold), **metrics})
    return rows


def pick_best_threshold(rows: list[dict], metric_name: str) -> dict:
    # Tie-break order favors higher recall, then precision, then accuracy.
    return max(
        rows,
        key=lambda r: (
            r[metric_name],
            r["recall_y1"],
            r["precision_y1"],
            r["accuracy"],
        ),
    )


def main() -> None:
    args = parse_args()

    if not (0.0 < args.split < 1.0):
        raise ValueError("--split must be between 0 and 1")

    thresholds = parse_float_list(args.threshold_grid)
    if any(t <= 0.0 or t >= 1.0 for t in thresholds):
        raise ValueError("All threshold-grid values must be between 0 and 1")

    features_path = Path(args.features)
    df = pd.read_parquet(features_path)

    feature_columns, feature_source = resolve_feature_columns(
        df,
        features_path=features_path,
        feature_meta_arg=args.feature_meta,
        feature_cols_arg=args.feature_columns,
    )

    one_bar_return_col = ensure_required_columns(df, feature_columns)
    df = df.sort_values("open_time").reset_index(drop=True)

    n_rows = len(df)
    if n_rows < 50:
        raise ValueError("Need at least 50 rows to train and evaluate.")

    split_idx = int(n_rows * args.split)
    split_idx = max(1, min(n_rows - 1, split_idx))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    x_train = train_df[feature_columns].to_numpy(dtype=np.float64)
    y_train = train_df["target"].to_numpy(dtype=np.float64)
    x_test = test_df[feature_columns].to_numpy(dtype=np.float64)
    y_test = test_df["target"].to_numpy(dtype=np.int64)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0.0] = 1.0

    x_train_norm = (x_train - mean) / std
    x_test_norm = (x_test - mean) / std

    pos_weight = resolve_pos_weight(y_train, args.class_weight_mode, args.pos_weight)

    weights, bias, loss_history = train_logistic_regression(
        x_train=x_train_norm,
        y_train=y_train,
        learning_rate=args.lr,
        epochs=args.epochs,
        l2=args.l2,
        pos_weight=pos_weight,
    )

    model_probs = sigmoid(x_test_norm @ weights + bias)
    model_preds = (model_probs >= args.decision_threshold).astype(int)

    baseline_preds = {
        "always_positive": np.ones_like(y_test, dtype=int),
        "always_negative": np.zeros_like(y_test, dtype=int),
        "prev_candle_direction": (test_df[one_bar_return_col].to_numpy() > 0.0).astype(int),
        "ma_direction": (test_df["sma_spread"].to_numpy() > 0.0).astype(int),
        "logistic_regression": model_preds,
    }

    comparison = {}
    for name, preds in baseline_preds.items():
        comparison[name] = classification_metrics(y_test, preds)

    sweep_rows = threshold_sweep_metrics(y_test, model_probs, thresholds)
    best_row = pick_best_threshold(sweep_rows, args.optimize_metric)

    metrics = {
        "primary_objective": "predictive_validity_vs_baselines",
        "secondary_objective": "fee_aware_backtest_reality_check",
        "dataset": {
            "rows_total": int(n_rows),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "split_fraction": float(args.split),
        },
        "feature_columns": feature_columns,
        "feature_column_source": feature_source,
        "resolved_columns": {
            "one_bar_return": one_bar_return_col,
        },
        "class_balance": {
            "overall": class_balance(df["target"].to_numpy(dtype=np.int64)),
            "train": class_balance(train_df["target"].to_numpy(dtype=np.int64)),
            "test": class_balance(y_test),
        },
        "model_training": {
            "learning_rate": float(args.lr),
            "epochs": int(args.epochs),
            "l2": float(args.l2),
            "decision_threshold": float(args.decision_threshold),
            "class_weight_mode": args.class_weight_mode,
            "pos_weight_used": float(pos_weight),
            "initial_loss": float(loss_history[0]),
            "final_loss": float(loss_history[-1]),
        },
        "baseline_vs_model": comparison,
        "threshold_sweep": {
            "optimize_metric": args.optimize_metric,
            "rows": sweep_rows,
            "best": best_row,
        },
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        model_out,
        weights=weights,
        bias=np.array([bias], dtype=np.float64),
        feature_names=np.array(feature_columns),
        mean=mean,
        std=std,
    )

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model to {model_out}")
    print(f"Saved metrics to {metrics_out}")
    print("Model test metrics at decision-threshold:")
    print(json.dumps(comparison["logistic_regression"], indent=2))
    print("Best threshold from sweep:")
    print(json.dumps(best_row, indent=2))


if __name__ == "__main__":
    main()
