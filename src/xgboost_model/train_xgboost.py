#!/usr/bin/env python3
"""Train an XGBoost classifier on leakage-safe tabular candle features."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train import (  # noqa: E402
    class_balance,
    classification_metrics,
    ensure_dual_targets,
    ensure_required_columns,
    parse_float_list,
    pick_best_threshold,
    resolve_feature_columns,
    threshold_sweep_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost on engineered candle features.")
    parser.add_argument("--features", required=True, help="Feature Parquet path")
    parser.add_argument("--feature-meta", default=None, help="Optional feature metadata JSON path")
    parser.add_argument("--feature-columns", default=None, help="Optional comma-separated feature columns override")
    parser.add_argument("--model-out", default="models/xgb/model.json", help="XGBoost model output path")
    parser.add_argument("--metrics-out", default="models/xgb/train_metrics.json", help="Training metrics JSON output")
    parser.add_argument("--split", type=float, default=0.8, help="Chronological train split fraction")
    parser.add_argument("--short-edge", type=float, default=None, help="Short label edge when feature file lacks target_down")
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Probability threshold for y=1")
    parser.add_argument("--threshold-grid", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99")
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
        help="How to set XGBoost scale_pos_weight",
    )
    parser.add_argument("--pos-weight", type=float, default=None, help="Positive-class weight when class-weight-mode=manual")
    parser.add_argument("--n-estimators", type=int, default=300, help="Boosted tree count")
    parser.add_argument("--max-depth", type=int, default=4, help="Max tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="XGBoost eta / learning rate")
    parser.add_argument("--subsample", type=float, default=0.8, help="Row subsample ratio")
    parser.add_argument("--colsample-bytree", type=float, default=0.8, help="Column subsample ratio per tree")
    parser.add_argument("--min-child-weight", type=float, default=5.0, help="Minimum child weight")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="L2 regularization")
    parser.add_argument("--reg-alpha", type=float, default=0.0, help="L1 regularization")
    parser.add_argument("--gamma", type=float, default=0.0, help="Minimum split loss")
    parser.add_argument("--tree-method", default="hist", help="XGBoost tree_method, usually hist")
    parser.add_argument("--device", default="cuda", help="cuda or cuda:<id>; CPU is disabled by default")
    parser.add_argument("--n-jobs", type=int, default=0, help="Thread count; 0 lets XGBoost choose")
    parser.add_argument("--seed", type=int, default=67, help="Random seed")
    return parser.parse_args()


def import_xgboost():
    try:
        import xgboost as xgb  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: xgboost. Install it with `make install` or `./.venv/bin/pip install xgboost`."
        ) from exc
    return xgb


def resolve_device(raw: str) -> str:
    requested = raw.strip().lower()
    if requested == "auto":
        requested = "cuda"
    if requested == "cpu":
        raise RuntimeError("CPU training is disabled for XGBoost. Use XGB_DEVICE=cuda or XGB_DEVICE=cuda:<id>.")
    if requested.startswith("cuda") and not shutil.which("nvidia-smi"):
        raise RuntimeError("Requested CUDA for XGBoost, but nvidia-smi was not found in PATH.")
    return requested


def resolve_pos_weight(y_train: np.ndarray, mode: str, manual_weight: float | None) -> float:
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    if mode == "none":
        return 1.0
    if mode == "manual":
        if manual_weight is None or manual_weight <= 0.0:
            raise ValueError("--pos-weight must be > 0 when --class-weight-mode=manual")
        return float(manual_weight)
    return float(negatives / positives) if positives > 0.0 else 1.0


def validate_fraction(name: str, value: float) -> None:
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{name} must be > 0 and <= 1")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.split < 1.0):
        raise ValueError("--split must be between 0 and 1")
    validate_fraction("--subsample", args.subsample)
    validate_fraction("--colsample-bytree", args.colsample_bytree)

    thresholds = parse_float_list(args.threshold_grid)
    if any(t <= 0.0 or t >= 1.0 for t in thresholds):
        raise ValueError("All threshold-grid values must be between 0 and 1")

    xgb = import_xgboost()
    features_path = Path(args.features)
    df = pd.read_parquet(features_path)
    df = ensure_dual_targets(df, short_edge=args.short_edge)
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

    x_train = train_df[feature_columns]
    y_train = train_df["target_up"].to_numpy(dtype=np.int64)
    y_train_down = train_df["target_down"].to_numpy(dtype=np.int64)
    x_test = test_df[feature_columns]
    y_test = test_df["target_up"].to_numpy(dtype=np.int64)
    y_test_down = test_df["target_down"].to_numpy(dtype=np.int64)

    scale_pos_weight = resolve_pos_weight(y_train, args.class_weight_mode, args.pos_weight)
    device = resolve_device(args.device)
    n_jobs = None if args.n_jobs == 0 else args.n_jobs

    model_params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "max_depth": int(args.max_depth),
        "eta": float(args.learning_rate),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "min_child_weight": float(args.min_child_weight),
        "lambda": float(args.reg_lambda),
        "alpha": float(args.reg_alpha),
        "gamma": float(args.gamma),
        "scale_pos_weight": float(scale_pos_weight),
        "tree_method": args.tree_method,
        "device": device,
        "seed": int(args.seed),
    }
    if n_jobs is not None:
        model_params["nthread"] = n_jobs

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=feature_columns)
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=feature_columns)
    evals_result: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params=model_params,
        dtrain=dtrain,
        num_boost_round=int(args.n_estimators),
        evals=[(dtrain, "train"), (dtest, "test")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    scale_pos_weight_down = resolve_pos_weight(y_train_down, args.class_weight_mode, args.pos_weight)
    down_params = dict(model_params)
    down_params["scale_pos_weight"] = float(scale_pos_weight_down)
    dtrain_down = xgb.DMatrix(x_train, label=y_train_down, feature_names=feature_columns)
    dtest_down = xgb.DMatrix(x_test, label=y_test_down, feature_names=feature_columns)
    evals_result_down: dict[str, dict[str, list[float]]] = {}
    booster_down = xgb.train(
        params=down_params,
        dtrain=dtrain_down,
        num_boost_round=int(args.n_estimators),
        evals=[(dtrain_down, "train"), (dtest_down, "test")],
        evals_result=evals_result_down,
        verbose_eval=False,
    )

    model_probs = booster.predict(dtest).astype(np.float64)
    model_preds = (model_probs >= args.decision_threshold).astype(int)
    down_probs = booster_down.predict(dtest_down).astype(np.float64)
    down_preds = (down_probs >= args.decision_threshold).astype(int)

    baseline_preds = {
        "always_positive": np.ones_like(y_test, dtype=int),
        "always_negative": np.zeros_like(y_test, dtype=int),
        "prev_candle_direction": (test_df[one_bar_return_col].to_numpy() > 0.0).astype(int),
        "ma_direction": (test_df["sma_spread"].to_numpy() > 0.0).astype(int),
        "xgboost": model_preds,
    }
    comparison = {name: classification_metrics(y_test, preds) for name, preds in baseline_preds.items()}
    sweep_rows = threshold_sweep_metrics(y_test, model_probs, thresholds)
    best_row = pick_best_threshold(sweep_rows, args.optimize_metric)

    train_logloss = evals_result.get("train", {}).get("logloss", [])
    test_logloss = evals_result.get("test", {}).get("logloss", [])
    train_aucpr = evals_result.get("train", {}).get("aucpr", [])
    test_aucpr = evals_result.get("test", {}).get("aucpr", [])

    metrics = {
        "model_type": "xgboost",
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
        "resolved_columns": {"one_bar_return": one_bar_return_col},
        "class_balance": {
            "overall": class_balance(df["target_up"].to_numpy(dtype=np.int64)),
            "train": class_balance(y_train),
            "test": class_balance(y_test),
        },
        "dual_targets": {
            "target_up": {
                "class_balance": {
                    "overall": class_balance(df["target_up"].to_numpy(dtype=np.int64)),
                    "train": class_balance(y_train),
                    "test": class_balance(y_test),
                },
                "metrics": classification_metrics(y_test, model_preds),
                "threshold_sweep": {
                    "optimize_metric": args.optimize_metric,
                    "rows": sweep_rows,
                    "best": best_row,
                },
            },
            "target_down": {
                "class_balance": {
                    "overall": class_balance(df["target_down"].to_numpy(dtype=np.int64)),
                    "train": class_balance(y_train_down),
                    "test": class_balance(y_test_down),
                },
                "metrics": classification_metrics(y_test_down, down_preds),
                "threshold_sweep": {
                    "optimize_metric": args.optimize_metric,
                    "rows": threshold_sweep_metrics(y_test_down, down_probs, thresholds),
                    "best": pick_best_threshold(
                        threshold_sweep_metrics(y_test_down, down_probs, thresholds),
                        args.optimize_metric,
                    ),
                },
            },
        },
        "model_training": {
            "backend": "xgboost",
            "device": device,
            "tree_method": args.tree_method,
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "learning_rate": float(args.learning_rate),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample_bytree),
            "min_child_weight": float(args.min_child_weight),
            "reg_lambda": float(args.reg_lambda),
            "reg_alpha": float(args.reg_alpha),
            "gamma": float(args.gamma),
            "class_weight_mode": args.class_weight_mode,
            "scale_pos_weight_used": float(scale_pos_weight),
            "down_scale_pos_weight_used": float(scale_pos_weight_down),
            "decision_threshold": float(args.decision_threshold),
            "seed": int(args.seed),
            "initial_loss": float(train_logloss[0]) if train_logloss else None,
            "final_loss": float(train_logloss[-1]) if train_logloss else None,
            "final_test_loss": float(test_logloss[-1]) if test_logloss else None,
            "final_train_aucpr": float(train_aucpr[-1]) if train_aucpr else None,
            "final_test_aucpr": float(test_aucpr[-1]) if test_aucpr else None,
        },
        "baseline_vs_model": comparison,
        "threshold_sweep": {
            "optimize_metric": args.optimize_metric,
            "rows": sweep_rows,
            "best": best_row,
        },
    }

    model_out = Path(args.model_out)
    down_model_out = model_out.with_suffix(".down.json")
    model_out.parent.mkdir(parents=True, exist_ok=True)
    booster.set_attr(
        model_type="xgboost",
        feature_columns=json.dumps(feature_columns),
        feature_column_source=str(feature_source),
        split=str(args.split),
        decision_threshold=str(args.decision_threshold),
        direction="up",
    )
    booster.save_model(model_out)
    booster_down.set_attr(
        model_type="xgboost",
        feature_columns=json.dumps(feature_columns),
        feature_column_source=str(feature_source),
        split=str(args.split),
        decision_threshold=str(args.decision_threshold),
        direction="down",
    )
    booster_down.save_model(down_model_out)

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2))

    print(f"Saved XGBoost model to {model_out}")
    print(f"Saved XGBoost down model to {down_model_out}")
    print(f"Saved metrics to {metrics_out}")
    print("XGBoost test metrics at decision-threshold:")
    print(json.dumps(comparison["xgboost"], indent=2))
    print("Best threshold from sweep:")
    print(json.dumps(best_row, indent=2))


if __name__ == "__main__":
    main()
