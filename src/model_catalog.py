"""Canonical model names and least-to-most complexity ordering."""

from __future__ import annotations

from typing import Any

MODEL_CATALOG = {
    "buy_hold": {"label": "Buy and Hold", "rank": 10, "group": "Baselines"},
    "prev_movement": {"label": "Previous Movement", "rank": 20, "group": "Baselines"},
    "ma": {"label": "MA Strategy", "rank": 30, "group": "Rule Strategies"},
    "logreg": {"label": "Logistic Regression", "rank": 40, "group": "Classical ML"},
    "xgboost": {"label": "XGBoost", "rank": 50, "group": "Classical ML"},
    "mlp": {"label": "MLP", "rank": 60, "group": "Neural Networks"},
    "cnn": {"label": "CNN", "rank": 70, "group": "Neural Networks"},
    "gru": {"label": "GRU", "rank": 80, "group": "Recurrent Neural Networks"},
    "lstm": {"label": "LSTM", "rank": 90, "group": "Recurrent Neural Networks"},
    "transformer": {"label": "Transformer", "rank": 100, "group": "Attention Models"},
}

MODEL_ALIASES = {
    "lr": "logreg",
    "logistic_regression": "logreg",
    "xgb": "xgboost",
    "previous_movement": "prev_movement",
    "prev_candle_direction": "prev_movement",
}


def normalize_model_type(model_type: str) -> str:
    value = str(model_type or "").strip().lower()
    if value.startswith("sequence_"):
        value = value.removeprefix("sequence_")
    if value.startswith("strategy_"):
        value = value.removeprefix("strategy_")
    return MODEL_ALIASES.get(value, value)


def model_catalog_entry(model_type: str) -> dict[str, Any]:
    normalized = normalize_model_type(model_type)
    return MODEL_CATALOG.get(
        normalized,
        {"label": normalized.replace("_", " ").title() or "Unknown Model", "rank": 999, "group": "Other"},
    )


def is_supported_model_type(model_type: str) -> bool:
    return normalize_model_type(model_type) in MODEL_CATALOG


def model_complexity_rank(model_type: str) -> int:
    return int(model_catalog_entry(model_type)["rank"])


def model_complexity_group(model_type: str) -> str:
    return str(model_catalog_entry(model_type)["group"])


def model_display_label(model_type: str) -> str:
    return str(model_catalog_entry(model_type)["label"])


def model_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    model_type = str(item.get("model_type") or item.get("id") or item.get("family") or "")
    rank = int(item.get("complexity_rank", model_complexity_rank(model_type)))
    label = str(item.get("label") or item.get("model_label") or model_display_label(model_type))
    return rank, label.lower()
