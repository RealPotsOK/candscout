#!/usr/bin/env python3
"""Sequence model math and portable artifacts for MLP, CNN, GRU, LSTM, and Transformer."""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def parse_int_list(raw: str, label: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"Expected at least one {label}")
    if any(v <= 0 for v in values):
        raise ValueError(f"{label} must be positive")
    return values


def parse_hidden_layers(raw: str) -> list[int]:
    return parse_int_list(raw, "hidden layer size")


def init_mlp(input_dim: int, hidden_layers: list[int], rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    layer_sizes = [input_dim, *hidden_layers, 1]
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []

    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        scale = np.sqrt(2.0 / fan_in)
        weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32))
        biases.append(np.zeros(fan_out, dtype=np.float32))

    return weights, biases


def forward_mlp(
    x: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    keep_cache: bool = False,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    activation = x
    cache: list[tuple[np.ndarray, np.ndarray]] = []

    for layer_idx in range(len(weights) - 1):
        z = activation @ weights[layer_idx] + biases[layer_idx]
        if keep_cache:
            cache.append((activation, z))
        activation = np.maximum(z, 0.0)

    logits = activation @ weights[-1] + biases[-1]
    if keep_cache:
        cache.append((activation, logits))
    return logits.reshape(-1), cache


def predict_proba(x: np.ndarray, weights: list[np.ndarray], biases: list[np.ndarray], batch_size: int = 8192) -> np.ndarray:
    """MLP probability prediction. Kept for backward compatibility."""
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        logits, _ = forward_mlp(x[start:end], weights, biases, keep_cache=False)
        out[start:end] = sigmoid(logits)
    return out


def weighted_bce_loss(y: np.ndarray, probs: np.ndarray, sample_weight: np.ndarray, weights: list[np.ndarray], l2: float) -> float:
    eps = 1e-12
    denom = float(np.sum(sample_weight))
    bce = -np.sum(sample_weight * (y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))) / denom
    penalty = 0.5 * l2 * sum(float(np.sum(w.astype(np.float64) * w.astype(np.float64))) for w in weights)
    return float(bce + penalty)


def _adam_update(
    params: list[np.ndarray],
    grads: list[np.ndarray],
    m_values: list[np.ndarray],
    v_values: list[np.ndarray],
    *,
    learning_rate: float,
    beta1: float,
    beta2: float,
    adam_eps: float,
    step: int,
) -> None:
    for idx in range(len(params)):
        m_values[idx] = beta1 * m_values[idx] + (1.0 - beta1) * grads[idx]
        v_values[idx] = beta2 * v_values[idx] + (1.0 - beta2) * (grads[idx] * grads[idx])

        m_hat = m_values[idx] / (1.0 - beta1**step)
        v_hat = v_values[idx] / (1.0 - beta2**step)
        params[idx] -= learning_rate * m_hat / (np.sqrt(v_hat) + adam_eps)


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hidden_layers: list[int],
    learning_rate: float,
    epochs: int,
    batch_size: int,
    l2: float,
    pos_weight: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    if epochs <= 0:
        raise ValueError("--epochs must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("--lr must be positive")

    rng = np.random.default_rng(seed)
    weights, biases = init_mlp(x_train.shape[1], hidden_layers, rng)
    m_w = [np.zeros_like(w) for w in weights]
    v_w = [np.zeros_like(w) for w in weights]
    m_b = [np.zeros_like(b) for b in biases]
    v_b = [np.zeros_like(b) for b in biases]

    beta1 = 0.9
    beta2 = 0.999
    adam_eps = 1e-8
    step = 0
    history: list[float] = []
    indices = np.arange(len(x_train))

    for epoch in range(epochs):
        rng.shuffle(indices)
        batch_losses: list[float] = []

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx].astype(np.float32, copy=False)
            sample_weight = np.where(yb == 1.0, pos_weight, 1.0).astype(np.float32)
            denom = float(np.sum(sample_weight))

            logits, cache = forward_mlp(xb, weights, biases, keep_cache=True)
            probs = sigmoid(logits).astype(np.float32)
            batch_losses.append(weighted_bce_loss(yb, probs, sample_weight, weights, l2))

            delta = ((probs - yb) * sample_weight / denom).reshape(-1, 1).astype(np.float32)
            grad_w: list[np.ndarray] = [np.empty_like(w) for w in weights]
            grad_b: list[np.ndarray] = [np.empty_like(b) for b in biases]

            for layer_idx in reversed(range(len(weights))):
                activation_prev, z = cache[layer_idx]
                grad_w[layer_idx] = activation_prev.T @ delta + l2 * weights[layer_idx]
                grad_b[layer_idx] = np.sum(delta, axis=0)

                if layer_idx > 0:
                    delta = delta @ weights[layer_idx].T
                    _, prev_z = cache[layer_idx - 1]
                    delta = delta * (prev_z > 0.0)

            step += 1
            _adam_update(
                weights,
                grad_w,
                m_w,
                v_w,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                adam_eps=adam_eps,
                step=step,
            )
            _adam_update(
                biases,
                grad_b,
                m_b,
                v_b,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                adam_eps=adam_eps,
                step=step,
            )

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        history.append(epoch_loss)
        print(f"epoch={epoch + 1}/{epochs} loss={epoch_loss:.6f}")

    return weights, biases, history


def _same_padding(kernel_size: int) -> tuple[int, int]:
    total = kernel_size - 1
    left = total // 2
    right = total - left
    return left, right


def conv1d_same_forward(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    kernel_size = weight.shape[0]
    pad_left, pad_right = _same_padding(kernel_size)
    x_pad = np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, window_shape=kernel_size, axis=1)
    return np.einsum("btck,kco->bto", windows, weight, optimize=True) + bias


def conv1d_same_backward(
    grad_z: np.ndarray,
    x_prev: np.ndarray,
    weight: np.ndarray,
    l2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel_size = weight.shape[0]
    pad_left, pad_right = _same_padding(kernel_size)
    batch_size, time_steps, _channels = x_prev.shape
    x_pad = np.pad(x_prev, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, window_shape=kernel_size, axis=1)

    grad_weight = np.einsum("btck,bto->kco", windows, grad_z, optimize=True).astype(np.float32)
    grad_weight += l2 * weight
    grad_bias = np.sum(grad_z, axis=(0, 1)).astype(np.float32)

    grad_x_pad = np.zeros_like(x_pad, dtype=np.float32)
    for kernel_idx in range(kernel_size):
        grad_x_pad[:, kernel_idx : kernel_idx + time_steps, :] += np.einsum(
            "bto,co->btc",
            grad_z,
            weight[kernel_idx],
            optimize=True,
        )

    if pad_right == 0:
        grad_x = grad_x_pad[:, pad_left:, :]
    else:
        grad_x = grad_x_pad[:, pad_left:-pad_right, :]
    if grad_x.shape[0] != batch_size:
        raise RuntimeError("Conv backward produced invalid batch shape")
    return grad_x.astype(np.float32), grad_weight, grad_bias


def init_cnn(
    input_channels: int,
    cnn_filters: list[int],
    cnn_kernel_sizes: list[int],
    hidden_layers: list[int],
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    if len(cnn_filters) != len(cnn_kernel_sizes):
        raise ValueError("--cnn-filters and --cnn-kernel-sizes must have the same number of entries")

    conv_weights: list[np.ndarray] = []
    conv_biases: list[np.ndarray] = []
    in_channels = input_channels
    for out_channels, kernel_size in zip(cnn_filters, cnn_kernel_sizes):
        fan_in = kernel_size * in_channels
        scale = np.sqrt(2.0 / fan_in)
        conv_weights.append(rng.normal(0.0, scale, size=(kernel_size, in_channels, out_channels)).astype(np.float32))
        conv_biases.append(np.zeros(out_channels, dtype=np.float32))
        in_channels = out_channels

    pooled_dim = 2 * cnn_filters[-1]
    dense_weights, dense_biases = init_mlp(pooled_dim, hidden_layers, rng)
    return conv_weights, conv_biases, dense_weights, dense_biases


def forward_cnn(
    x: np.ndarray,
    conv_weights: list[np.ndarray],
    conv_biases: list[np.ndarray],
    dense_weights: list[np.ndarray],
    dense_biases: list[np.ndarray],
    keep_cache: bool = False,
) -> tuple[np.ndarray, dict]:
    activation = x
    conv_cache: list[tuple[np.ndarray, np.ndarray]] = []

    for weight, bias in zip(conv_weights, conv_biases):
        z = conv1d_same_forward(activation, weight, bias)
        if keep_cache:
            conv_cache.append((activation, z))
        activation = np.maximum(z, 0.0)

    max_indices = np.argmax(activation, axis=1)
    max_pool = np.max(activation, axis=1)
    avg_pool = np.mean(activation, axis=1)
    pooled = np.concatenate([max_pool, avg_pool], axis=1).astype(np.float32, copy=False)
    logits, dense_cache = forward_mlp(pooled, dense_weights, dense_biases, keep_cache=keep_cache)

    cache = {}
    if keep_cache:
        cache = {
            "conv": conv_cache,
            "last_activation": activation,
            "max_indices": max_indices,
            "dense": dense_cache,
        }
    return logits, cache


def predict_cnn_proba(
    x: np.ndarray,
    conv_weights: list[np.ndarray],
    conv_biases: list[np.ndarray],
    dense_weights: list[np.ndarray],
    dense_biases: list[np.ndarray],
    batch_size: int = 4096,
) -> np.ndarray:
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        logits, _ = forward_cnn(
            x[start:end],
            conv_weights,
            conv_biases,
            dense_weights,
            dense_biases,
            keep_cache=False,
        )
        out[start:end] = sigmoid(logits)
    return out


def _dense_backward(
    delta: np.ndarray,
    dense_cache: list[tuple[np.ndarray, np.ndarray]],
    dense_weights: list[np.ndarray],
    l2: float,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    grad_w: list[np.ndarray] = [np.empty_like(w) for w in dense_weights]
    grad_b: list[np.ndarray] = [np.empty(w.shape[1], dtype=np.float32) for w in dense_weights]
    d_activation = delta
    d_input: np.ndarray | None = None

    for layer_idx in reversed(range(len(dense_weights))):
        activation_prev, _z = dense_cache[layer_idx]
        grad_w[layer_idx] = activation_prev.T @ d_activation + l2 * dense_weights[layer_idx]
        grad_b[layer_idx] = np.sum(d_activation, axis=0).astype(np.float32)
        d_prev = d_activation @ dense_weights[layer_idx].T
        if layer_idx == 0:
            d_input = d_prev.astype(np.float32)
        else:
            _, prev_z = dense_cache[layer_idx - 1]
            d_activation = d_prev * (prev_z > 0.0)

    if d_input is None:
        raise RuntimeError("Dense backward did not produce input gradient")
    return d_input, grad_w, grad_b


def _pool_backward(d_pooled: np.ndarray, last_activation: np.ndarray, max_indices: np.ndarray) -> np.ndarray:
    batch_size, time_steps, filters = last_activation.shape
    d_max = d_pooled[:, :filters]
    d_avg = d_pooled[:, filters:]
    grad_activation = np.zeros_like(last_activation, dtype=np.float32)

    batch_idx = np.arange(batch_size)[:, None]
    filter_idx = np.arange(filters)[None, :]
    grad_activation[batch_idx, max_indices, filter_idx] += d_max
    grad_activation += d_avg[:, None, :] / float(time_steps)
    return grad_activation


def train_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cnn_filters: list[int],
    cnn_kernel_sizes: list[int],
    hidden_layers: list[int],
    learning_rate: float,
    epochs: int,
    batch_size: int,
    l2: float,
    pos_weight: float,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float]]:
    if epochs <= 0:
        raise ValueError("--epochs must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("--lr must be positive")
    if x_train.ndim != 3:
        raise ValueError("CNN training expects x_train shape [rows, lookback, channels]")

    rng = np.random.default_rng(seed)
    conv_weights, conv_biases, dense_weights, dense_biases = init_cnn(
        input_channels=x_train.shape[2],
        cnn_filters=cnn_filters,
        cnn_kernel_sizes=cnn_kernel_sizes,
        hidden_layers=hidden_layers,
        rng=rng,
    )

    m_conv_w = [np.zeros_like(w) for w in conv_weights]
    v_conv_w = [np.zeros_like(w) for w in conv_weights]
    m_conv_b = [np.zeros_like(b) for b in conv_biases]
    v_conv_b = [np.zeros_like(b) for b in conv_biases]
    m_dense_w = [np.zeros_like(w) for w in dense_weights]
    v_dense_w = [np.zeros_like(w) for w in dense_weights]
    m_dense_b = [np.zeros_like(b) for b in dense_biases]
    v_dense_b = [np.zeros_like(b) for b in dense_biases]

    beta1 = 0.9
    beta2 = 0.999
    adam_eps = 1e-8
    step = 0
    history: list[float] = []
    indices = np.arange(len(x_train))

    for epoch in range(epochs):
        rng.shuffle(indices)
        batch_losses: list[float] = []

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx].astype(np.float32, copy=False)
            sample_weight = np.where(yb == 1.0, pos_weight, 1.0).astype(np.float32)
            denom = float(np.sum(sample_weight))

            logits, cache = forward_cnn(
                xb,
                conv_weights,
                conv_biases,
                dense_weights,
                dense_biases,
                keep_cache=True,
            )
            probs = sigmoid(logits).astype(np.float32)
            batch_losses.append(weighted_bce_loss(yb, probs, sample_weight, [*conv_weights, *dense_weights], l2))

            delta = ((probs - yb) * sample_weight / denom).reshape(-1, 1).astype(np.float32)
            d_pooled, grad_dense_w, grad_dense_b = _dense_backward(delta, cache["dense"], dense_weights, l2)
            d_activation = _pool_backward(d_pooled, cache["last_activation"], cache["max_indices"])

            grad_conv_w: list[np.ndarray] = [np.empty_like(w) for w in conv_weights]
            grad_conv_b: list[np.ndarray] = [np.empty_like(b) for b in conv_biases]
            conv_cache = cache["conv"]
            for layer_idx in reversed(range(len(conv_weights))):
                x_prev, z = conv_cache[layer_idx]
                grad_z = d_activation * (z > 0.0)
                d_activation, grad_conv_w[layer_idx], grad_conv_b[layer_idx] = conv1d_same_backward(
                    grad_z.astype(np.float32, copy=False),
                    x_prev,
                    conv_weights[layer_idx],
                    l2,
                )

            step += 1
            _adam_update(
                conv_weights,
                grad_conv_w,
                m_conv_w,
                v_conv_w,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                adam_eps=adam_eps,
                step=step,
            )
            _adam_update(
                conv_biases,
                grad_conv_b,
                m_conv_b,
                v_conv_b,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                adam_eps=adam_eps,
                step=step,
            )
            _adam_update(
                dense_weights,
                grad_dense_w,
                m_dense_w,
                v_dense_w,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                adam_eps=adam_eps,
                step=step,
            )
            _adam_update(
                dense_biases,
                grad_dense_b,
                m_dense_b,
                v_dense_b,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                adam_eps=adam_eps,
                step=step,
            )

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        history.append(epoch_loss)
        print(f"epoch={epoch + 1}/{epochs} loss={epoch_loss:.6f}")

    return conv_weights, conv_biases, dense_weights, dense_biases, history


def save_sequence_model(
    path,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    *,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    lookback: int,
    channel_names: list[str],
    hidden_layers: list[int],
    edge: float,
    short_edge: float | None = None,
    training_backend: str = "numpy",
    training_device: str = "cpu",
) -> None:
    payload = {
        "model_type": np.array(["sequence_mlp"]),
        "training_backend": np.array([training_backend]),
        "training_device": np.array([training_device]),
        "lookback": np.array([lookback], dtype=np.int64),
        "channel_names": np.array(channel_names),
        "hidden_layers": np.array(hidden_layers, dtype=np.int64),
        "input_mean": input_mean.astype(np.float32),
        "input_std": input_std.astype(np.float32),
        "edge": np.array([edge], dtype=np.float64),
        "short_edge": np.array([edge if short_edge is None else short_edge], dtype=np.float64),
        "layer_count": np.array([len(weights)], dtype=np.int64),
    }
    for idx, (weight, bias) in enumerate(zip(weights, biases)):
        payload[f"W_{idx}"] = weight.astype(np.float32)
        payload[f"b_{idx}"] = bias.astype(np.float32)
    np.savez(path, **payload)


def save_cnn_sequence_model(
    path,
    *,
    conv_weights: list[np.ndarray],
    conv_biases: list[np.ndarray],
    dense_weights: list[np.ndarray],
    dense_biases: list[np.ndarray],
    input_mean: np.ndarray,
    input_std: np.ndarray,
    lookback: int,
    channel_names: list[str],
    cnn_filters: list[int],
    cnn_kernel_sizes: list[int],
    hidden_layers: list[int],
    edge: float,
    short_edge: float | None = None,
    training_backend: str = "numpy",
    training_device: str = "cpu",
) -> None:
    payload = {
        "model_type": np.array(["sequence_cnn"]),
        "training_backend": np.array([training_backend]),
        "training_device": np.array([training_device]),
        "lookback": np.array([lookback], dtype=np.int64),
        "channel_names": np.array(channel_names),
        "cnn_filters": np.array(cnn_filters, dtype=np.int64),
        "cnn_kernel_sizes": np.array(cnn_kernel_sizes, dtype=np.int64),
        "hidden_layers": np.array(hidden_layers, dtype=np.int64),
        "input_mean": input_mean.astype(np.float32),
        "input_std": input_std.astype(np.float32),
        "edge": np.array([edge], dtype=np.float64),
        "short_edge": np.array([edge if short_edge is None else short_edge], dtype=np.float64),
        "conv_layer_count": np.array([len(conv_weights)], dtype=np.int64),
        "dense_layer_count": np.array([len(dense_weights)], dtype=np.int64),
    }
    for idx, (weight, bias) in enumerate(zip(conv_weights, conv_biases)):
        payload[f"conv_W_{idx}"] = weight.astype(np.float32)
        payload[f"conv_b_{idx}"] = bias.astype(np.float32)
    for idx, (weight, bias) in enumerate(zip(dense_weights, dense_biases)):
        payload[f"dense_W_{idx}"] = weight.astype(np.float32)
        payload[f"dense_b_{idx}"] = bias.astype(np.float32)
    np.savez(path, **payload)


def save_lstm_sequence_model(
    path,
    *,
    state: dict[str, np.ndarray],
    input_mean: np.ndarray,
    input_std: np.ndarray,
    lookback: int,
    channel_names: list[str],
    sequence_feature_set: str,
    lstm_hidden_size: int,
    lstm_layers: int,
    lstm_dropout: float,
    hidden_layers: list[int],
    edge: float,
    short_edge: float | None = None,
    training_backend: str = "torch",
    training_device: str = "cpu",
) -> None:
    payload = {
        "model_type": np.array(["sequence_lstm"]),
        "training_backend": np.array([training_backend]),
        "training_device": np.array([training_device]),
        "lookback": np.array([lookback], dtype=np.int64),
        "channel_names": np.array(channel_names),
        "sequence_feature_set": np.array([sequence_feature_set]),
        "lstm_hidden_size": np.array([lstm_hidden_size], dtype=np.int64),
        "lstm_layers": np.array([lstm_layers], dtype=np.int64),
        "lstm_dropout": np.array([lstm_dropout], dtype=np.float64),
        "hidden_layers": np.array(hidden_layers, dtype=np.int64),
        "input_mean": input_mean.astype(np.float32),
        "input_std": input_std.astype(np.float32),
        "edge": np.array([edge], dtype=np.float64),
        "short_edge": np.array([edge if short_edge is None else short_edge], dtype=np.float64),
        "state_key_count": np.array([len(state)], dtype=np.int64),
    }
    for idx, (key, value) in enumerate(state.items()):
        payload[f"state_key_{idx}"] = np.array([key])
        payload[f"state_value_{idx}"] = value.astype(np.float32)
    np.savez(path, **payload)


def save_gru_sequence_model(
    path,
    *,
    state: dict[str, np.ndarray],
    input_mean: np.ndarray,
    input_std: np.ndarray,
    lookback: int,
    channel_names: list[str],
    sequence_feature_set: str,
    gru_hidden_size: int,
    gru_layers: int,
    gru_dropout: float,
    hidden_layers: list[int],
    edge: float,
    short_edge: float | None = None,
    training_backend: str = "torch",
    training_device: str = "cpu",
) -> None:
    payload = {
        "model_type": np.array(["sequence_gru"]),
        "training_backend": np.array([training_backend]),
        "training_device": np.array([training_device]),
        "lookback": np.array([lookback], dtype=np.int64),
        "channel_names": np.array(channel_names),
        "sequence_feature_set": np.array([sequence_feature_set]),
        "gru_hidden_size": np.array([gru_hidden_size], dtype=np.int64),
        "gru_layers": np.array([gru_layers], dtype=np.int64),
        "gru_dropout": np.array([gru_dropout], dtype=np.float64),
        "hidden_layers": np.array(hidden_layers, dtype=np.int64),
        "input_mean": input_mean.astype(np.float32),
        "input_std": input_std.astype(np.float32),
        "edge": np.array([edge], dtype=np.float64),
        "short_edge": np.array([edge if short_edge is None else short_edge], dtype=np.float64),
        "state_key_count": np.array([len(state)], dtype=np.int64),
    }
    for idx, (key, value) in enumerate(state.items()):
        payload[f"state_key_{idx}"] = np.array([key])
        payload[f"state_value_{idx}"] = value.astype(np.float32)
    np.savez(path, **payload)


def save_transformer_sequence_model(
    path,
    *,
    state: dict[str, np.ndarray],
    input_mean: np.ndarray,
    input_std: np.ndarray,
    lookback: int,
    channel_names: list[str],
    sequence_feature_set: str,
    transformer_d_model: int,
    transformer_heads: int,
    transformer_layers: int,
    transformer_ff_dim: int,
    transformer_dropout: float,
    hidden_layers: list[int],
    edge: float,
    short_edge: float | None = None,
    training_backend: str = "torch",
    training_device: str = "cpu",
) -> None:
    payload = {
        "model_type": np.array(["sequence_transformer"]),
        "training_backend": np.array([training_backend]),
        "training_device": np.array([training_device]),
        "lookback": np.array([lookback], dtype=np.int64),
        "channel_names": np.array(channel_names),
        "sequence_feature_set": np.array([sequence_feature_set]),
        "transformer_d_model": np.array([transformer_d_model], dtype=np.int64),
        "transformer_heads": np.array([transformer_heads], dtype=np.int64),
        "transformer_layers": np.array([transformer_layers], dtype=np.int64),
        "transformer_ff_dim": np.array([transformer_ff_dim], dtype=np.int64),
        "transformer_dropout": np.array([transformer_dropout], dtype=np.float64),
        "hidden_layers": np.array(hidden_layers, dtype=np.int64),
        "input_mean": input_mean.astype(np.float32),
        "input_std": input_std.astype(np.float32),
        "edge": np.array([edge], dtype=np.float64),
        "short_edge": np.array([edge if short_edge is None else short_edge], dtype=np.float64),
        "state_key_count": np.array([len(state)], dtype=np.int64),
    }
    for idx, (key, value) in enumerate(state.items()):
        payload[f"state_key_{idx}"] = np.array([key])
        payload[f"state_value_{idx}"] = value.astype(np.float32)
    np.savez(path, **payload)


def load_sequence_model(path) -> dict:
    model = np.load(path)
    model_type = str(model["model_type"][0]) if "model_type" in model.files else "sequence_mlp"
    base = {
        "model_type": model_type,
        "training_backend": str(model["training_backend"][0]) if "training_backend" in model.files else "unknown",
        "training_device": str(model["training_device"][0]) if "training_device" in model.files else "unknown",
        "lookback": int(model["lookback"][0]),
        "channel_names": [str(x) for x in model["channel_names"]],
        "hidden_layers": [int(x) for x in model["hidden_layers"]],
        "input_mean": model["input_mean"].astype(np.float32),
        "input_std": model["input_std"].astype(np.float32),
        "edge": float(model["edge"][0]),
        "short_edge": float(model["short_edge"][0]) if "short_edge" in model.files else float(model["edge"][0]),
    }

    if model_type == "sequence_mlp":
        layer_count = int(model["layer_count"][0])
        base.update(
            {
                "weights": [model[f"W_{idx}"].astype(np.float32) for idx in range(layer_count)],
                "biases": [model[f"b_{idx}"].astype(np.float32) for idx in range(layer_count)],
            }
        )
        return base

    if model_type == "sequence_cnn":
        conv_layer_count = int(model["conv_layer_count"][0])
        dense_layer_count = int(model["dense_layer_count"][0])
        base.update(
            {
                "cnn_filters": [int(x) for x in model["cnn_filters"]],
                "cnn_kernel_sizes": [int(x) for x in model["cnn_kernel_sizes"]],
                "conv_weights": [model[f"conv_W_{idx}"].astype(np.float32) for idx in range(conv_layer_count)],
                "conv_biases": [model[f"conv_b_{idx}"].astype(np.float32) for idx in range(conv_layer_count)],
                "dense_weights": [model[f"dense_W_{idx}"].astype(np.float32) for idx in range(dense_layer_count)],
                "dense_biases": [model[f"dense_b_{idx}"].astype(np.float32) for idx in range(dense_layer_count)],
            }
        )
        return base

    if model_type == "sequence_lstm":
        state_key_count = int(model["state_key_count"][0])
        base.update(
            {
                "sequence_feature_set": str(model["sequence_feature_set"][0])
                if "sequence_feature_set" in model.files
                else "technical",
                "lstm_hidden_size": int(model["lstm_hidden_size"][0]),
                "lstm_layers": int(model["lstm_layers"][0]),
                "lstm_dropout": float(model["lstm_dropout"][0]) if "lstm_dropout" in model.files else 0.0,
                "state": {
                    str(model[f"state_key_{idx}"][0]): model[f"state_value_{idx}"].astype(np.float32)
                    for idx in range(state_key_count)
                },
            }
        )
        return base

    if model_type == "sequence_gru":
        state_key_count = int(model["state_key_count"][0])
        base.update(
            {
                "sequence_feature_set": str(model["sequence_feature_set"][0])
                if "sequence_feature_set" in model.files
                else "technical",
                "gru_hidden_size": int(model["gru_hidden_size"][0]),
                "gru_layers": int(model["gru_layers"][0]),
                "gru_dropout": float(model["gru_dropout"][0]) if "gru_dropout" in model.files else 0.0,
                "state": {
                    str(model[f"state_key_{idx}"][0]): model[f"state_value_{idx}"].astype(np.float32)
                    for idx in range(state_key_count)
                },
            }
        )
        return base

    if model_type == "sequence_transformer":
        state_key_count = int(model["state_key_count"][0])
        base.update(
            {
                "sequence_feature_set": str(model["sequence_feature_set"][0])
                if "sequence_feature_set" in model.files
                else "technical",
                "transformer_d_model": int(model["transformer_d_model"][0]),
                "transformer_heads": int(model["transformer_heads"][0]),
                "transformer_layers": int(model["transformer_layers"][0]),
                "transformer_ff_dim": int(model["transformer_ff_dim"][0]),
                "transformer_dropout": float(model["transformer_dropout"][0]),
                "state": {
                    str(model[f"state_key_{idx}"][0]): model[f"state_value_{idx}"].astype(np.float32)
                    for idx in range(state_key_count)
                },
            }
        )
        return base

    raise ValueError(f"Unsupported sequence model type: {model_type}")


def predict_loaded_sequence_model(model: dict, x_seq: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    if model["model_type"] == "sequence_mlp":
        x = x_seq.reshape(x_seq.shape[0], -1).astype(np.float32, copy=False)
        x_norm = ((x - model["input_mean"]) / model["input_std"]).astype(np.float32, copy=False)
        return predict_proba(x_norm, model["weights"], model["biases"], batch_size=batch_size)

    if model["model_type"] == "sequence_cnn":
        mean = model["input_mean"].reshape(1, 1, -1)
        std = model["input_std"].reshape(1, 1, -1)
        x_norm = ((x_seq - mean) / std).astype(np.float32, copy=False)
        return predict_cnn_proba(
            x_norm,
            model["conv_weights"],
            model["conv_biases"],
            model["dense_weights"],
            model["dense_biases"],
            batch_size=batch_size,
        )

    if model["model_type"] == "sequence_lstm":
        mean = model["input_mean"].reshape(1, 1, -1)
        std = model["input_std"].reshape(1, 1, -1)
        x_norm = ((x_seq - mean) / std).astype(np.float32, copy=False)
        try:
            from torch_sequence_nn import predict_lstm_proba_from_state
        except ImportError as exc:
            raise RuntimeError("sequence_lstm prediction requires PyTorch installed in this virtualenv.") from exc
        return predict_lstm_proba_from_state(
            x_norm,
            state=model["state"],
            input_channels=len(model["channel_names"]),
            lstm_hidden_size=model["lstm_hidden_size"],
            lstm_layers=model["lstm_layers"],
            hidden_layers=model["hidden_layers"],
            dropout=model.get("lstm_dropout", 0.0),
            device_name="cuda",
            batch_size=batch_size,
        )

    if model["model_type"] == "sequence_gru":
        mean = model["input_mean"].reshape(1, 1, -1)
        std = model["input_std"].reshape(1, 1, -1)
        x_norm = ((x_seq - mean) / std).astype(np.float32, copy=False)
        try:
            from torch_sequence_nn import predict_gru_proba_from_state
        except ImportError as exc:
            raise RuntimeError("sequence_gru prediction requires PyTorch installed in this virtualenv.") from exc
        return predict_gru_proba_from_state(
            x_norm,
            state=model["state"],
            input_channels=len(model["channel_names"]),
            gru_hidden_size=model["gru_hidden_size"],
            gru_layers=model["gru_layers"],
            hidden_layers=model["hidden_layers"],
            dropout=model.get("gru_dropout", 0.0),
            device_name="cuda",
            batch_size=batch_size,
        )

    if model["model_type"] == "sequence_transformer":
        mean = model["input_mean"].reshape(1, 1, -1)
        std = model["input_std"].reshape(1, 1, -1)
        x_norm = ((x_seq - mean) / std).astype(np.float32, copy=False)
        try:
            from torch_sequence_nn import predict_transformer_proba_from_state
        except ImportError as exc:
            raise RuntimeError("sequence_transformer prediction requires PyTorch installed in this virtualenv.") from exc
        return predict_transformer_proba_from_state(
            x_norm,
            state=model["state"],
            input_channels=len(model["channel_names"]),
            lookback=model["lookback"],
            d_model=model["transformer_d_model"],
            heads=model["transformer_heads"],
            layers=model["transformer_layers"],
            ff_dim=model["transformer_ff_dim"],
            hidden_layers=model["hidden_layers"],
            dropout=model["transformer_dropout"],
            device_name="cuda",
            batch_size=batch_size,
        )

    raise ValueError(f"Unsupported sequence model type: {model['model_type']}")
