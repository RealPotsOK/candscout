#!/usr/bin/env python3
"""Torch training backend for sequence models, saving weights compatible with sequence_nn.py."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TorchSequenceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_layers:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TorchSequenceCNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        cnn_filters: list[int],
        cnn_kernel_sizes: list[int],
        hidden_layers: list[int],
    ) -> None:
        super().__init__()
        if len(cnn_filters) != len(cnn_kernel_sizes):
            raise ValueError("cnn_filters and cnn_kernel_sizes must have the same length")

        conv_layers: list[nn.Conv1d] = []
        in_channels = input_channels
        for out_channels, kernel_size in zip(cnn_filters, cnn_kernel_sizes):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same"))
            in_channels = out_channels
        self.conv_layers = nn.ModuleList(conv_layers)

        dense_layers: list[nn.Module] = []
        prev = 2 * cnn_filters[-1]
        for hidden in hidden_layers:
            dense_layers.append(nn.Linear(prev, hidden))
            dense_layers.append(nn.ReLU())
            prev = hidden
        dense_layers.append(nn.Linear(prev, 1))
        self.dense = nn.Sequential(*dense_layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Training data is [batch, lookback, channels]; Conv1d expects [batch, channels, lookback].
        activation = x.transpose(1, 2)
        for conv in self.conv_layers:
            activation = torch.relu(conv(activation))
        max_pool = torch.amax(activation, dim=2)
        avg_pool = torch.mean(activation, dim=2)
        pooled = torch.cat([max_pool, avg_pool], dim=1)
        return self.dense(pooled).squeeze(-1)


class TorchSequenceLSTM(nn.Module):
    def __init__(
        self,
        input_channels: int,
        lstm_hidden_size: int,
        lstm_layers: int,
        hidden_layers: list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if lstm_hidden_size <= 0:
            raise ValueError("lstm_hidden_size must be positive")
        if lstm_layers <= 0:
            raise ValueError("lstm_layers must be positive")
        lstm_dropout = float(dropout) if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )

        dense_layers: list[nn.Module] = []
        prev = lstm_hidden_size
        for hidden in hidden_layers:
            dense_layers.append(nn.Linear(prev, hidden))
            dense_layers.append(nn.ReLU())
            prev = hidden
        dense_layers.append(nn.Linear(prev, 1))
        self.dense = nn.Sequential(*dense_layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for module in self.dense:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _hidden = self.lstm(x)
        last_step = output[:, -1, :]
        return self.dense(last_step).squeeze(-1)


class TorchSequenceGRU(nn.Module):
    def __init__(
        self,
        input_channels: int,
        gru_hidden_size: int,
        gru_layers: int,
        hidden_layers: list[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if gru_hidden_size <= 0:
            raise ValueError("gru_hidden_size must be positive")
        if gru_layers <= 0:
            raise ValueError("gru_layers must be positive")
        gru_dropout = float(dropout) if gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            dropout=gru_dropout,
            batch_first=True,
        )

        dense_layers: list[nn.Module] = []
        prev = gru_hidden_size
        for hidden in hidden_layers:
            dense_layers.append(nn.Linear(prev, hidden))
            dense_layers.append(nn.ReLU())
            prev = hidden
        dense_layers.append(nn.Linear(prev, 1))
        self.dense = nn.Sequential(*dense_layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for module in self.dense:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _hidden = self.gru(x)
        last_step = output[:, -1, :]
        return self.dense(last_step).squeeze(-1)


class TorchSequenceTransformer(nn.Module):
    def __init__(
        self,
        input_channels: int,
        lookback: int,
        d_model: int,
        heads: int,
        layers: int,
        ff_dim: int,
        hidden_layers: list[int],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        if d_model <= 0 or heads <= 0 or layers <= 0 or ff_dim <= 0:
            raise ValueError("Transformer dimensions and layer counts must be positive")
        if d_model % heads != 0:
            raise ValueError("Transformer d_model must be divisible by heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Transformer dropout must be >= 0 and < 1")

        self.lookback = int(lookback)
        self.input_projection = nn.Linear(input_channels, d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, lookback, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        dense_layers: list[nn.Module] = []
        prev = 2 * d_model
        for hidden in hidden_layers:
            dense_layers.extend([nn.Linear(prev, hidden), nn.GELU(), nn.Dropout(dropout)])
            prev = hidden
        dense_layers.append(nn.Linear(prev, 1))
        self.dense = nn.Sequential(*dense_layers)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        for module in self.dense:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Transformer input must have shape [batch, lookback, channels]")
        time_steps = int(x.shape[1])
        if time_steps > self.lookback:
            raise ValueError(f"Transformer received {time_steps} steps but supports at most {self.lookback}")
        activation = self.input_projection(x) + self.positional_embedding[:, :time_steps, :]
        encoded = self.encoder(activation)
        pooled = torch.cat([encoded[:, -1, :], encoded.mean(dim=1)], dim=1)
        return self.dense(pooled).squeeze(-1)


@dataclass
class TorchTrainingResult:
    loss_history: list[float]
    probabilities: np.ndarray
    device: str


def resolve_torch_device(raw_device: str) -> torch.device:
    requested = (raw_device or "cuda").strip().lower()
    if requested == "auto":
        requested = "cuda"
    device = torch.device(requested)
    if device.type == "cpu":
        raise RuntimeError("CPU training is disabled for torch sequence models. Set a CUDA device such as 'cuda' or 'cuda:0'.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested CUDA device but torch.cuda.is_available() is false. "
            f"torch.version.cuda={torch.version.cuda!r}. Check the NVIDIA driver, Docker GPU access, "
            "and that this environment installed a CUDA-enabled PyTorch build."
        )
    return device


def set_torch_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def l2_penalty(model: nn.Module) -> torch.Tensor:
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if param.requires_grad and name.endswith("weight"):
            penalty = penalty + torch.sum(param * param)
    return penalty


def weighted_bce_with_logits(logits: torch.Tensor, y: torch.Tensor, pos_weight: float, l2: float, model: nn.Module) -> torch.Tensor:
    sample_weight = torch.where(y > 0.5, torch.full_like(y, float(pos_weight)), torch.ones_like(y))
    per_row = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
    denom = torch.clamp(sample_weight.sum(), min=1.0)
    loss = (per_row * sample_weight).sum() / denom
    if l2 > 0.0:
        loss = loss + 0.5 * float(l2) * l2_penalty(model)
    return loss


def predict_torch_proba(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device=device, dtype=torch.float32, non_blocking=True)
            logits = model(xb)
            batch_probs = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
            probs.append(batch_probs)
    return np.concatenate(probs) if probs else np.empty(0, dtype=np.float64)


def train_torch_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    l2: float,
    pos_weight: float,
    seed: int,
    device_name: str,
) -> TorchTrainingResult:
    if epochs <= 0:
        raise ValueError("--epochs must be positive")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("--lr must be positive")

    device = resolve_torch_device(device_name)
    set_torch_seed(seed, device)
    model.to(device)
    print(f"torch_device={device}")
    if device.type == "cuda":
        print(f"torch_cuda_name={torch.cuda.get_device_name(device)}")

    x_tensor = torch.from_numpy(x_train.astype(np.float32, copy=False))
    y_tensor = torch.from_numpy(y_train.astype(np.float32, copy=False))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[float] = []

    for epoch in range(epochs):
        model.train()
        weighted_loss_sum = 0.0
        row_count = 0
        for xb_cpu, yb_cpu in loader:
            xb = xb_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            yb = yb_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = weighted_bce_with_logits(logits, yb, pos_weight=pos_weight, l2=l2, model=model)
            loss.backward()
            optimizer.step()
            batch_rows = int(len(yb_cpu))
            weighted_loss_sum += float(loss.detach().cpu()) * batch_rows
            row_count += batch_rows

        epoch_loss = weighted_loss_sum / max(row_count, 1)
        history.append(float(epoch_loss))
        print(f"epoch={epoch + 1}/{epochs} loss={epoch_loss:.6f}", flush=True)

    probabilities = predict_torch_proba(model, x_eval.astype(np.float32, copy=False), device, batch_size=max(batch_size, 1024))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return TorchTrainingResult(loss_history=history, probabilities=probabilities, device=str(device))


def dense_arrays_from_torch_mlp(model: TorchSequenceMLP) -> tuple[list[np.ndarray], list[np.ndarray]]:
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for module in model.net:
        if isinstance(module, nn.Linear):
            weights.append(module.weight.detach().cpu().numpy().T.astype(np.float32))
            biases.append(module.bias.detach().cpu().numpy().astype(np.float32))
    return weights, biases


def arrays_from_torch_cnn(
    model: TorchSequenceCNN,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    conv_weights: list[np.ndarray] = []
    conv_biases: list[np.ndarray] = []
    for conv in model.conv_layers:
        # Torch Conv1d is [out_channels, in_channels, kernel]; NumPy model uses [kernel, in_channels, out_channels].
        conv_weights.append(np.transpose(conv.weight.detach().cpu().numpy(), (2, 1, 0)).astype(np.float32))
        conv_biases.append(conv.bias.detach().cpu().numpy().astype(np.float32))

    dense_weights: list[np.ndarray] = []
    dense_biases: list[np.ndarray] = []
    for module in model.dense:
        if isinstance(module, nn.Linear):
            dense_weights.append(module.weight.detach().cpu().numpy().T.astype(np.float32))
            dense_biases.append(module.bias.detach().cpu().numpy().astype(np.float32))
    return conv_weights, conv_biases, dense_weights, dense_biases


def train_torch_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    hidden_layers: list[int],
    learning_rate: float,
    epochs: int,
    batch_size: int,
    l2: float,
    pos_weight: float,
    seed: int,
    device_name: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], np.ndarray, str]:
    model = TorchSequenceMLP(input_dim=x_train.shape[1], hidden_layers=hidden_layers)
    result = train_torch_model(
        model,
        x_train,
        y_train,
        x_eval,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        l2=l2,
        pos_weight=pos_weight,
        seed=seed,
        device_name=device_name,
    )
    weights, biases = dense_arrays_from_torch_mlp(model)
    return weights, biases, result.loss_history, result.probabilities, result.device


def train_torch_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    cnn_filters: list[int],
    cnn_kernel_sizes: list[int],
    hidden_layers: list[int],
    learning_rate: float,
    epochs: int,
    batch_size: int,
    l2: float,
    pos_weight: float,
    seed: int,
    device_name: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float], np.ndarray, str]:
    model = TorchSequenceCNN(
        input_channels=x_train.shape[2],
        cnn_filters=cnn_filters,
        cnn_kernel_sizes=cnn_kernel_sizes,
        hidden_layers=hidden_layers,
    )
    result = train_torch_model(
        model,
        x_train,
        y_train,
        x_eval,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        l2=l2,
        pos_weight=pos_weight,
        seed=seed,
        device_name=device_name,
    )
    conv_weights, conv_biases, dense_weights, dense_biases = arrays_from_torch_cnn(model)
    return conv_weights, conv_biases, dense_weights, dense_biases, result.loss_history, result.probabilities, result.device


def train_torch_lstm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    lstm_hidden_size: int,
    lstm_layers: int,
    hidden_layers: list[int],
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    l2: float,
    pos_weight: float,
    seed: int,
    device_name: str,
) -> tuple[dict[str, np.ndarray], list[float], np.ndarray, str]:
    model = TorchSequenceLSTM(
        input_channels=x_train.shape[2],
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        hidden_layers=hidden_layers,
        dropout=dropout,
    )
    result = train_torch_model(
        model,
        x_train,
        y_train,
        x_eval,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        l2=l2,
        pos_weight=pos_weight,
        seed=seed,
        device_name=device_name,
    )
    state = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in model.state_dict().items()}
    return state, result.loss_history, result.probabilities, result.device


def train_torch_gru(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    gru_hidden_size: int,
    gru_layers: int,
    hidden_layers: list[int],
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    l2: float,
    pos_weight: float,
    seed: int,
    device_name: str,
) -> tuple[dict[str, np.ndarray], list[float], np.ndarray, str]:
    model = TorchSequenceGRU(
        input_channels=x_train.shape[2],
        gru_hidden_size=gru_hidden_size,
        gru_layers=gru_layers,
        hidden_layers=hidden_layers,
        dropout=dropout,
    )
    result = train_torch_model(
        model,
        x_train,
        y_train,
        x_eval,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        l2=l2,
        pos_weight=pos_weight,
        seed=seed,
        device_name=device_name,
    )
    state = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in model.state_dict().items()}
    return state, result.loss_history, result.probabilities, result.device


def train_torch_transformer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    lookback: int,
    d_model: int,
    heads: int,
    layers: int,
    ff_dim: int,
    hidden_layers: list[int],
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    l2: float,
    pos_weight: float,
    seed: int,
    device_name: str,
) -> tuple[dict[str, np.ndarray], list[float], np.ndarray, str]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = TorchSequenceTransformer(
        input_channels=x_train.shape[2],
        lookback=lookback,
        d_model=d_model,
        heads=heads,
        layers=layers,
        ff_dim=ff_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
    )
    result = train_torch_model(
        model,
        x_train,
        y_train,
        x_eval,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        l2=l2,
        pos_weight=pos_weight,
        seed=seed,
        device_name=device_name,
    )
    state = {key: value.detach().cpu().numpy().astype(np.float32) for key, value in model.state_dict().items()}
    return state, result.loss_history, result.probabilities, result.device


def predict_lstm_proba_from_state(
    x: np.ndarray,
    *,
    state: dict[str, np.ndarray],
    input_channels: int,
    lstm_hidden_size: int,
    lstm_layers: int,
    hidden_layers: list[int],
    dropout: float,
    device_name: str = "cuda",
    batch_size: int = 8192,
) -> np.ndarray:
    device = resolve_torch_device(device_name)
    model = TorchSequenceLSTM(
        input_channels=input_channels,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        hidden_layers=hidden_layers,
        dropout=dropout,
    )
    torch_state = {key: torch.from_numpy(value.astype(np.float32, copy=False)) for key, value in state.items()}
    model.load_state_dict(torch_state)
    model.to(device)
    return predict_torch_proba(model, x.astype(np.float32, copy=False), device, batch_size=max(batch_size, 1024))


def predict_gru_proba_from_state(
    x: np.ndarray,
    *,
    state: dict[str, np.ndarray],
    input_channels: int,
    gru_hidden_size: int,
    gru_layers: int,
    hidden_layers: list[int],
    dropout: float,
    device_name: str = "cuda",
    batch_size: int = 8192,
) -> np.ndarray:
    device = resolve_torch_device(device_name)
    model = TorchSequenceGRU(
        input_channels=input_channels,
        gru_hidden_size=gru_hidden_size,
        gru_layers=gru_layers,
        hidden_layers=hidden_layers,
        dropout=dropout,
    )
    torch_state = {key: torch.from_numpy(value.astype(np.float32, copy=False)) for key, value in state.items()}
    model.load_state_dict(torch_state)
    model.to(device)
    return predict_torch_proba(model, x.astype(np.float32, copy=False), device, batch_size=max(batch_size, 1024))


def predict_transformer_proba_from_state(
    x: np.ndarray,
    *,
    state: dict[str, np.ndarray],
    input_channels: int,
    lookback: int,
    d_model: int,
    heads: int,
    layers: int,
    ff_dim: int,
    hidden_layers: list[int],
    dropout: float,
    device_name: str = "cuda",
    batch_size: int = 8192,
) -> np.ndarray:
    device = resolve_torch_device(device_name)
    model = TorchSequenceTransformer(
        input_channels=input_channels,
        lookback=lookback,
        d_model=d_model,
        heads=heads,
        layers=layers,
        ff_dim=ff_dim,
        hidden_layers=hidden_layers,
        dropout=dropout,
    )
    torch_state = {key: torch.from_numpy(value.astype(np.float32, copy=False)) for key, value in state.items()}
    model.load_state_dict(torch_state)
    model.to(device)
    return predict_torch_proba(model, x.astype(np.float32, copy=False), device, batch_size=max(batch_size, 1024))
