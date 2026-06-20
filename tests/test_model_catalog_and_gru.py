from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch

from dashboard_server import load_registry
from model_catalog import model_complexity_rank
from sequence_nn import load_sequence_model, save_gru_sequence_model, save_transformer_sequence_model
from strategy_model.strategies import previous_movement_scores
from torch_sequence_nn import TorchSequenceGRU, TorchSequenceTransformer


class ModelCatalogTest(unittest.TestCase):
    def test_complexity_order_places_gru_before_lstm(self) -> None:
        ordered = [
            "buy_hold",
            "prev_movement",
            "ma",
            "logreg",
            "xgboost",
            "mlp",
            "cnn",
            "gru",
            "lstm",
            "transformer",
        ]
        ranks = [model_complexity_rank(model_type) for model_type in ordered]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual(len(ranks), len(set(ranks)))

    def test_all_dashboard_models_use_recommended_profile(self) -> None:
        models = load_registry()
        self.assertEqual(
            [model["id"] for model in models],
            ["buy_hold", "prev_movement", "ma", "lr", "xgboost", "mlp", "cnn", "gru", "lstm", "transformer"],
        )
        for model in models:
            defaults = model["defaults"]
            self.assertEqual(model.get("preset"), "recommended_v1")
            self.assertEqual(defaults["SPLIT"], "0.9")
            self.assertEqual(defaults["EDGE"], "0")
            self.assertEqual(defaults["THRESHOLD"], "0.55")
            self.assertEqual(defaults["EXIT_THRESHOLD"], "0.48")
            self.assertEqual(defaults["SIM_STARTING_CASH"], "10000")
            self.assertEqual(defaults["SIM_MIN_INVEST"], "2500")
            self.assertEqual(defaults["SIM_MAX_INVEST"], "8000")
            self.assertEqual(defaults["SIM_MAX_SHORT_INVEST"], "8000")

    def test_neural_recommendations_are_regularized_and_cuda_only(self) -> None:
        models = {model["id"]: model for model in load_registry()}
        expected_epochs = {"mlp": "100", "cnn": "100", "gru": "80", "lstm": "80", "transformer": "80"}
        for model_id, epochs in expected_epochs.items():
            defaults = models[model_id]["defaults"]
            self.assertEqual(defaults["NN_BACKEND"], "torch")
            self.assertEqual(defaults["NN_DEVICE"], "cuda")
            self.assertEqual(defaults["NN_EPOCHS"], epochs)
            self.assertEqual(defaults["NN_LR"], "0.0005")
            self.assertEqual(defaults["NN_L2"], "0.0005")
            self.assertEqual(defaults["NN_CLASS_WEIGHT_MODE"], "balanced")


class PreviousMovementTest(unittest.TestCase):
    def test_uses_latest_completed_candle_direction(self) -> None:
        frame = pd.DataFrame({"close": [100.0, 101.0, 100.0, 100.0, 102.0]})
        np.testing.assert_allclose(previous_movement_scores(frame), [0.5, 0.8, 0.2, 0.5, 0.8])


class GRUArtifactTest(unittest.TestCase):
    def test_artifact_round_trip(self) -> None:
        model = TorchSequenceGRU(
            input_channels=5,
            gru_hidden_size=8,
            gru_layers=1,
            hidden_layers=[4],
            dropout=0.0,
        )
        sample = np.zeros((3, 12, 5), dtype=np.float32)
        self.assertEqual(tuple(model.forward(torch.from_numpy(sample)).shape), (3,))

        state = {key: value.detach().numpy() for key, value in model.state_dict().items()}
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gru.npz"
            save_gru_sequence_model(
                path,
                state=state,
                input_mean=np.zeros(5, dtype=np.float32),
                input_std=np.ones(5, dtype=np.float32),
                lookback=12,
                channel_names=["open", "high", "low", "close", "volume"],
                sequence_feature_set="basic",
                gru_hidden_size=8,
                gru_layers=1,
                gru_dropout=0.0,
                hidden_layers=[4],
                edge=0.0,
                training_device="cuda",
            )

            loaded = load_sequence_model(path)
        self.assertEqual(loaded["model_type"], "sequence_gru")
        self.assertEqual(loaded["lookback"], 12)
        self.assertEqual(loaded["gru_hidden_size"], 8)
        self.assertEqual(loaded["gru_layers"], 1)
        self.assertEqual(loaded["hidden_layers"], [4])
        self.assertEqual(set(loaded["state"]), set(state))


class TransformerArtifactTest(unittest.TestCase):
    def test_artifact_round_trip(self) -> None:
        model = TorchSequenceTransformer(
            input_channels=5,
            lookback=12,
            d_model=16,
            heads=4,
            layers=1,
            ff_dim=32,
            hidden_layers=[8],
            dropout=0.1,
        )
        sample = np.zeros((3, 12, 5), dtype=np.float32)
        self.assertEqual(tuple(model.forward(torch.from_numpy(sample)).shape), (3,))

        state = {key: value.detach().numpy() for key, value in model.state_dict().items()}
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "transformer.npz"
            save_transformer_sequence_model(
                path,
                state=state,
                input_mean=np.zeros(5, dtype=np.float32),
                input_std=np.ones(5, dtype=np.float32),
                lookback=12,
                channel_names=["open", "high", "low", "close", "volume"],
                sequence_feature_set="basic",
                transformer_d_model=16,
                transformer_heads=4,
                transformer_layers=1,
                transformer_ff_dim=32,
                transformer_dropout=0.1,
                hidden_layers=[8],
                edge=0.0,
                training_device="cuda",
            )
            loaded = load_sequence_model(path)

        self.assertEqual(loaded["model_type"], "sequence_transformer")
        self.assertEqual(loaded["lookback"], 12)
        self.assertEqual(loaded["transformer_d_model"], 16)
        self.assertEqual(loaded["transformer_heads"], 4)
        self.assertEqual(loaded["transformer_layers"], 1)
        self.assertEqual(loaded["transformer_ff_dim"], 32)
        self.assertEqual(loaded["hidden_layers"], [8])
        self.assertEqual(set(loaded["state"]), set(state))


if __name__ == "__main__":
    unittest.main()
