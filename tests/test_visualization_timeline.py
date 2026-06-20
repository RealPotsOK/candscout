from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from visualize import load_predictions, outcome_summary


class VisualizationTimelineTest(unittest.TestCase):
    def test_model_visualization_keeps_train_and_test_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "open_time": pd.to_datetime(
                    [
                        "2026-01-01T00:00:00Z",
                        "2026-01-01T00:05:00Z",
                        "2026-01-01T00:10:00Z",
                        "2026-01-01T00:15:00Z",
                    ],
                    utc=True,
                ),
                "close": [100.0, 101.0, 102.0, 103.0],
                "target": [0, 1, 0, 1],
                "forward_return": [0.001, 0.002, -0.001, 0.0],
                "prob_up": [0.2, 0.8, 0.4, 0.7],
                "dataset_split": ["train", "train", "test", "test"],
            }
        )

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "predictions.parquet"
            frame.to_parquet(path, index=False)
            loaded = load_predictions(path, threshold=0.55, exit_threshold=0.45, fee=0.0)

        self.assertEqual(len(loaded), 4)
        self.assertEqual(loaded.attrs["split_counts"], {"train": 2, "test": 2})
        self.assertEqual(loaded.attrs["test_start_open_time"], "2026-01-01T00:10:00+00:00")

        equity_summary = {
            "fees_paid_cash": 0.0,
            "model_net_profit_cash": 0.0,
            "starting_cash": 100.0,
            "model_ending_cash": 100.0,
            "buy_hold_ending_cash": 100.0,
            "ma_baseline_ending_cash": 100.0,
            "buy_hold_net_profit_cash": 0.0,
            "ma_baseline_net_profit_cash": 0.0,
            "ma_baseline_window": 20,
        }
        summary = outcome_summary(loaded, fee=0.0, equity_summary=equity_summary)
        self.assertEqual(summary["prediction_rows"], 4)
        self.assertEqual(summary["train_prediction_rows"], 2)
        self.assertEqual(summary["test_prediction_rows"], 2)


if __name__ == "__main__":
    unittest.main()
