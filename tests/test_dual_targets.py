from __future__ import annotations

import unittest

import pandas as pd

from daily_bank_sim import choose_entry_side
from features import build_features
from sequence_data import build_sequence_dataset


def candles() -> pd.DataFrame:
    close = [100.0, 101.0, 99.0, 100.0, 98.0, 99.0, 101.0, 100.0, 103.0, 101.0, 102.0, 100.0]
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2026-01-01", periods=len(close), freq="5min", tz="UTC"),
            "open": close,
            "high": [x * 1.002 for x in close],
            "low": [x * 0.998 for x in close],
            "close": close,
            "volume": [1000.0 + i for i in range(len(close))],
        }
    )


class DualTargetsTest(unittest.TestCase):
    def test_tabular_features_create_up_and_down_targets(self) -> None:
        df, _features, meta = build_features(
            candles(),
            edge=0.005,
            short_edge=0.01,
            return_windows=[1],
            vol_windows=[2],
            sma_short_window=2,
            sma_long_window=3,
            extra_sma_windows=[],
            volume_z_window=2,
            volume_ratio_windows=[],
            include_time_features=False,
        )
        expected_up = (df["forward_return"] > 0.005).astype(int)
        expected_down = (df["forward_return"] < -0.01).astype(int)
        self.assertTrue((df["target_up"] == expected_up).all())
        self.assertTrue((df["target_down"] == expected_down).all())
        self.assertTrue((df["target"] == df["target_up"]).all())
        self.assertEqual(meta["include_time_features"], False)

    def test_sequence_dataset_create_up_and_down_targets(self) -> None:
        _x, y, meta, _channels = build_sequence_dataset(
            candles(),
            lookback=3,
            edge=0.005,
            short_edge=0.01,
            feature_set="basic",
        )
        self.assertTrue((y == meta["target_up"].to_numpy()).all())
        self.assertTrue(((meta["forward_return"] > 0.005).astype(int) == meta["target_up"]).all())
        self.assertTrue(((meta["forward_return"] < -0.01).astype(int) == meta["target_down"]).all())
        self.assertTrue((meta["target"] == meta["target_up"]).all())

    def test_conflict_resolution_uses_stronger_normalized_confidence(self) -> None:
        self.assertEqual(
            choose_entry_side(
                prob_up=0.60,
                prob_down=0.80,
                trade_mode="long_short",
                long_threshold=0.55,
                short_threshold=0.45,
            ),
            "short",
        )
        self.assertEqual(
            choose_entry_side(
                prob_up=0.90,
                prob_down=0.55,
                trade_mode="long_short",
                long_threshold=0.55,
                short_threshold=0.45,
            ),
            "long",
        )

    def test_short_entry_uses_down_probability_not_inverse_up(self) -> None:
        self.assertEqual(
            choose_entry_side(
                prob_up=0.05,
                prob_down=0.40,
                trade_mode="long_short",
                long_threshold=0.55,
                short_threshold=0.55,
            ),
            "",
        )
        self.assertEqual(
            choose_entry_side(
                prob_up=0.05,
                prob_down=0.60,
                trade_mode="long_short",
                long_threshold=0.55,
                short_threshold=0.55,
            ),
            "short",
        )


if __name__ == "__main__":
    unittest.main()
