from __future__ import annotations

import unittest

import pandas as pd

from daily_bank_sim import default_test_window, short_investment_size, simulate_day


def prediction_frame(rows: list[tuple[str, float, float, str] | tuple[str, float, float, str, float]]) -> pd.DataFrame:
    prob_down = [row[4] if len(row) > 4 else 0.0 for row in rows]
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime([row[0] for row in rows], utc=True),
            "close": [row[1] for row in rows],
            "prob_up": [row[2] for row in rows],
            "prob_down": prob_down,
            "dataset_split": [row[3] for row in rows],
            "forward_return": [0.0] * len(rows),
        }
    )


class SimulationValidityTest(unittest.TestCase):
    def test_short_confidence_multiplier_increases_short_size_within_cap(self) -> None:
        base_size = short_investment_size(
            prob_down=0.47,
            threshold=0.45,
            min_invest=100.0,
            max_invest=2500.0,
            confidence_multiplier=1.0,
        )
        boosted_size = short_investment_size(
            prob_down=0.47,
            threshold=0.45,
            min_invest=100.0,
            max_invest=2500.0,
            confidence_multiplier=3.0,
        )
        self.assertGreater(boosted_size, base_size)
        self.assertLessEqual(boosted_size, 2500.0)

    def test_default_window_uses_explicit_test_rows(self) -> None:
        frame = prediction_frame(
            [
                ("2026-01-01T00:00:00Z", 100.0, 0.9, "train"),
                ("2026-01-01T00:05:00Z", 101.0, 0.9, "train"),
                ("2026-01-01T00:10:00Z", 102.0, 0.9, "test"),
                ("2026-01-01T00:15:00Z", 103.0, 0.9, "test"),
            ]
        )
        window, start, _end = default_test_window(frame, test_fraction=0.9)
        self.assertEqual(len(window), 2)
        self.assertEqual(start, pd.Timestamp("2026-01-01T00:10:00Z"))
        self.assertEqual(window.attrs["window_selection"], "explicit_dataset_split_test")

    def test_live_mode_does_not_exit_and_reenter_on_same_bar(self) -> None:
        frame = prediction_frame(
            [
                ("2026-01-01T00:00:00Z", 100.0, 0.9, "test"),
                ("2026-01-01T00:05:00Z", 102.0, 0.9, "test"),
                ("2026-01-01T00:10:00Z", 101.0, 0.1, "test"),
                ("2026-01-01T00:15:00Z", 101.0, 0.1, "test"),
            ]
        )
        report, trades = simulate_day(
            frame,
            start=pd.Timestamp("2026-01-01T00:00:00Z"),
            end=pd.Timestamp("2026-01-01T00:20:00Z"),
            starting_cash=100.0,
            min_invest=1.0,
            max_invest="m",
            threshold=0.55,
            fee=0.0,
            confidence_multiplier=1.0,
            position_mode="live",
            exit_threshold=0.5,
            max_hold_bars=60,
            stop_loss=0.0,
            take_profit=0.01,
            slippage=0.0,
            spread_pct=0.0,
        )
        self.assertEqual(report["trade_count"], 1)
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(report["total_profit"], trades["net_profit"].sum())
        self.assertAlmostEqual(report["accounting_error"], 0.0)

    def test_short_only_profits_when_price_falls(self) -> None:
        frame = prediction_frame(
            [
                ("2026-01-01T00:00:00Z", 100.0, 0.1, "test", 0.9),
                ("2026-01-01T00:05:00Z", 95.0, 0.1, "test", 0.9),
                ("2026-01-01T00:10:00Z", 90.0, 0.6, "test", 0.1),
            ]
        )
        report, trades = simulate_day(
            frame,
            start=pd.Timestamp("2026-01-01T00:00:00Z"),
            end=pd.Timestamp("2026-01-01T00:15:00Z"),
            starting_cash=100.0,
            min_invest=10.0,
            max_invest="m",
            threshold=0.55,
            fee=0.0,
            confidence_multiplier=1.0,
            position_mode="live",
            exit_threshold=0.48,
            max_hold_bars=60,
            stop_loss=0.0,
            take_profit=0.0,
            slippage=0.0,
            spread_pct=0.0,
            trade_mode="short_only",
            short_entry_threshold=0.45,
            short_exit_threshold=0.52,
            max_short_invest="m",
        )
        self.assertEqual(report["short_trade_count"], 1)
        self.assertEqual(trades.iloc[0]["side"], "short")
        self.assertGreater(report["total_profit"], 0.0)
        self.assertAlmostEqual(report["accounting_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
