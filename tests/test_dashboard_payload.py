from __future__ import annotations

import unittest

from dashboard_server import model_action_targets, sample_aligned_payload


class DashboardPayloadTest(unittest.TestCase):
    def test_simulate_rebuilds_predictions_before_running_simulation(self) -> None:
        model = {
            "actions": {
                "train": ["download", "experiment"],
                "simulate": ["nn-sim", "nn-sim-visualize", "reports-index"],
            }
        }

        self.assertEqual(
            model_action_targets(model, "simulate"),
            ["download", "experiment", "nn-sim", "nn-sim-visualize", "reports-index"],
        )
        self.assertEqual(
            model_action_targets(model, "quick_simulate"),
            ["nn-sim", "nn-sim-visualize", "reports-index"],
        )

    def test_nested_simulation_series_remain_distinct_and_aligned(self) -> None:
        payload = {
            "t": list(range(100)),
            "equity": {
                "model": [10_000 + index for index in range(100)],
                "model_long_short": [10_000 + index * 2 for index in range(100)],
            },
        }

        sampled = sample_aligned_payload(payload, "t", 10)

        self.assertEqual(len(sampled["t"]), len(sampled["equity"]["model"]))
        self.assertEqual(len(sampled["t"]), len(sampled["equity"]["model_long_short"]))
        self.assertNotEqual(
            sampled["equity"]["model"],
            sampled["equity"]["model_long_short"],
        )
        self.assertEqual(sampled["t"][-1], 99)


if __name__ == "__main__":
    unittest.main()
