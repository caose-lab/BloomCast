from __future__ import annotations

import math
import unittest

import pandas as pd

from bloomcast.hab_forecasting import (
    predict_binary_high_risk,
    predict_operational_package,
    predict_three_class_risk,
    predict_weekly_regression,
)

CSV_PATH = "src/pipeline/final_data/SJL_daily_df.csv"
EXPECTED_DATE = (
    pd.read_csv(CSV_PATH, usecols=["date"], parse_dates=["date"])["date"].max() + pd.Timedelta(days=1)
).date().isoformat()


class HabForecastingSmokeTests(unittest.TestCase):
    def test_weekly_regression_smoke(self) -> None:
        for horizon in (1, 2, 3):
            result = predict_weekly_regression(CSV_PATH, horizon=horizon)
            self.assertEqual(result["prediction_date"], EXPECTED_DATE)
            self.assertIn(f"week_{horizon}_ahead_avg", result)
            self.assertTrue(math.isfinite(result[f"week_{horizon}_ahead_avg"]))

    def test_three_class_risk_smoke(self) -> None:
        for horizon in (1, 2, 3):
            result = predict_three_class_risk(CSV_PATH, horizon=horizon)
            self.assertEqual(result["prediction_date"], EXPECTED_DATE)
            self.assertEqual(result["horizon"], horizon)
            self.assertIn(result["predicted_risk"], {"low", "medium", "high"})
            probability_sum = result["prob_low"] + result["prob_medium"] + result["prob_high"]
            self.assertAlmostEqual(probability_sum, 1.0, places=6)

    def test_binary_high_risk_smoke(self) -> None:
        for horizon in (1, 2, 3):
            result = predict_binary_high_risk(CSV_PATH, horizon=horizon)
            self.assertEqual(result["prediction_date"], EXPECTED_DATE)
            self.assertEqual(result["horizon"], horizon)
            self.assertIn(result["predicted_high_risk"], {"high", "not_high"})
            self.assertGreaterEqual(result["prob_high"], 0.0)
            self.assertLessEqual(result["prob_high"], 1.0)

    def test_combined_package_smoke(self) -> None:
        package = predict_operational_package(CSV_PATH)
        self.assertEqual(sorted(package.keys()), ["week_1", "week_2", "week_3"])
        for week_key, payload in package.items():
            self.assertEqual(payload["regression"]["prediction_date"], EXPECTED_DATE)
            self.assertEqual(payload["risk_3class"]["prediction_date"], EXPECTED_DATE)
            self.assertEqual(payload["high_risk"]["prediction_date"], EXPECTED_DATE)
            self.assertIn("regression", payload, week_key)
            self.assertIn("risk_3class", payload, week_key)
            self.assertIn("high_risk", payload, week_key)


if __name__ == "__main__":
    unittest.main()
