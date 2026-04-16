from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.hab_forecast_pipeline import build_operational_summary, run


class HabForecastPipelineSmokeTests(unittest.TestCase):
    def test_run_writes_forecast_and_summary_json(self) -> None:
        input_path = Path("src/pipeline/final_data/SJL_daily_df.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_paths = run(input_path, Path(tmpdir))

            forecast_path = output_paths["forecast"]
            summary_path = output_paths["summary"]
            self.assertEqual(forecast_path.name, "operational_hab_forecast.json")
            self.assertEqual(summary_path.name, "operational_summary.json")

            forecast_payload = json.loads(forecast_path.read_text(encoding="utf-8"))
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(forecast_payload["input_csv"], str(input_path))
            self.assertIn("generated_at_utc", forecast_payload)
            self.assertNotIn("prediction_date", forecast_payload)
            self.assertEqual(sorted(forecast_payload["forecast_package"].keys()), ["week_1", "week_2", "week_3"])

            self.assertIn("generated_at_utc", summary_payload)
            self.assertNotIn("prediction_date", summary_payload)
            self.assertEqual(sorted(summary_payload.keys()), ["generated_at_utc", "weeks"])
            self.assertEqual(sorted(summary_payload["weeks"].keys()), ["week1", "week2", "week3"])

    def test_build_operational_summary_matches_reference_shape(self) -> None:
        forecast_package = {
            "week_1": {
                "regression": {
                    "prediction_date": "2026-04-13",
                    "week_1_ahead_avg": 10.0,
                    "week_1_ahead_lower_50": 8.0,
                    "week_1_ahead_upper_50": 12.0,
                    "week_1_ahead_lower_68": 7.0,
                    "week_1_ahead_upper_68": 13.0,
                    "week_1_ahead_lower_80": 6.0,
                    "week_1_ahead_upper_80": 14.0,
                    "week_1_ahead_lower_90": 5.0,
                    "week_1_ahead_upper_90": 15.0,
                },
                "risk_3class": {
                    "prediction_date": "2026-04-13",
                    "horizon": 1,
                    "predicted_risk": "medium",
                    "prob_low": 0.2,
                    "prob_medium": 0.7,
                    "prob_high": 0.1,
                    "low_upper_q25": 9.5,
                    "high_lower_quantile": 16.41,
                    "high_quantile": 0.75,
                    "high_threshold_mode": "fixed_value",
                },
                "high_risk": {
                    "prediction_date": "2026-04-13",
                    "horizon": 1,
                    "predicted_high_risk": "high",
                    "prob_high": 0.2,
                    "probability_threshold": 0.1,
                    "high_lower_q75": 16.41,
                    "high_quantile": 0.75,
                    "high_threshold_mode": "fixed_value",
                },
            },
            "week_2": {
                "regression": {
                    "prediction_date": "2026-04-13",
                    "week_2_ahead_avg": 12.0,
                    "week_2_ahead_lower_50": 10.0,
                    "week_2_ahead_upper_50": 14.0,
                    "week_2_ahead_lower_68": 9.0,
                    "week_2_ahead_upper_68": 17.0,
                    "week_2_ahead_lower_80": 8.0,
                    "week_2_ahead_upper_80": 18.0,
                    "week_2_ahead_lower_90": 7.0,
                    "week_2_ahead_upper_90": 19.0,
                },
                "risk_3class": {
                    "prediction_date": "2026-04-13",
                    "horizon": 2,
                    "predicted_risk": "high",
                    "prob_low": 0.1,
                    "prob_medium": 0.2,
                    "prob_high": 0.7,
                    "low_upper_q25": 9.5,
                    "high_lower_quantile": 16.41,
                    "high_quantile": 0.75,
                    "high_threshold_mode": "fixed_value",
                },
                "high_risk": {
                    "prediction_date": "2026-04-13",
                    "horizon": 2,
                    "predicted_high_risk": "high",
                    "prob_high": 0.2,
                    "probability_threshold": 0.125,
                    "high_lower_q75": 16.41,
                    "high_quantile": 0.75,
                    "high_threshold_mode": "fixed_value",
                },
            },
            "week_3": {
                "regression": {
                    "prediction_date": "2026-04-13",
                    "week_3_ahead_avg": 11.0,
                    "week_3_ahead_lower_50": 9.0,
                    "week_3_ahead_upper_50": 13.0,
                    "week_3_ahead_lower_68": 8.0,
                    "week_3_ahead_upper_68": 17.0,
                    "week_3_ahead_lower_80": 7.0,
                    "week_3_ahead_upper_80": 18.0,
                    "week_3_ahead_lower_90": 6.0,
                    "week_3_ahead_upper_90": 19.0,
                },
                "risk_3class": {
                    "prediction_date": "2026-04-13",
                    "horizon": 3,
                    "predicted_risk": "medium",
                    "prob_low": 0.2,
                    "prob_medium": 0.7,
                    "prob_high": 0.1,
                    "low_upper_q25": 9.5,
                    "high_lower_quantile": 16.41,
                    "high_quantile": 0.75,
                    "high_threshold_mode": "fixed_value",
                },
                "high_risk": {
                    "prediction_date": "2026-04-13",
                    "horizon": 3,
                    "predicted_high_risk": "not_high",
                    "prob_high": 0.03,
                    "probability_threshold": 0.1,
                    "high_lower_q75": 16.41,
                    "high_quantile": 0.75,
                    "high_threshold_mode": "fixed_value",
                },
            },
        }

        summary = build_operational_summary(
            forecast_package,
            generated_at_utc="2026-04-16T12:34:56",
        )

        self.assertEqual(summary["generated_at_utc"], "2026-04-16T12:34:56")
        self.assertEqual(
            summary["weeks"]["week1"]["regression"]["predicted_avg_mg_m3"],
            10.0,
        )
        self.assertEqual(
            summary["weeks"]["week1"]["high_risk_alert"]["predicted_high_risk"],
            "high",
        )
        self.assertEqual(
            summary["weeks"]["week2"]["operational_assessment"]["warning_level"],
            "high",
        )
        self.assertIn(
            "signal_agreement_count",
            summary["weeks"]["week3"]["operational_assessment"],
        )


if __name__ == "__main__":
    unittest.main()
