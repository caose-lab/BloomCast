from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline.latest_conditions_pipeline import build_latest_conditions_payload, run


class LatestConditionsPipelineTests(unittest.TestCase):
    def test_build_payload_uses_latest_value_per_field_independently(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2026-04-10",
                    "air_temperature": 26.1,
                    "water_temperature": None,
                    "AWND": 3.0,
                    "Watt_per_m2": 100.0,
                    "precipitation": None,
                    "CHLL_NN_TOTAL": 7.5,
                    "tidal_range": 0.31,
                },
                {
                    "date": "2026-04-11",
                    "air_temperature": None,
                    "water_temperature": 28.2,
                    "AWND": None,
                    "Watt_per_m2": None,
                    "precipitation": 4.5,
                    "CHLL_NN_TOTAL": None,
                    "tidal_range": 0.45,
                },
                {
                    "date": "2026-04-12",
                    "air_temperature": 27.0,
                    "water_temperature": None,
                    "AWND": 5.1,
                    "Watt_per_m2": None,
                    "precipitation": None,
                    "CHLL_NN_TOTAL": None,
                    "tidal_range": None,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.csv"
            frame.to_csv(input_path, index=False)
            payload = build_latest_conditions_payload(input_path)

        latest = payload["latest_conditions"]
        self.assertEqual(latest["air_temperature"]["as_of"], "2026-04-12")
        self.assertEqual(latest["water_temperature"]["as_of"], "2026-04-11")
        self.assertEqual(latest["wind_speed"]["as_of"], "2026-04-12")
        self.assertEqual(latest["radiation"]["as_of"], "2026-04-10")
        self.assertEqual(latest["precipitation"]["as_of"], "2026-04-11")
        self.assertEqual(latest["chlorophyll_a"]["as_of"], "2026-04-10")
        self.assertEqual(latest["tidal_range"]["as_of"], "2026-04-11")
        self.assertFalse(latest["air_temperature"]["missing"])

    def test_run_writes_json_and_marks_missing_fields(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2026-04-10",
                    "air_temperature": 26.1,
                    "water_temperature": 27.8,
                    "AWND": 3.0,
                    "Watt_per_m2": 100.0,
                    "precipitation": 1.2,
                    "CHLL_NN_TOTAL": None,
                    "tidal_range": 0.31,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.csv"
            output_dir = Path(tmpdir) / "out"
            frame.to_csv(input_path, index=False)

            output_path = run(input_path, output_dir)
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(output_path.name, "latest_conditions.json")
        self.assertIn("generated_at_utc", payload)
        self.assertTrue(payload["latest_conditions"]["chlorophyll_a"]["missing"])
        self.assertIsNone(payload["latest_conditions"]["chlorophyll_a"]["value"])
        self.assertIsNone(payload["latest_conditions"]["chlorophyll_a"]["as_of"])
        self.assertEqual(payload["latest_conditions"]["wind_speed"]["source_column"], "AWND")
        self.assertEqual(payload["latest_conditions"]["radiation"]["units"], "W/m^2")


if __name__ == "__main__":
    unittest.main()
