from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.pipeline.environment_timeseries_pipeline import (
    FIELD_DECIMALS,
    OUTPUT_FILENAME,
    SERIES_FIELD_ORDER,
    WINDOW_DAYS,
    build_environment_timeseries_payload,
    run,
)
from src.pipeline.latest_conditions_pipeline import FIELD_SPECS


class EnvironmentTimeseriesPipelineTests(unittest.TestCase):
    def test_build_payload_uses_field_specs_units_and_frontend_field_order(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2026-04-17",
                    "CHLL_NN_TOTAL": 1.5,
                    "air_temperature": 26.0,
                    "precipitation": 2.0,
                    "Watt_per_m2": 150.0,
                    "tidal_range": 0.4,
                    "water_temperature": 28.0,
                    "AWND": 5.0,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.csv"
            frame.to_csv(input_path, index=False)
            payload = build_environment_timeseries_payload(
                input_path=input_path,
                generated_at="2026-04-18T00:00:00Z",
            )

        self.assertEqual(list(payload["fields"].keys()), list(SERIES_FIELD_ORDER))
        self.assertEqual(
            payload["fields"],
            {
                field_name: {"units": FIELD_SPECS[field_name]["units"]}
                for field_name in SERIES_FIELD_ORDER
            },
        )
        self.assertEqual(list(payload["series"][0].keys()), ["date", *SERIES_FIELD_ORDER])

    def test_build_payload_defaults_as_of_date_to_latest_valid_csv_date(self) -> None:
        frame = pd.DataFrame(
            [
                {"date": None, "CHLL_NN_TOTAL": 1.0, "air_temperature": 25.0, "precipitation": 0.0, "Watt_per_m2": 1.0, "tidal_range": 0.1, "water_temperature": 27.0, "AWND": 4.0},
                {"date": "2026-04-15", "CHLL_NN_TOTAL": 2.0, "air_temperature": 25.0, "precipitation": 0.0, "Watt_per_m2": 1.0, "tidal_range": 0.1, "water_temperature": 27.0, "AWND": 4.0},
                {"date": "2026-04-17", "CHLL_NN_TOTAL": 3.0, "air_temperature": 25.0, "precipitation": 0.0, "Watt_per_m2": 1.0, "tidal_range": 0.1, "water_temperature": 27.0, "AWND": 4.0},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.csv"
            frame.to_csv(input_path, index=False)
            payload = build_environment_timeseries_payload(
                input_path=input_path,
                generated_at="2026-04-18T00:00:00Z",
            )

        self.assertEqual(payload["end_date"], "2026-04-17")
        self.assertEqual(payload["start_date"], "2026-02-17")

    def test_build_payload_uses_inclusive_60_day_window_and_sorts_ascending(self) -> None:
        frame = pd.DataFrame(
            [
                {"date": "2026-04-17", "CHLL_NN_TOTAL": 60.0, "air_temperature": 26.0, "precipitation": 0.0, "Watt_per_m2": 160.0, "tidal_range": 0.6, "water_temperature": 28.0, "AWND": 6.0},
                {"date": "2026-02-16", "CHLL_NN_TOTAL": 1.0, "air_temperature": 20.0, "precipitation": 0.0, "Watt_per_m2": 100.0, "tidal_range": 0.1, "water_temperature": 24.0, "AWND": 1.0},
                {"date": "2026-02-17", "CHLL_NN_TOTAL": 2.0, "air_temperature": 21.0, "precipitation": 1.0, "Watt_per_m2": 101.0, "tidal_range": 0.2, "water_temperature": 25.0, "AWND": 2.0},
                {"date": "2026-03-01", "CHLL_NN_TOTAL": 3.0, "air_temperature": 22.0, "precipitation": 2.0, "Watt_per_m2": 102.0, "tidal_range": 0.3, "water_temperature": 26.0, "AWND": 3.0},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.csv"
            frame.to_csv(input_path, index=False)
            payload = build_environment_timeseries_payload(
                input_path=input_path,
                as_of_date="2026-04-17",
                generated_at="2026-04-18T00:00:00Z",
            )

        self.assertEqual(payload["start_date"], "2026-02-17")
        self.assertEqual(payload["end_date"], "2026-04-17")
        self.assertEqual([entry["date"] for entry in payload["series"]], ["2026-02-17", "2026-03-01", "2026-04-17"])

    def test_build_payload_preserves_nulls_and_excludes_all_null_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {"date": "2026-04-15", "CHLL_NN_TOTAL": None, "air_temperature": None, "precipitation": None, "Watt_per_m2": None, "tidal_range": None, "water_temperature": None, "AWND": None},
                {"date": "2026-04-16", "CHLL_NN_TOTAL": 5.0, "air_temperature": None, "precipitation": None, "Watt_per_m2": "bad", "tidal_range": None, "water_temperature": 28.5, "AWND": None},
                {"date": "2026-04-17", "CHLL_NN_TOTAL": None, "air_temperature": 26.1, "precipitation": None, "Watt_per_m2": 180.0, "tidal_range": None, "water_temperature": None, "AWND": 4.2},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.csv"
            frame.to_csv(input_path, index=False)
            payload = build_environment_timeseries_payload(
                input_path=input_path,
                as_of_date="2026-04-17",
                generated_at="2026-04-18T00:00:00Z",
            )

        self.assertEqual([entry["date"] for entry in payload["series"]], ["2026-04-16", "2026-04-17"])
        self.assertEqual(payload["series"][0]["chlorophyll_a"], 5.0)
        self.assertIsNone(payload["series"][0]["air_temperature"])
        self.assertIsNone(payload["series"][0]["radiation"])
        self.assertEqual(payload["series"][0]["water_temperature"], 28.5)
        self.assertIsInstance(payload["series"][0]["chlorophyll_a"], float)
        self.assertNotIsInstance(payload["series"][0]["chlorophyll_a"], str)

    def test_build_payload_aggregates_duplicate_dates_by_daily_mean(self) -> None:
        frame = pd.DataFrame(
            [
                {"date": "2026-04-17", "CHLL_NN_TOTAL": 4.0, "air_temperature": 25.0, "precipitation": 0.0, "Watt_per_m2": 100.0, "tidal_range": 0.2, "water_temperature": 27.0, "AWND": 3.0},
                {"date": "2026-04-17", "CHLL_NN_TOTAL": 8.0, "air_temperature": 27.0, "precipitation": 2.0, "Watt_per_m2": 200.0, "tidal_range": 0.4, "water_temperature": 29.0, "AWND": 5.0},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.csv"
            frame.to_csv(input_path, index=False)
            payload = build_environment_timeseries_payload(
                input_path=input_path,
                as_of_date="2026-04-17",
                generated_at="2026-04-18T00:00:00Z",
            )

        self.assertEqual(len(payload["series"]), 1)
        self.assertEqual(
            payload["series"][0],
            {
                "date": "2026-04-17",
                "chlorophyll_a": 6.0,
                "air_temperature": 26.0,
                "precipitation": 1.0,
                "radiation": 150.0,
                "tidal_range": 0.3,
                "water_temperature": 28.0,
                "wind_speed": 4.0,
            },
        )

    def test_build_payload_applies_expected_rounding_only_in_final_output(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "date": "2026-04-17",
                    "CHLL_NN_TOTAL": 1.234,
                    "air_temperature": 25.554,
                    "precipitation": 0.124,
                    "Watt_per_m2": 100.004,
                    "tidal_range": 0.1234,
                    "water_temperature": 27.994,
                    "AWND": 4.444,
                },
                {
                    "date": "2026-04-17",
                    "CHLL_NN_TOTAL": 1.235,
                    "air_temperature": 25.555,
                    "precipitation": 0.125,
                    "Watt_per_m2": 100.005,
                    "tidal_range": 0.1235,
                    "water_temperature": 27.995,
                    "AWND": 4.445,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.csv"
            frame.to_csv(input_path, index=False)
            payload = build_environment_timeseries_payload(
                input_path=input_path,
                as_of_date="2026-04-17",
                generated_at="2026-04-18T00:00:00Z",
            )

        point = payload["series"][0]
        self.assertEqual(point["chlorophyll_a"], 1.23)
        self.assertEqual(point["air_temperature"], 25.55)
        self.assertEqual(point["precipitation"], 0.12)
        self.assertEqual(point["radiation"], 100.0)
        self.assertEqual(point["tidal_range"], 0.123)
        self.assertEqual(point["water_temperature"], 27.99)
        self.assertEqual(point["wind_speed"], 4.44)
        for field_name in SERIES_FIELD_ORDER:
            self.assertFalse(isinstance(point[field_name], str))
            self.assertIsInstance(point[field_name], float)

    def test_run_writes_compact_json_with_expected_structure(self) -> None:
        frame = pd.DataFrame(
            [
                {"date": "2026-04-17", "CHLL_NN_TOTAL": 5.0, "air_temperature": 26.1, "precipitation": 1.2, "Watt_per_m2": 180.0, "tidal_range": 0.5, "water_temperature": 28.1, "AWND": 4.2}
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.csv"
            output_dir = Path(tmpdir) / "out"
            frame.to_csv(input_path, index=False)

            with patch(
                "src.pipeline.environment_timeseries_pipeline._generated_at_utc",
                return_value="2026-04-18T00:00:00Z",
            ):
                output_path = run(
                    input_path=input_path,
                    output_dir=output_dir,
                    as_of_date="2026-04-17",
                )

            raw = output_path.read_text(encoding="utf-8")
            payload = json.loads(raw)

        self.assertEqual(output_path.name, OUTPUT_FILENAME)
        self.assertTrue(raw.endswith("\n"))
        self.assertNotIn("\n  ", raw)
        self.assertEqual(
            list(payload.keys()),
            ["generated_at_utc", "input_csv", "start_date", "end_date", "fields", "series"],
        )
        self.assertEqual(payload["generated_at_utc"], "2026-04-18T00:00:00Z")
        self.assertEqual(payload["input_csv"], str(input_path))
        self.assertEqual(payload["start_date"], "2026-02-17")
        self.assertEqual(payload["end_date"], "2026-04-17")
        self.assertEqual(list(payload["fields"].keys()), list(SERIES_FIELD_ORDER))
        self.assertEqual(list(payload["series"][0].keys()), ["date", *SERIES_FIELD_ORDER])
        self.assertEqual(WINDOW_DAYS, 60)
        self.assertEqual(FIELD_DECIMALS["tidal_range"], 3)


if __name__ == "__main__":
    unittest.main()
