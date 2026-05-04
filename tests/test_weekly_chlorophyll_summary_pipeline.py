from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.pipeline.weekly_chlorophyll_summary_pipeline import (
    OUTPUT_FILENAME,
    build_trailing_windows,
    build_weekly_summary_payload,
    compute_window_mean,
    run,
)


class WeeklyChlorophyllSummaryPipelineTests(unittest.TestCase):
    def test_build_trailing_windows_are_consecutive_and_non_overlapping(self) -> None:
        windows = build_trailing_windows(as_of=date(2026, 4, 17), window_days=7, window_count=4)

        self.assertEqual(
            windows,
            [
                ("week_1", date(2026, 4, 11), date(2026, 4, 17)),
                ("week_2", date(2026, 4, 4), date(2026, 4, 10)),
                ("week_3", date(2026, 3, 28), date(2026, 4, 3)),
                ("week_4", date(2026, 3, 21), date(2026, 3, 27)),
            ],
        )

    def test_build_payload_uses_latest_csv_date_by_default(self) -> None:
        rows = []
        current = date(2026, 3, 21)
        for offset in range(28):
            current_date = current + timedelta(days=offset)
            rows.append({"date": current_date.isoformat(), "CHLL_NN_TOTAL": 1.0})
        frame = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.csv"
            frame.to_csv(input_path, index=False)

            payload = build_weekly_summary_payload(
                input_path=input_path,
                run_date=date(2026, 4, 18),
                generated_at="2026-04-18T00:00:00Z",
            )

        self.assertEqual(payload["as_of_date"], "2026-04-17")
        self.assertEqual(payload["weeks"][0]["end_date"], "2026-04-17")

    def test_build_payload_computes_expected_weekly_means(self) -> None:
        rows = []
        current = date(2026, 3, 21)
        for offset in range(28):
            current_date = current + timedelta(days=offset)
            if current_date <= date(2026, 3, 27):
                value = 4.0
            elif current_date <= date(2026, 4, 3):
                value = 3.0
            elif current_date <= date(2026, 4, 10):
                value = 2.0
            else:
                value = 1.0
            rows.append({"date": current_date.isoformat(), "CHLL_NN_TOTAL": value})

        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.csv"
            frame.to_csv(input_path, index=False)

            payload = build_weekly_summary_payload(
                input_path=input_path,
                as_of_date="2026-04-17",
                run_date=date(2026, 4, 18),
                generated_at="2026-04-18T00:00:00Z",
                input_csv_label="src/pipeline/final_data/SJL_daily_df.csv",
            )

        self.assertEqual(
            payload["weeks"],
            [
                {
                    "label": "week_1",
                    "start_date": "2026-04-11",
                    "end_date": "2026-04-17",
                    "chlorophyll": 1.0,
                },
                {
                    "label": "week_2",
                    "start_date": "2026-04-04",
                    "end_date": "2026-04-10",
                    "chlorophyll": 2.0,
                },
                {
                    "label": "week_3",
                    "start_date": "2026-03-28",
                    "end_date": "2026-04-03",
                    "chlorophyll": 3.0,
                },
                {
                    "label": "week_4",
                    "start_date": "2026-03-21",
                    "end_date": "2026-03-27",
                    "chlorophyll": 4.0,
                },
            ],
        )

    def test_compute_window_mean_ignores_missing_values_when_min_valid_is_met(self) -> None:
        frame = pd.DataFrame(
            [
                {"date": "2026-04-11", "CHLL_NN_TOTAL": 2.0},
                {"date": "2026-04-12", "CHLL_NN_TOTAL": None},
                {"date": "2026-04-13", "CHLL_NN_TOTAL": 4.0},
            ]
        )
        frame["date"] = pd.to_datetime(frame["date"])

        result = compute_window_mean(
            df=frame,
            chl_col="CHLL_NN_TOTAL",
            start_date=date(2026, 4, 11),
            end_date=date(2026, 4, 13),
            min_valid=2,
        )

        self.assertEqual(result, 3.0)

    def test_build_payload_fails_when_window_has_too_few_valid_values(self) -> None:
        frame = pd.DataFrame(
            [
                {"date": "2026-04-11", "CHLL_NN_TOTAL": None},
                {"date": "2026-04-12", "CHLL_NN_TOTAL": None},
                {"date": "2026-04-13", "CHLL_NN_TOTAL": 4.0},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.csv"
            frame.to_csv(input_path, index=False)

            with self.assertRaisesRegex(
                ValueError,
                "Not enough valid chlorophyll values in window 2026-04-07..2026-04-13: have 1, require 2.",
            ):
                build_weekly_summary_payload(
                    input_path=input_path,
                    as_of_date="2026-04-13",
                    min_valid=2,
                    run_date=date(2026, 4, 18),
                    generated_at="2026-04-18T00:00:00Z",
                )

    def test_run_writes_compact_json_with_expected_shape(self) -> None:
        rows = []
        current = date(2026, 3, 21)
        for offset in range(28):
            current_date = current + timedelta(days=offset)
            rows.append({"date": current_date.isoformat(), "CHLL_NN_TOTAL": 1.5})

        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.csv"
            output_dir = Path(tmpdir) / "out"
            frame.to_csv(input_path, index=False)

            with patch("src.pipeline.weekly_chlorophyll_summary_pipeline.local_today", return_value=date(2026, 4, 18)):
                with patch(
                    "src.pipeline.weekly_chlorophyll_summary_pipeline.generated_at_utc",
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
            [
                "generated_at_utc",
                "run_date",
                "as_of_date",
                "input_csv",
                "chl_col",
                "window_days",
                "window_count",
                "weeks",
            ],
        )
        self.assertEqual(payload["generated_at_utc"], "2026-04-18T00:00:00Z")
        self.assertEqual(payload["run_date"], "2026-04-18")
        self.assertEqual(payload["input_csv"], str(input_path))
        self.assertEqual(payload["chl_col"], "CHLL_NN_TOTAL")
        self.assertEqual(payload["window_days"], 7)
        self.assertEqual(payload["window_count"], 4)
        self.assertEqual(len(payload["weeks"]), 4)
        self.assertEqual(
            sorted(payload["weeks"][0].keys()),
            ["chlorophyll", "end_date", "label", "start_date"],
        )


if __name__ == "__main__":
    unittest.main()
