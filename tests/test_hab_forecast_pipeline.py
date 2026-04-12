from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline.hab_forecast_pipeline import run


class HabForecastPipelineSmokeTests(unittest.TestCase):
    def test_run_writes_operational_package_json(self) -> None:
        input_path = Path("src/pipeline/final_data/SJL_daily_df.csv")
        expected_prediction_date = (
            pd.read_csv(input_path, usecols=["date"], parse_dates=["date"])["date"].max()
            + pd.Timedelta(days=1)
        ).date().isoformat()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = run(input_path, Path(tmpdir))

            self.assertEqual(output_path.name, "operational_hab_forecast.json")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["input_csv"], str(input_path))
            self.assertEqual(payload["prediction_date"], expected_prediction_date)
            self.assertEqual(sorted(payload["forecast_package"].keys()), ["week_1", "week_2", "week_3"])


if __name__ == "__main__":
    unittest.main()
