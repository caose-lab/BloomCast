from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.pipeline.weekly_chlorophyll_map_pipeline import (
    aggregate_uniform_grid,
    build_weekly_chlorophyll_payload,
    list_scene_files,
    parse_scene_timestamp,
    run,
    select_latest_7_calendar_days,
)


KML_TEXT = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>Test Polygon</name>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              -66.001,18.0,0 -65.999,18.0,0 -65.999,18.002,0 -66.001,18.002,0 -66.001,18.0,0
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>
"""


class WeeklyChlorophyllMapPipelineTests(unittest.TestCase):
    def test_parse_scene_timestamp(self) -> None:
        self.assertEqual(
            parse_scene_timestamp(Path("20260416T142915.csv")),
            datetime(2026, 4, 16, 14, 29, 15),
        )
        self.assertIsNone(parse_scene_timestamp(Path("chl_daily.csv")))

    def test_select_latest_7_calendar_days_uses_calendar_boundaries(self) -> None:
        files = [
            Path("20260408T120000.csv"),
            Path("20260410T010000.csv"),
            Path("20260410T235959.csv"),
            Path("20260410T142915.csv"),
            Path("20260411T142915.csv"),
            Path("20260413T090000.csv"),
            Path("20260416T142915.csv"),
        ]

        selected, start_date, end_date, included_dates = select_latest_7_calendar_days(files)

        self.assertEqual(start_date, date(2026, 4, 10))
        self.assertEqual(end_date, date(2026, 4, 16))
        self.assertEqual(
            included_dates,
            [
                "2026-04-10",
                "2026-04-11",
                "2026-04-12",
                "2026-04-13",
                "2026-04-14",
                "2026-04-15",
                "2026-04-16",
            ],
        )
        self.assertEqual(
            [path.name for path in selected],
            [
                "20260410T010000.csv",
                "20260410T142915.csv",
                "20260410T235959.csv",
                "20260411T142915.csv",
                "20260413T090000.csv",
                "20260416T142915.csv",
            ],
        )

    def test_list_scene_files_skips_non_timestamp_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "20260416T142915.csv").write_text("latitude,longitude,CHL_NN\n", encoding="utf-8")
            (directory / "chl_daily.csv").write_text("date,CHLL_NN_TOTAL\n", encoding="utf-8")

            files = list_scene_files(directory)

        self.assertEqual([path.name for path in files], ["20260416T142915.csv"])

    def test_aggregate_uniform_grid_averages_by_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            polygon_path = Path(tmpdir) / "mask.kml"
            polygon_path.write_text(KML_TEXT, encoding="utf-8")
            observations = pd.DataFrame(
                [
                    {"latitude": 18.00004, "longitude": -66.00096, "chlorophyll": 1.0},
                    {"latitude": 18.00006, "longitude": -66.00095, "chlorophyll": 3.0},
                    {"latitude": 18.00120, "longitude": -66.00080, "chlorophyll": None},
                    {"latitude": 18.00124, "longitude": -66.00078, "chlorophyll": 5.0},
                ]
            )

            grid = aggregate_uniform_grid(observations, polygon_path, cell_size_deg=0.001)

        self.assertEqual(len(grid), 2)
        self.assertAlmostEqual(grid.iloc[0]["chlorophyll"], 2.0)
        self.assertAlmostEqual(grid.iloc[1]["chlorophyll"], 5.0)

    def test_build_payload_masks_points_outside_polygon(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            source_dir = base / "chl"
            source_dir.mkdir()
            polygon_path = base / "mask.kml"
            polygon_path.write_text(KML_TEXT, encoding="utf-8")

            frame1 = pd.DataFrame(
                [
                    {"latitude": 18.0002, "longitude": -66.0008, "CHL_NN": 0.0},
                    {"latitude": 18.0003, "longitude": -66.0007, "CHL_NN": 0.30103},
                    {"latitude": 18.0050, "longitude": -66.0050, "CHL_NN": 1.0},
                ]
            )
            frame2 = pd.DataFrame(
                [
                    {"latitude": 18.00025, "longitude": -66.00075, "CHL_NN": 0.47712125},
                    {"latitude": 18.00035, "longitude": -66.00065, "CHL_NN": None},
                ]
            )
            frame1.to_csv(source_dir / "20260415T120000.csv", index=False)
            frame2.to_csv(source_dir / "20260416T120000.csv", index=False)

            payload = build_weekly_chlorophyll_payload(
                source_dir=source_dir,
                polygon_path=polygon_path,
                cell_size_deg=0.001,
            )

        self.assertEqual(payload["start_timestamp"], "2026-04-10T00:00:00")
        self.assertEqual(payload["end_timestamp"], "2026-04-16T23:59:59")
        self.assertEqual(
            payload["included_dates"],
            [
                "2026-04-10",
                "2026-04-11",
                "2026-04-12",
                "2026-04-13",
                "2026-04-14",
                "2026-04-15",
                "2026-04-16",
            ],
        )
        self.assertEqual(payload["source_file_count"], 2)
        self.assertEqual(payload["cell_count"], 1)
        self.assertEqual(len(payload["cells"]), 1)
        self.assertAlmostEqual(payload["cells"][0]["chlorophyll"], 2.0)

    def test_run_writes_expected_json_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            source_dir = base / "chl"
            output_dir = base / "out"
            source_dir.mkdir()
            polygon_path = base / "mask.kml"
            polygon_path.write_text(KML_TEXT, encoding="utf-8")

            frame = pd.DataFrame(
                [
                    {"latitude": 18.0002, "longitude": -66.0008, "CHL_NN": 0.0},
                    {"latitude": 18.0003, "longitude": -66.0007, "CHL_NN": 0.30103},
                ]
            )
            frame.to_csv(source_dir / "20260416T120000.csv", index=False)

            output_path = run(
                source_dir=source_dir,
                polygon_path=polygon_path,
                output_dir=output_dir,
                cell_size_deg=0.001,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(output_path.name, "weekly_chlorophyll_map.json")
        self.assertIn("generated_at_utc", payload)
        self.assertIn("included_dates", payload)
        self.assertEqual(payload["cell_count"], 1)
        self.assertEqual(sorted(payload["cells"][0].keys()), ["chlorophyll", "lat", "lon"])


if __name__ == "__main__":
    unittest.main()
