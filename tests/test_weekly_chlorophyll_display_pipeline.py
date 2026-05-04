from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.pipeline.chl_daily_coverage import _load_polygon_from_kml
from src.pipeline.weekly_chlorophyll_display_pipeline import (
    DEFAULT_DISPLAY_GRID_SIZE_DEG,
    OUTPUT_FILENAME,
    TARGET_SMOOTHING_METERS,
    _compute_sigma_pixels,
    build_display_surface_payload,
    run,
)


class WeeklyChlorophyllDisplayPipelineTests(unittest.TestCase):
    def test_build_payload_has_expected_metadata_and_cells(self) -> None:
        source_dir = Path("src/pipeline/data/chl_data")
        polygon_path = Path("src/pipeline/polygon/LSJmasking.kml")

        payload = build_display_surface_payload(
            source_dir=source_dir,
            polygon_path=polygon_path,
            generated_at_utc="2026-04-17T00:00:00+00:00",
        )

        self.assertEqual(payload["generated_at_utc"], "2026-04-17T00:00:00+00:00")
        self.assertEqual(payload["interpolation_method"], "idw")
        self.assertEqual(payload["smoothing_method"], "gaussian_filter")
        self.assertAlmostEqual(
            payload["smoothing_sigma_pixels"],
            _compute_sigma_pixels(DEFAULT_DISPLAY_GRID_SIZE_DEG, polygon_path),
        )
        self.assertEqual(payload["target_smoothing_meters"], TARGET_SMOOTHING_METERS)
        self.assertEqual(payload["grid_resolution_degrees"], DEFAULT_DISPLAY_GRID_SIZE_DEG)
        self.assertEqual(payload["coordinate_precision_decimal_places"], 5)
        self.assertEqual(payload["chlorophyll_precision_decimal_places"], 3)
        self.assertIn("surface_description", payload)
        self.assertTrue(payload["cells"])
        self.assertEqual(sorted(payload["cells"][0].keys()), ["chlorophyll", "lat", "lon"])

    def test_build_payload_is_deterministic_given_fixed_timestamp(self) -> None:
        source_dir = Path("src/pipeline/data/chl_data")
        polygon_path = Path("src/pipeline/polygon/LSJmasking.kml")

        payload1 = build_display_surface_payload(
            source_dir=source_dir,
            polygon_path=polygon_path,
            generated_at_utc="2026-04-17T00:00:00+00:00",
        )
        payload2 = build_display_surface_payload(
            source_dir=source_dir,
            polygon_path=polygon_path,
            generated_at_utc="2026-04-17T00:00:00+00:00",
        )

        self.assertEqual(payload1, payload2)

    def test_cells_are_non_negative_and_within_polygon_extent(self) -> None:
        source_dir = Path("src/pipeline/data/chl_data")
        polygon_path = Path("src/pipeline/polygon/LSJmasking.kml")
        payload = build_display_surface_payload(
            source_dir=source_dir,
            polygon_path=polygon_path,
            generated_at_utc="2026-04-17T00:00:00+00:00",
        )
        polygon = _load_polygon_from_kml(str(polygon_path))
        min_lon, min_lat, max_lon, max_lat = polygon.bounds

        for cell in payload["cells"]:
            self.assertGreaterEqual(cell["chlorophyll"], 0.0)
            self.assertGreaterEqual(cell["lat"], min_lat)
            self.assertLessEqual(cell["lat"], max_lat)
            self.assertGreaterEqual(cell["lon"], min_lon)
            self.assertLessEqual(cell["lon"], max_lon)
            self.assertEqual(cell["lat"], round(cell["lat"], 5))
            self.assertEqual(cell["lon"], round(cell["lon"], 5))
            self.assertEqual(cell["chlorophyll"], round(cell["chlorophyll"], 3))

    def test_run_writes_json_file(self) -> None:
        source_dir = Path("src/pipeline/data/chl_data")
        polygon_path = Path("src/pipeline/polygon/LSJmasking.kml")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = run(
                source_dir=source_dir,
                polygon_path=polygon_path,
                output_dir=Path(tmpdir),
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(output_path.name, OUTPUT_FILENAME)
        self.assertEqual(payload["interpolation_method"], "idw")
        self.assertAlmostEqual(
            payload["smoothing_sigma_pixels"],
            _compute_sigma_pixels(DEFAULT_DISPLAY_GRID_SIZE_DEG, polygon_path),
        )
        self.assertTrue(payload["cells"])


if __name__ == "__main__":
    unittest.main()
