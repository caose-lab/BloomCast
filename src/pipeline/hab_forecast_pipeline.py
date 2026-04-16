#!/usr/bin/env python3
"""Run BloomCast operational HAB forecasting from a merged daily CSV."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Bootstrap imports when executed as `python src/pipeline/hab_forecast_pipeline.py`.
_here = Path(__file__).resolve()
for parent in [_here.parent] + list(_here.parents):
    if (parent / "src").exists() and (parent / "utils").exists():
        sys.path.insert(0, str(parent))
        sys.path.insert(0, str(parent / "src"))
        break

from bloomcast.hab_forecasting import predict_operational_package

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOGGER = logging.getLogger("hab_forecast_pipeline")
FORECAST_FILENAME = "operational_hab_forecast.json"
SUMMARY_FILENAME = "operational_summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to merged HAB input CSV.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where operational HAB forecast JSON will be written.",
    )
    return parser.parse_args(argv)


def _validate_input(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")


def _generated_at_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _transform_regression(regression: dict[str, Any], horizon: int) -> dict[str, Any]:
    prefix = f"week_{horizon}_ahead"
    return {
        "predicted_avg_mg_m3": regression[f"{prefix}_avg"],
        "lower_50_mg_m3": regression.get(f"{prefix}_lower_50"),
        "upper_50_mg_m3": regression.get(f"{prefix}_upper_50"),
        "lower_68_mg_m3": regression.get(f"{prefix}_lower_68"),
        "upper_68_mg_m3": regression.get(f"{prefix}_upper_68"),
        "lower_80_mg_m3": regression.get(f"{prefix}_lower_80"),
        "upper_80_mg_m3": regression.get(f"{prefix}_upper_80"),
        "lower_90_mg_m3": regression.get(f"{prefix}_lower_90"),
        "upper_90_mg_m3": regression.get(f"{prefix}_upper_90"),
    }


def _transform_risk_3class(risk: dict[str, Any]) -> dict[str, Any]:
    return {
        "predicted_risk": risk["predicted_risk"],
        "prob_low": risk["prob_low"],
        "prob_medium": risk["prob_medium"],
        "prob_high": risk["prob_high"],
        "low_upper_q25": risk["low_upper_q25"],
        "high_lower_quantile": risk["high_lower_quantile"],
        "high_quantile": risk["high_quantile"],
        "high_threshold_mode": risk["high_threshold_mode"],
    }


def _transform_high_risk_alert(high_risk: dict[str, Any]) -> dict[str, Any]:
    return {
        "predicted_high_risk": high_risk["predicted_high_risk"],
        "prob_high": high_risk["prob_high"],
        "probability_threshold": high_risk["probability_threshold"],
        "high_lower_q75": high_risk["high_lower_q75"],
        "high_quantile": high_risk["high_quantile"],
        "high_threshold_mode": high_risk["high_threshold_mode"],
    }


def _combine_operational_signals(
    regression: dict[str, Any],
    risk_3class: dict[str, Any],
    high_risk_alert: dict[str, Any],
) -> dict[str, Any]:
    high_threshold = float(risk_3class["high_lower_quantile"])
    point = float(regression["predicted_avg_mg_m3"])
    upper_50 = regression.get("upper_50_mg_m3")
    upper_68 = regression.get("upper_68_mg_m3")
    upper_80 = regression.get("upper_80_mg_m3")
    binary_prob = float(high_risk_alert["prob_high"])
    binary_threshold = float(high_risk_alert["probability_threshold"])
    binary_label = high_risk_alert["predicted_high_risk"]
    multiclass_label = risk_3class["predicted_risk"]
    multiclass_high_prob = float(risk_3class["prob_high"])

    exceeds_point = point >= high_threshold
    exceeds_upper_50 = upper_50 is not None and upper_50 >= high_threshold
    exceeds_upper_68 = upper_68 is not None and upper_68 >= high_threshold
    exceeds_upper_80 = upper_80 is not None and upper_80 >= high_threshold

    evidence_score = 0
    if binary_label == "high":
        evidence_score += 3
    elif binary_prob >= max(binary_threshold * 0.8, binary_threshold - 0.03):
        evidence_score += 1

    if multiclass_label == "high":
        evidence_score += 3
    elif multiclass_label == "medium":
        evidence_score += 1

    if exceeds_point:
        evidence_score += 3
    elif exceeds_upper_50:
        evidence_score += 1

    if exceeds_upper_68:
        evidence_score += 1
    if exceeds_upper_80:
        evidence_score += 1
    if multiclass_high_prob >= 0.30:
        evidence_score += 1

    if binary_label == "high" and (exceeds_point or multiclass_label == "high" or exceeds_upper_68):
        warning_level = "high"
    elif evidence_score >= 4:
        warning_level = "elevated"
    elif multiclass_label == "medium" or exceeds_upper_68:
        warning_level = "moderate"
    else:
        warning_level = "low"

    agreement_count = sum(
        [
            binary_label == "high",
            multiclass_label == "high",
            exceeds_point,
        ]
    )
    if warning_level == "low":
        confidence = "high" if agreement_count == 0 and not exceeds_upper_68 else "moderate"
    elif agreement_count >= 2:
        confidence = "high"
    elif binary_label == "high" or exceeds_upper_68 or multiclass_label == "medium":
        confidence = "moderate"
    else:
        confidence = "low"

    if warning_level == "high":
        if exceeds_point:
            summary = "Elevated conditions expected, and bloom risk is high."
        else:
            summary = "Moderate conditions expected, but elevated bloom risk remains likely."
    elif warning_level == "elevated":
        summary = "Moderate conditions expected, but elevated bloom risk remains possible."
    elif warning_level == "moderate":
        summary = "Moderate conditions expected, with some uncertainty about elevated bloom risk."
    else:
        summary = "Low to moderate conditions expected, and elevated bloom risk appears limited."

    return {
        "warning_level": warning_level,
        "confidence": confidence,
        "summary": summary,
        "bloom_threshold_mg_m3": high_threshold,
        "point_exceeds_bloom_threshold": exceeds_point,
        "upper_50_exceeds_bloom_threshold": exceeds_upper_50,
        "upper_68_exceeds_bloom_threshold": exceeds_upper_68,
        "upper_80_exceeds_bloom_threshold": exceeds_upper_80,
        "binary_alert_supports_high_risk": binary_label == "high",
        "binary_high_risk_probability": binary_prob,
        "three_class_supports_high_risk": multiclass_label == "high",
        "three_class_high_probability": multiclass_high_prob,
        "signal_agreement_count": agreement_count,
        "evidence_score": evidence_score,
    }


def build_operational_summary(
    forecast_package: dict[str, dict[str, Any]],
    generated_at_utc: str,
) -> dict[str, Any]:
    weeks: dict[str, dict[str, Any]] = {}

    for horizon in (1, 2, 3):
        source_week_key = f"week_{horizon}"
        summary_week_key = f"week{horizon}"
        source_payload = forecast_package[source_week_key]

        regression = _transform_regression(source_payload["regression"], horizon)
        risk_3class = _transform_risk_3class(source_payload["risk_3class"])
        high_risk_alert = _transform_high_risk_alert(source_payload["high_risk"])

        weeks[summary_week_key] = {
            "regression": regression,
            "risk_3class": risk_3class,
            "high_risk_alert": high_risk_alert,
            "operational_assessment": _combine_operational_signals(
                regression=regression,
                risk_3class=risk_3class,
                high_risk_alert=high_risk_alert,
            ),
        }

    return {
        "generated_at_utc": generated_at_utc,
        "weeks": weeks,
    }


def _write_json(output_path: Path, payload: dict[str, Any]) -> Path:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def run(input_path: Path, output_dir: Path) -> dict[str, Path]:
    _validate_input(input_path)
    LOGGER.info("Running HAB operational forecast for %s", input_path)

    forecast_package = predict_operational_package(csv_path=input_path)
    generated_at_utc = _generated_at_utc()
    forecast_payload = {
        "generated_at_utc": generated_at_utc,
        "input_csv": str(input_path),
        "forecast_package": forecast_package,
    }
    summary_payload = build_operational_summary(forecast_package, generated_at_utc=generated_at_utc)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "forecast": _write_json(output_dir / FORECAST_FILENAME, forecast_payload),
        "summary": _write_json(output_dir / SUMMARY_FILENAME, summary_payload),
    }

    LOGGER.info("Wrote HAB operational forecast to %s", output_paths["forecast"])
    LOGGER.info("Wrote HAB operational summary to %s", output_paths["summary"])
    return output_paths


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args(argv)
    try:
        run(Path(args.input), Path(args.output_dir))
    except Exception as exc:
        LOGGER.error("HAB operational forecast failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
