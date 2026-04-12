from __future__ import annotations

from pathlib import Path
from typing import Any

from chl_forecast.forecasting import (
    predict_from_bundle,
    predict_horizon_high_risk,
    predict_horizon_risk,
)
from utils.paths import REPO_ROOT

MODEL_ROOT = REPO_ROOT / "models" / "hab_operational" / "operational_models"

_REGRESSION_BUNDLES = {
    1: "week1/regression/chl_weekly_forecast_bundle.joblib",
    2: "week2/regression/chl_weekly_forecast_bundle.joblib",
    3: "week3/regression/chl_weekly_forecast_bundle.joblib",
}
_RISK_BUNDLES = {
    1: "week1/risk_3class/week1_risk_model_bundle.joblib",
    2: "week2/risk_3class/horizon_2_risk_model_bundle.joblib",
    3: "week3/risk_3class/horizon_3_risk_model_bundle.joblib",
}
_HIGH_RISK_BUNDLES = {
    1: "week1/high_risk/week1_high_risk_bundle.joblib",
    2: "week2/high_risk/horizon_2_high_risk_bundle.joblib",
    3: "week3/high_risk/horizon_3_high_risk_bundle.joblib",
}


def _bundle_path(bundle_map: dict[int, str], horizon: int, model_root: Path | None = None) -> Path:
    if horizon not in bundle_map:
        raise ValueError(f"Unsupported horizon: {horizon}. Expected one of 1, 2, 3.")
    root = model_root or MODEL_ROOT
    return root / bundle_map[horizon]


def _single_record(df) -> dict[str, Any]:
    return df.to_dict(orient="records")[0]


def predict_weekly_regression(
    csv_path: str | Path,
    horizon: int,
    prediction_date: str | None = None,
    model_root: Path | None = None,
) -> dict[str, Any]:
    return _single_record(
        predict_from_bundle(
            csv_path=csv_path,
            bundle_path=_bundle_path(_REGRESSION_BUNDLES, horizon, model_root=model_root),
            prediction_date=prediction_date,
        )
    )


def predict_three_class_risk(
    csv_path: str | Path,
    horizon: int,
    prediction_date: str | None = None,
    model_root: Path | None = None,
) -> dict[str, Any]:
    return _single_record(
        predict_horizon_risk(
            csv_path=csv_path,
            bundle_path=_bundle_path(_RISK_BUNDLES, horizon, model_root=model_root),
            prediction_date=prediction_date,
        )
    )


def predict_binary_high_risk(
    csv_path: str | Path,
    horizon: int,
    prediction_date: str | None = None,
    model_root: Path | None = None,
) -> dict[str, Any]:
    return _single_record(
        predict_horizon_high_risk(
            csv_path=csv_path,
            bundle_path=_bundle_path(_HIGH_RISK_BUNDLES, horizon, model_root=model_root),
            prediction_date=prediction_date,
        )
    )


def predict_operational_package(
    csv_path: str | Path,
    prediction_date: str | None = None,
    model_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        f"week_{horizon}": {
            "regression": predict_weekly_regression(
                csv_path=csv_path,
                horizon=horizon,
                prediction_date=prediction_date,
                model_root=model_root,
            ),
            "risk_3class": predict_three_class_risk(
                csv_path=csv_path,
                horizon=horizon,
                prediction_date=prediction_date,
                model_root=model_root,
            ),
            "high_risk": predict_binary_high_risk(
                csv_path=csv_path,
                horizon=horizon,
                prediction_date=prediction_date,
                model_root=model_root,
            ),
        }
        for horizon in (1, 2, 3)
    }
