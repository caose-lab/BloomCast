from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd

TARGET_COLUMN = "CHLL_NN_TOTAL"
DATE_COLUMN = "date"
QUALITY_COLUMN = "coverage_percent"
MIN_COVERAGE_PERCENT = 40.0
ROLLING_WINDOWS = (7, 14, 28)
LAG_STEPS = (1, 7, 14, 28)
EXCLUDED_PREDICTOR_COLUMNS = {
    DATE_COLUMN,
    TARGET_COLUMN,
    "temp_max",
    "temp_min",
    "wind_avg",
    "wind_speed_2m",
    "CHL_NN_R1",
    "CHL_NN_R2",
    "CHL_NN_R3",
    "CHL_OC4ME",
}
QUALITY_GATE_EXEMPT_COLUMNS = {
    DATE_COLUMN,
    QUALITY_COLUMN,
    "precipitation",
    "temp_max",
    "temp_min",
    "wind_avg",
    "wind_speed_2m",
    "air_pressure",
    "air_temperature",
    "water_level",
    "water_temperature",
    "Watt_per_m2",
    "AWND",
    "tidal_range",
}


def _safe_float_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    numeric_columns = [column for column in df.columns if column != DATE_COLUMN]
    df = _safe_float_columns(df, numeric_columns)
    return _apply_quality_gate(df)


def _apply_quality_gate(df: pd.DataFrame) -> pd.DataFrame:
    if QUALITY_COLUMN not in df.columns:
        return df

    gated = df.copy()
    low_coverage_mask = gated[QUALITY_COLUMN].lt(MIN_COVERAGE_PERCENT).fillna(False)
    gated_columns = [column for column in gated.columns if column not in QUALITY_GATE_EXEMPT_COLUMNS]
    gated.loc[low_coverage_mask, gated_columns] = np.nan
    return gated


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in EXCLUDED_PREDICTOR_COLUMNS]


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    day_of_year = out[DATE_COLUMN].dt.dayofyear
    out["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
    out["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)
    out["day_of_week"] = out[DATE_COLUMN].dt.dayofweek
    out["month"] = out[DATE_COLUMN].dt.month
    out["quarter"] = out[DATE_COLUMN].dt.quarter
    return out


def _days_since_last_observation(series: pd.Series) -> pd.Series:
    idx = np.arange(len(series))
    last_idx = np.where(series.notna(), idx, np.nan)
    last_idx = pd.Series(last_idx, index=series.index).ffill()
    return pd.Series(idx - last_idx, index=series.index, dtype=float)


def _extend_to_prediction_date(df: pd.DataFrame, prediction_date: pd.Timestamp) -> pd.DataFrame:
    max_date = df[DATE_COLUMN].max()
    if prediction_date <= max_date:
        return df

    future_dates = pd.date_range(start=max_date + pd.Timedelta(days=1), end=prediction_date, freq="D")
    future_frame = pd.DataFrame({DATE_COLUMN: future_dates})
    for column in df.columns:
        if column != DATE_COLUMN:
            future_frame[column] = np.nan
    return pd.concat([df, future_frame], ignore_index=True)


def _build_feature_frame(df: pd.DataFrame, base_features: list[str]) -> pd.DataFrame:
    out = _add_calendar_features(df[[DATE_COLUMN]].copy())
    series_for_features = base_features + [TARGET_COLUMN]
    engineered: dict[str, pd.Series] = {}

    for column in series_for_features:
        observed = df[column].shift(1)
        engineered[f"{column}_is_missing"] = observed.isna().astype(float)
        engineered[f"{column}_raw"] = observed
        for lag in LAG_STEPS:
            engineered[f"{column}_lag_{lag}"] = observed.shift(lag)
        for window in ROLLING_WINDOWS:
            roll = observed.rolling(window=window, min_periods=max(3, window // 2))
            engineered[f"{column}_roll_mean_{window}"] = roll.mean()
            engineered[f"{column}_roll_std_{window}"] = roll.std()
            engineered[f"{column}_roll_p95_{window}"] = roll.quantile(0.95)
            q75 = roll.quantile(0.75)
            q25 = roll.quantile(0.25)
            engineered[f"{column}_roll_iqr_{window}"] = q75 - q25
            engineered[f"{column}_valid_count_{window}"] = observed.notna().rolling(
                window=window, min_periods=1
            ).sum()

        if column == TARGET_COLUMN:
            engineered["target_days_since_last_obs"] = _days_since_last_observation(observed)
            engineered["target_ewm_mean_7"] = observed.ewm(
                halflife=7,
                min_periods=3,
                adjust=False,
            ).mean()
            engineered["target_ewm_mean_21"] = observed.ewm(
                halflife=21,
                min_periods=3,
                adjust=False,
            ).mean()
            engineered["target_delta_raw_lag_7"] = observed - observed.shift(7)
            engineered["target_delta_raw_lag_14"] = observed - observed.shift(14)
            engineered["target_delta_raw_lag_28"] = observed - observed.shift(28)

    engineered_frame = pd.DataFrame(engineered, index=df.index)
    target_observed = df[TARGET_COLUMN].shift(1)
    if "water_temperature" in df.columns:
        engineered_frame["target_level_x_water_temperature"] = (
            target_observed * df["water_temperature"].shift(1)
        )
    if "precipitation" in df.columns:
        engineered_frame["target_level_x_precipitation"] = (
            target_observed * df["precipitation"].shift(1)
        )
    return pd.concat([out, engineered_frame], axis=1)


def build_inference_frame(
    df: pd.DataFrame,
    base_features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if base_features is None:
        base_features = _select_feature_columns(df)
    feature_frame = _build_feature_frame(df, base_features)
    feature_columns = [column for column in feature_frame.columns if column != DATE_COLUMN]
    return feature_frame, feature_columns


def load_bundle(bundle_path: str | Path) -> dict[str, Any]:
    return joblib.load(bundle_path)


def _prediction_timestamp(df: pd.DataFrame, prediction_date: str | None) -> pd.Timestamp:
    if prediction_date is None:
        return df[DATE_COLUMN].max() + pd.Timedelta(days=1)
    return pd.Timestamp(prediction_date)


def _candidate_prediction_row(
    csv_path: str | Path,
    bundle_path: str | Path,
    prediction_date: str | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    df = load_data(csv_path)
    bundle = load_bundle(bundle_path)
    prediction_timestamp = _prediction_timestamp(df, prediction_date)
    df = _extend_to_prediction_date(df, prediction_timestamp)
    frame, _ = build_inference_frame(df, base_features=bundle["base_feature_columns"])

    candidate_rows = frame.loc[frame[DATE_COLUMN] == prediction_timestamp].copy()
    if candidate_rows.empty:
        raise ValueError(f"No feature row is available for prediction date {prediction_timestamp.date()}.")
    return bundle, candidate_rows.tail(1)


def predict_from_bundle(
    csv_path: str | Path,
    bundle_path: str | Path,
    prediction_date: str | None = None,
) -> pd.DataFrame:
    bundle, row = _candidate_prediction_row(csv_path, bundle_path, prediction_date)
    feature_columns = bundle["feature_columns"]
    feature_columns_by_horizon = bundle.get("feature_columns_by_horizon", {})
    conformal_by_horizon = bundle.get("conformal_intervals_by_horizon", {})

    missing_features = [column for column in feature_columns if column not in row.columns]
    if missing_features:
        preview = ", ".join(missing_features[:10])
        raise ValueError(f"Missing engineered feature columns required by the model: {preview}")

    forecast_date = row[DATE_COLUMN].iloc[0].date().isoformat()
    predictions: dict[str, Any] = {"prediction_date": forecast_date}
    for horizon in bundle["metadata"]["horizons"]:
        model_columns = feature_columns_by_horizon.get(horizon, feature_columns)
        value = bundle["models"][horizon].predict(row[model_columns])[0]
        predictions[f"week_{horizon}_ahead_avg"] = float(value)

        conformal = conformal_by_horizon.get(horizon, {})
        for alpha_label, quantile in conformal.get("alphas", {}).items():
            if np.isnan(quantile):
                continue
            coverage = int(round((1.0 - float(alpha_label)) * 100))
            predictions[f"week_{horizon}_ahead_lower_{coverage}"] = float(value - quantile)
            predictions[f"week_{horizon}_ahead_upper_{coverage}"] = float(value + quantile)

    return pd.DataFrame([predictions])


def predict_horizon_risk(
    csv_path: str | Path,
    bundle_path: str | Path,
    prediction_date: str | None = None,
) -> pd.DataFrame:
    bundle, row = _candidate_prediction_row(csv_path, bundle_path, prediction_date)
    probabilities = bundle["model"].predict_proba(row[bundle["feature_columns"]])[0]
    predicted_risk = bundle["model"].predict(row[bundle["feature_columns"]])[0]
    result = {
        "prediction_date": row[DATE_COLUMN].iloc[0].date().isoformat(),
        "horizon": int(bundle["metadata"]["horizon"]),
        "predicted_risk": predicted_risk,
    }
    for label, prob in zip(bundle["model"].classes_, probabilities):
        result[f"prob_{label}"] = float(prob)
    result["low_upper_q25"] = float(bundle["metadata"]["thresholds"]["low_upper_q25"])
    result["high_lower_quantile"] = float(bundle["metadata"]["thresholds"]["high_lower_quantile"])
    result["high_quantile"] = float(bundle["metadata"]["thresholds"].get("high_quantile", 0.75))
    result["high_threshold_mode"] = bundle["metadata"]["thresholds"].get(
        "high_threshold_mode",
        "training_quantile",
    )
    return pd.DataFrame([result])


def predict_horizon_high_risk(
    csv_path: str | Path,
    bundle_path: str | Path,
    prediction_date: str | None = None,
) -> pd.DataFrame:
    bundle, row = _candidate_prediction_row(csv_path, bundle_path, prediction_date)
    probabilities = bundle["model"].predict_proba(row[bundle["feature_columns"]])[0]
    class_index = {label: idx for idx, label in enumerate(bundle["model"].classes_)}
    prob_high = float(probabilities[class_index["high"]])
    threshold = float(bundle["metadata"]["probability_threshold"])
    predicted_label = "high" if prob_high >= threshold else "not_high"
    result = {
        "prediction_date": row[DATE_COLUMN].iloc[0].date().isoformat(),
        "horizon": int(bundle["metadata"]["horizon"]),
        "predicted_high_risk": predicted_label,
        "prob_high": prob_high,
        "probability_threshold": threshold,
        "high_lower_q75": float(bundle["metadata"]["high_lower_q75"]),
        "high_quantile": float(bundle["metadata"].get("high_quantile", 0.75)),
        "high_threshold_mode": bundle["metadata"].get("high_threshold_mode", "training_quantile"),
    }
    return pd.DataFrame([result])
