#!/usr/bin/env python3
"""Experimental interpolation previews for weekly chlorophyll observations."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from shapely import contains_xy

# Bootstrap imports when executed as `python src/pipeline/weekly_chlorophyll_interpolation_experiment.py`.
_here = Path(__file__).resolve()
for parent in [_here.parent] + list(_here.parents):
    if (parent / "src").exists() and (parent / "utils").exists():
        sys.path.insert(0, str(parent))
        sys.path.insert(0, str(parent / "src"))
        break

from src.pipeline.chl_daily_coverage import _load_polygon_from_kml
from src.pipeline.weekly_chlorophyll_map_pipeline import (
    DEFAULT_GRID_SIZE_DEG,
    DEFAULT_POLYGON_FILE,
    DEFAULT_SOURCE_DIR,
    load_weekly_masked_observations,
)

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOGGER = logging.getLogger("weekly_chlorophyll_interpolation_experiment")

SCIPY_AVAILABLE = False
PYKRIGE_AVAILABLE = False
try:
    from scipy.interpolate import Rbf, griddata  # type: ignore
    from scipy.ndimage import gaussian_filter  # type: ignore

    SCIPY_AVAILABLE = True
except Exception:
    Rbf = None
    griddata = None
    gaussian_filter = None

try:
    import pykrige  # type: ignore  # noqa: F401

    PYKRIGE_AVAILABLE = True
except Exception:
    PYKRIGE_AVAILABLE = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Directory containing timestamped chlorophyll scene CSVs.",
    )
    parser.add_argument(
        "--polygon",
        default=str(DEFAULT_POLYGON_FILE),
        help="KML polygon used to clip the preview grid.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where interpolation previews and summary files should be written.",
    )
    parser.add_argument(
        "--preview-grid-size-deg",
        type=float,
        default=0.00005,
        help="Display-grid spacing in degrees for interpolation previews (default: %(default)s).",
    )
    parser.add_argument(
        "--idw-power",
        type=float,
        default=2.0,
        help="IDW power parameter (default: %(default)s).",
    )
    parser.add_argument(
        "--rbf-epsilon",
        type=float,
        default=0.00035,
        help="Gaussian RBF epsilon in degrees for the numpy fallback (default: %(default)s).",
    )
    parser.add_argument(
        "--rbf-smoothing",
        type=float,
        default=0.01,
        help="Diagonal smoothing added to the RBF system (default: %(default)s).",
    )
    parser.add_argument(
        "--idw-gaussian-only",
        action="store_true",
        help="Only render Gaussian-smoothed IDW outputs for the requested sigmas.",
    )
    parser.add_argument(
        "--gaussian-sigmas",
        default="10,20,40,60",
        help="Comma-separated Gaussian sigmas to apply to IDW (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _build_preview_grid(polygon_path: Path, grid_size_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    if grid_size_deg <= 0:
        raise ValueError("Preview grid size must be positive.")

    polygon = _load_polygon_from_kml(str(polygon_path))
    min_lon, min_lat, max_lon, max_lat = polygon.bounds
    lon_values = np.arange(min_lon, max_lon + grid_size_deg, grid_size_deg)
    lat_values = np.arange(min_lat, max_lat + grid_size_deg, grid_size_deg)
    lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
    mask = contains_xy(polygon, lon_grid, lat_grid)
    return lon_grid, lat_grid, mask, polygon


def _idw_interpolate(
    obs_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_values: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    power: float,
) -> np.ndarray:
    target_points = np.column_stack([target_lon, target_lat])
    obs_points = np.column_stack([obs_lon, obs_lat])
    interpolated = np.empty(target_points.shape[0], dtype=float)

    for idx, point in enumerate(target_points):
        diff = obs_points - point
        distance = np.sqrt(np.sum(diff * diff, axis=1))
        zero_distance = distance <= 1e-12
        if np.any(zero_distance):
            interpolated[idx] = float(obs_values[zero_distance][0])
            continue
        weights = 1.0 / np.power(distance, power)
        interpolated[idx] = float(np.sum(weights * obs_values) / np.sum(weights))

    return interpolated


def _numpy_rbf_interpolate(
    obs_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_values: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    epsilon: float,
    smoothing: float,
) -> np.ndarray:
    obs_points = np.column_stack([obs_lon, obs_lat])
    target_points = np.column_stack([target_lon, target_lat])

    diff_obs = obs_points[:, None, :] - obs_points[None, :, :]
    obs_distance_sq = np.sum(diff_obs * diff_obs, axis=2)
    kernel = np.exp(-obs_distance_sq / max(epsilon * epsilon, 1e-12))
    kernel += np.eye(kernel.shape[0]) * smoothing
    weights = np.linalg.solve(kernel, obs_values)

    diff_target = target_points[:, None, :] - obs_points[None, :, :]
    target_distance_sq = np.sum(diff_target * diff_target, axis=2)
    target_kernel = np.exp(-target_distance_sq / max(epsilon * epsilon, 1e-12))
    return target_kernel @ weights


def _scipy_linear_interpolate(
    obs_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_values: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    if not SCIPY_AVAILABLE or griddata is None:
        raise RuntimeError("scipy is not installed")
    points = np.column_stack([obs_lon, obs_lat])
    targets = np.column_stack([target_lon, target_lat])
    return griddata(points, obs_values, targets, method="linear")


def _scipy_nearest_interpolate(
    obs_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_values: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    if not SCIPY_AVAILABLE or griddata is None:
        raise RuntimeError("scipy is not installed")
    points = np.column_stack([obs_lon, obs_lat])
    targets = np.column_stack([target_lon, target_lat])
    return griddata(points, obs_values, targets, method="nearest")


def _gaussian_smooth_masked(surface: np.ndarray, polygon_mask: np.ndarray, sigma: float) -> np.ndarray:
    if not SCIPY_AVAILABLE or gaussian_filter is None:
        raise RuntimeError("scipy is not installed")
    valid_mask = polygon_mask & np.isfinite(surface)
    values = np.where(valid_mask, surface, 0.0)
    weights = valid_mask.astype(float)
    smooth_values = gaussian_filter(values, sigma=sigma, mode="nearest")
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    out = np.full(surface.shape, np.nan, dtype=float)
    safe = polygon_mask & (smooth_weights > 1e-12)
    out[safe] = smooth_values[safe] / smooth_weights[safe]
    return out


def _scipy_rbf_interpolate(
    obs_lon: np.ndarray,
    obs_lat: np.ndarray,
    obs_values: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
    epsilon: float,
    smoothing: float,
) -> np.ndarray:
    if not SCIPY_AVAILABLE or Rbf is None:
        raise RuntimeError("scipy is not installed")
    model = Rbf(obs_lon, obs_lat, obs_values, function="gaussian", epsilon=epsilon, smooth=smoothing)
    return model(target_lon, target_lat)


def interpolate_methods(
    observations,
    polygon_path: Path,
    preview_grid_size_deg: float,
    idw_power: float,
    rbf_epsilon: float,
    rbf_smoothing: float,
    gaussian_sigmas: list[float],
    idw_gaussian_only: bool,
) -> tuple[dict[str, dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, Any]:
    frame = observations.copy()
    frame["longitude"] = frame["longitude"].astype(float)
    frame["latitude"] = frame["latitude"].astype(float)
    frame["chlorophyll"] = frame["chlorophyll"].astype(float)
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["longitude", "latitude", "chlorophyll"])
    if frame.empty:
        raise ValueError("No valid weekly observations available for interpolation.")

    lon_grid, lat_grid, polygon_mask, polygon = _build_preview_grid(polygon_path, preview_grid_size_deg)
    target_lon = lon_grid[polygon_mask]
    target_lat = lat_grid[polygon_mask]

    obs_lon = frame["longitude"].to_numpy()
    obs_lat = frame["latitude"].to_numpy()
    obs_values = frame["chlorophyll"].to_numpy()

    methods: dict[str, dict[str, Any]] = {}

    idw_values = _idw_interpolate(obs_lon, obs_lat, obs_values, target_lon, target_lat, power=idw_power)
    idw_surface = np.full(lon_grid.shape, np.nan, dtype=float)
    idw_surface[polygon_mask] = idw_values
    methods["idw"] = {
        "surface": idw_surface,
        "valid_cell_count": int(np.isfinite(idw_surface).sum()),
        "status": "ok",
        "issue": None,
    }
    if SCIPY_AVAILABLE:
        for sigma in gaussian_sigmas:
            smoothed_surface = _gaussian_smooth_masked(idw_surface, polygon_mask, sigma=sigma)
            methods[f"idw_gaussian_sigma{int(sigma) if float(sigma).is_integer() else str(sigma).replace('.', '_')}"] = {
                "surface": smoothed_surface,
                "valid_cell_count": int(np.isfinite(smoothed_surface).sum()),
                "status": "ok",
                "issue": None,
                "smoothing_method": "gaussian_filter",
                "smoothing_sigma": sigma,
            }

    if idw_gaussian_only:
        gaussian_methods = {
            name: payload for name, payload in methods.items() if name.startswith("idw_gaussian_")
        }
        return gaussian_methods, lon_grid, lat_grid, polygon_mask, polygon

    try:
        if SCIPY_AVAILABLE:
            rbf_values = _scipy_rbf_interpolate(
                obs_lon, obs_lat, obs_values, target_lon, target_lat, epsilon=rbf_epsilon, smoothing=rbf_smoothing
            )
            rbf_status = "ok"
            rbf_issue = None
            method_name = "rbf_scipy"
        else:
            rbf_values = _numpy_rbf_interpolate(
                obs_lon, obs_lat, obs_values, target_lon, target_lat, epsilon=rbf_epsilon, smoothing=rbf_smoothing
            )
            rbf_status = "ok_numpy_fallback"
            rbf_issue = "scipy unavailable; used numpy Gaussian RBF fallback"
            method_name = "rbf"
        rbf_surface = np.full(lon_grid.shape, np.nan, dtype=float)
        rbf_surface[polygon_mask] = rbf_values
        methods[method_name] = {
            "surface": rbf_surface,
            "valid_cell_count": int(np.isfinite(rbf_surface).sum()),
            "status": rbf_status,
            "issue": rbf_issue,
        }
    except Exception as exc:
        methods["rbf"] = {
            "surface": np.full(lon_grid.shape, np.nan, dtype=float),
            "valid_cell_count": 0,
            "status": "error",
            "issue": str(exc),
        }

    if SCIPY_AVAILABLE:
        try:
            linear_values = _scipy_linear_interpolate(obs_lon, obs_lat, obs_values, target_lon, target_lat)
            linear_surface = np.full(lon_grid.shape, np.nan, dtype=float)
            linear_surface[polygon_mask] = linear_values
            methods["linear_griddata"] = {
                "surface": linear_surface,
                "valid_cell_count": int(np.isfinite(linear_surface).sum()),
                "status": "ok",
                "issue": None,
            }

            nearest_values = _scipy_nearest_interpolate(obs_lon, obs_lat, obs_values, target_lon, target_lat)
            hybrid_values = linear_values.copy()
            hybrid_nan = ~np.isfinite(hybrid_values)
            hybrid_values[hybrid_nan] = nearest_values[hybrid_nan]
            hybrid_surface = np.full(lon_grid.shape, np.nan, dtype=float)
            hybrid_surface[polygon_mask] = hybrid_values
            methods["linear_nearest_fill"] = {
                "surface": hybrid_surface,
                "valid_cell_count": int(np.isfinite(hybrid_surface).sum()),
                "status": "ok",
                "issue": None,
            }
        except Exception as exc:
            methods["linear_griddata"] = {
                "surface": np.full(lon_grid.shape, np.nan, dtype=float),
                "valid_cell_count": 0,
                "status": "error",
                "issue": str(exc),
            }
            methods["linear_nearest_fill"] = {
                "surface": np.full(lon_grid.shape, np.nan, dtype=float),
                "valid_cell_count": 0,
                "status": "error",
                "issue": str(exc),
            }
    else:
        methods["linear_griddata"] = {
            "surface": np.full(lon_grid.shape, np.nan, dtype=float),
            "valid_cell_count": 0,
            "status": "unavailable",
            "issue": "scipy is not installed",
        }
        methods["linear_nearest_fill"] = {
            "surface": np.full(lon_grid.shape, np.nan, dtype=float),
            "valid_cell_count": 0,
            "status": "unavailable",
            "issue": "scipy is not installed",
        }

    methods["kriging"] = {
        "surface": np.full(lon_grid.shape, np.nan, dtype=float),
        "valid_cell_count": 0,
        "status": "unavailable",
        "issue": None if PYKRIGE_AVAILABLE else "pykrige is not installed",
    }

    return methods, lon_grid, lat_grid, polygon_mask, polygon


def _plot_polygon_outline(ax: plt.Axes, polygon) -> None:
    geoms = getattr(polygon, "geoms", [polygon])
    for geom in geoms:
        x_coords, y_coords = geom.exterior.xy
        ax.plot(x_coords, y_coords, color="black", linewidth=1.0, alpha=0.9, zorder=3)


def _shared_color_limits(methods: dict[str, dict[str, Any]], observations) -> tuple[float, float]:
    finite_values = []
    for payload in methods.values():
        surface = payload["surface"]
        if surface is not None:
            values = surface[np.isfinite(surface)]
            if values.size:
                finite_values.append(values)
    obs_values = observations["chlorophyll"].to_numpy()
    if np.isfinite(obs_values).any():
        finite_values.append(obs_values[np.isfinite(obs_values)])
    if not finite_values:
        return 0.0, 1.0
    merged = np.concatenate(finite_values)
    vmin = float(np.nanpercentile(merged, 5))
    vmax = float(np.nanpercentile(merged, 95))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(merged))
        vmax = float(np.nanmax(merged))
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def _render_method_png(
    output_path: Path,
    method_name: str,
    surface: np.ndarray,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    polygon,
    vmin: float,
    vmax: float,
    start_timestamp: str,
    end_timestamp: str,
    status: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    mesh = ax.pcolormesh(
        lon_grid,
        lat_grid,
        surface,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )
    _plot_polygon_outline(ax, polygon)
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("Chlorophyll (mg/m^3)")
    mean_lat = float(np.nanmean(lat_grid))
    aspect = 1.0 / max(math.cos(math.radians(mean_lat)), 1e-6)
    ax.set_aspect(aspect)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Weekly Chlorophyll Interpolation Preview: {method_name}\n"
        f"{start_timestamp} to {end_timestamp} ({status})"
    )
    ax.grid(True, alpha=0.15, linewidth=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _render_comparison_png(
    output_path: Path,
    methods: dict[str, dict[str, Any]],
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    polygon,
    vmin: float,
    vmax: float,
    start_timestamp: str,
    end_timestamp: str,
) -> None:
    names = list(methods.keys())
    cols = 2
    rows = int(math.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.8 * rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    mesh = None
    for ax, name in zip(axes_flat, names):
        payload = methods[name]
        surface = payload["surface"]
        mesh = ax.pcolormesh(
            lon_grid,
            lat_grid,
            surface,
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            zorder=1,
        )
        _plot_polygon_outline(ax, polygon)
        mean_lat = float(np.nanmean(lat_grid))
        aspect = 1.0 / max(math.cos(math.radians(mean_lat)), 1e-6)
        ax.set_aspect(aspect)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{name}\n{payload['status']}")
        ax.grid(True, alpha=0.15, linewidth=0.5)
    for ax in axes_flat[len(names) :]:
        ax.axis("off")
    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=axes_flat.tolist(), shrink=0.9)
        colorbar.set_label("Chlorophyll (mg/m^3)")
    fig.suptitle(f"Weekly Chlorophyll Interpolation Comparison\n{start_timestamp} to {end_timestamp}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(
    source_dir: Path,
    polygon_path: Path,
    output_dir: Path,
    preview_grid_size_deg: float,
    idw_power: float,
    rbf_epsilon: float,
    rbf_smoothing: float,
    gaussian_sigmas: list[float],
    idw_gaussian_only: bool,
) -> dict[str, Any]:
    observations, start_date, end_date, included_dates, source_file_count = load_weekly_masked_observations(
        source_dir=source_dir,
        polygon_path=polygon_path,
    )
    methods, lon_grid, lat_grid, polygon_mask, polygon = interpolate_methods(
        observations=observations,
        polygon_path=polygon_path,
        preview_grid_size_deg=preview_grid_size_deg,
        idw_power=idw_power,
        rbf_epsilon=rbf_epsilon,
        rbf_smoothing=rbf_smoothing,
        gaussian_sigmas=gaussian_sigmas,
        idw_gaussian_only=idw_gaussian_only,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    vmin, vmax = _shared_color_limits(methods, observations)
    start_timestamp = f"{start_date.isoformat()}T00:00:00"
    end_timestamp = f"{end_date.isoformat()}T23:59:59"

    summary_methods: dict[str, Any] = {}
    for method_name, payload in methods.items():
        png_path = output_dir / f"weekly_chlorophyll_interpolation_{method_name}.png"
        _render_method_png(
            output_path=png_path,
            method_name=method_name,
            surface=payload["surface"],
            lon_grid=lon_grid,
            lat_grid=lat_grid,
            polygon=polygon,
            vmin=vmin,
            vmax=vmax,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            status=payload["status"],
        )
        summary_methods[method_name] = {
            "status": payload["status"],
            "issue": payload["issue"],
            "valid_cell_count": payload["valid_cell_count"],
            "hole_cell_count": int(polygon_mask.sum()) - int(payload["valid_cell_count"]),
            "min": float(np.nanmin(payload["surface"])) if np.isfinite(payload["surface"]).any() else None,
            "median": float(np.nanmedian(payload["surface"])) if np.isfinite(payload["surface"]).any() else None,
            "max": float(np.nanmax(payload["surface"])) if np.isfinite(payload["surface"]).any() else None,
            "negative_cell_count": int(np.sum(payload["surface"][np.isfinite(payload["surface"])] < 0))
            if np.isfinite(payload["surface"]).any()
            else 0,
            "smoothing_method": payload.get("smoothing_method"),
            "smoothing_sigma": payload.get("smoothing_sigma"),
            "preview_png": str(png_path.resolve()),
        }

    comparison_path = output_dir / "weekly_chlorophyll_interpolation_comparison.png"
    _render_comparison_png(
        output_path=comparison_path,
        methods=methods,
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        polygon=polygon,
        vmin=vmin,
        vmax=vmax,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )

    summary = {
        "source_directory": str(source_dir.resolve()),
        "polygon_file": str(polygon_path.resolve()),
        "included_dates": included_dates,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "source_file_count": source_file_count,
        "masked_observation_count": int(len(observations)),
        "preview_grid_size_degrees": preview_grid_size_deg,
        "base_weekly_grid_size_degrees": DEFAULT_GRID_SIZE_DEG,
        "polygon_grid_cell_count": int(polygon_mask.sum()),
        "shared_color_scale": {"vmin": vmin, "vmax": vmax},
        "dependency_status": {
            "scipy_available": SCIPY_AVAILABLE,
            "pykrige_available": PYKRIGE_AVAILABLE,
        },
        "methods": summary_methods,
        "comparison_png": str(comparison_path.resolve()),
    }
    summary_path = output_dir / "weekly_chlorophyll_interpolation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    LOGGER.info("Wrote interpolation experiment summary to %s", summary_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    args = parse_args(argv)
    try:
        gaussian_sigmas = [
            float(part.strip()) for part in args.gaussian_sigmas.split(",") if part.strip()
        ]
        run(
            source_dir=Path(args.source_dir),
            polygon_path=Path(args.polygon),
            output_dir=Path(args.output_dir),
            preview_grid_size_deg=args.preview_grid_size_deg,
            idw_power=args.idw_power,
            rbf_epsilon=args.rbf_epsilon,
            rbf_smoothing=args.rbf_smoothing,
            gaussian_sigmas=gaussian_sigmas,
            idw_gaussian_only=args.idw_gaussian_only,
        )
    except Exception as exc:
        LOGGER.error("Weekly chlorophyll interpolation experiment failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
