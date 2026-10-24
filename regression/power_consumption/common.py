"""Shared data, preprocessing, metrics, and reporting helpers for the assignment."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


PROGRAM_DIR = Path(__file__).resolve().parent
DATA_PATH = PROGRAM_DIR / "household_power_consumption.txt"
RESULTS_DIR = PROGRAM_DIR / "results_power_consumption"
RANDOM_STATE = 42
TARGET = "Global_active_power"
DEFAULT_SAMPLE_SIZE = None  # Use all valid observations when the prompt is left blank.
MIN_SAMPLE_SIZE = 100

MEASUREMENT_COLUMNS = [
    "Global_reactive_power", "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]
POLYNOMIAL_COLUMNS = ["Global_reactive_power", "Voltage", "Global_intensity"]
TIME_COLUMNS = [
    "HOUR_SIN", "HOUR_COS", "WEEKDAY_SIN", "WEEKDAY_COS",
    "YEAR_DAY_SIN", "YEAR_DAY_COS", "YEAR_TREND",
]
NUMERIC_COLUMNS = MEASUREMENT_COLUMNS + TIME_COLUMNS
OTHER_NUMERIC_COLUMNS = [column for column in NUMERIC_COLUMNS if column not in POLYNOMIAL_COLUMNS]
RAW_COLUMNS = ["Date", "Time", TARGET, *MEASUREMENT_COLUMNS]


def _requested_sample_size(valid_rows: int) -> int:
    """Choose a row count; Enter/all uses the complete valid dataset by default."""
    configured = os.environ.get("REGRESSION_SAMPLE_SIZE")
    if configured:
        answer = configured.strip().lower()
    elif sys.stdin.isatty():
        answer = input(
            f"Number of valid power observations [default all; available {valid_rows:,}; "
            "enter 'all' for all]: "
        ).strip().lower()
    else:
        answer = "all"
        print(
            f"No interactive terminal detected; using all {valid_rows:,} valid observations. "
            "Set REGRESSION_SAMPLE_SIZE to a smaller count for a quicker run."
        )

    if not answer or answer in {"all", "max"}:
        return valid_rows
    try:
        count = int(answer.replace(",", ""))
    except ValueError as exc:
        raise ValueError("Enter a positive row count or 'all'.") from exc
    if count < MIN_SAMPLE_SIZE:
        raise ValueError(f"Choose at least {MIN_SAMPLE_SIZE} rows for the chronological split and CV.")
    return min(count, valid_rows)


def _read_power_data() -> pd.DataFrame:
    """Read only the target, six meter readings, and date/time columns."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Keep the household power dataset beside Regression-Programs."
        )
    numeric_dtype = {column: np.float32 for column in [TARGET, *MEASUREMENT_COLUMNS]}
    numeric_dtype.update({"Date": "string", "Time": "string"})
    frame = pd.read_csv(
        DATA_PATH,
        sep=";",
        usecols=RAW_COLUMNS,
        dtype=numeric_dtype,
        na_values=["?"],
        low_memory=False,
    )
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    return frame.loc[frame[TARGET].notna()].reset_index(drop=True)


def _select_evenly_spaced_rows(frame: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Sample across the complete timeline while preserving chronological order."""
    if sample_size >= len(frame):
        return frame.reset_index(drop=True)
    positions = np.linspace(0, len(frame) - 1, num=sample_size, dtype=np.int64)
    return frame.iloc[positions].reset_index(drop=True)


def _make_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build meter and cyclical calendar features for same-time power estimation."""
    features = frame[MEASUREMENT_COLUMNS].apply(pd.to_numeric, errors="coerce")
    dates = pd.to_datetime(frame["Date"], format="%d/%m/%Y", errors="coerce")
    hour = pd.to_numeric(frame["Time"].str.slice(0, 2), errors="coerce")
    weekday = dates.dt.dayofweek.astype("float64")
    year_day = dates.dt.dayofyear.astype("float64")
    hour_angle = 2.0 * np.pi * hour / 24.0
    weekday_angle = 2.0 * np.pi * weekday / 7.0
    year_day_angle = 2.0 * np.pi * (year_day - 1.0) / 365.25
    first_date = dates.min()
    year_trend = (dates - first_date).dt.total_seconds() / (365.25 * 24.0 * 60.0 * 60.0)

    features["HOUR_SIN"] = np.sin(hour_angle)
    features["HOUR_COS"] = np.cos(hour_angle)
    features["WEEKDAY_SIN"] = np.sin(weekday_angle)
    features["WEEKDAY_COS"] = np.cos(weekday_angle)
    features["YEAR_DAY_SIN"] = np.sin(year_day_angle)
    features["YEAR_DAY_COS"] = np.cos(year_day_angle)
    features["YEAR_TREND"] = year_trend
    target = frame[TARGET].astype(np.float64)
    return features[NUMERIC_COLUMNS], target


def load_split():
    """Load household power observations and use an ordered 80/20 split."""
    print(f"Reading household power data: {DATA_PATH}")
    frame = _read_power_data()
    valid_rows = len(frame)
    if valid_rows < MIN_SAMPLE_SIZE:
        raise ValueError(f"Only {valid_rows} valid target rows were found.")
    sample_size = _requested_sample_size(valid_rows)
    frame = _select_evenly_spaced_rows(frame, sample_size)
    X, y = _make_features(frame)

    split_index = int(len(frame) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    if len(X_train) < 80 or len(X_test) < 20:
        raise ValueError("The selected sample must contain at least 100 valid observations.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": "Individual Household Electric Power Consumption",
        "target": TARGET,
        "target_units": "kilowatts",
        "total_valid_rows": int(valid_rows),
        "rows_used": int(sample_size),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "split": "chronological 80/20; no shuffling",
        "sampling": "evenly spaced across the chronological file order",
        "features": NUMERIC_COLUMNS,
        "preprocessing": "median imputation and standard scaling inside model pipelines",
    }
    (RESULTS_DIR / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Target: {TARGET} (kilowatts)")
    print(f"Rows used: {len(frame):,}; train: {len(X_train):,}; test: {len(X_test):,}")
    print("Split: first 80% of the timeline for training; last 20% for testing, with no shuffling.")
    print("Other simultaneous meter readings are predictors; missing feature values are imputed in the pipeline.")
    return X_train, X_test, y_train, y_test


def make_preprocessor(degree=1):
    """Scale inputs and expand only three measurements to keep polynomial fitting tractable."""
    numeric_poly = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale_before_poly", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale_after_poly", StandardScaler()),
    ])
    numeric_other = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("poly_numeric", numeric_poly, POLYNOMIAL_COLUMNS),
        ("other_numeric", numeric_other, OTHER_NUMERIC_COLUMNS),
    ])


def scale_target(estimator):
    """Fit on a standardized target while returning predictions in kilowatts."""
    return TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())


def regression_metrics(y_true, y_pred):
    """Return MSE, RMSE, Pearson R, and R-squared."""
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = float("nan")
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "pearson_r": pearson_r,
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_model_result(model_name, model, X_train, y_train, X_test, y_test,
                      extra=None, save_predictions=True):
    """Save train/test metrics and append test metrics to a comparison CSV."""
    train_metrics = regression_metrics(y_train, model.predict(X_train))
    test_predictions = model.predict(X_test)
    test_metrics = regression_metrics(y_test, test_predictions)
    payload = {
        "model": model_name,
        "target": f"{TARGET} (kilowatts)",
        "train": train_metrics,
        "test": test_metrics,
        "train_test_r2_gap": train_metrics["r2"] - test_metrics["r2"],
    }
    if extra:
        payload.update(extra)

    safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
    with (RESULTS_DIR / f"{safe_name}_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)

    if save_predictions:
        pd.DataFrame({"actual_kw": np.asarray(y_test),
                      "predicted_kw": test_predictions}).to_csv(
            RESULTS_DIR / f"{safe_name}_predictions.csv", index=False
        )

    row = {
        "model": model_name,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "train_test_r2_gap": payload["train_test_r2_gap"],
    }
    if extra:
        row.update(extra)
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    previous = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    previous = previous[previous["model"] != model_name] if not previous.empty else previous
    pd.concat([previous, pd.DataFrame([row])], ignore_index=True).to_csv(
        comparison_path, index=False
    )
    return payload


def save_train_test_chart(payload, filename):
    """Save a train/test RMSE comparison in kilowatts."""
    labels = ["Train", "Test"]
    rmses = [payload["train"]["rmse"], payload["test"]["rmse"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, rmses, color=["#4c78a8", "#f58518"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylabel("RMSE (kilowatts; lower is better)")
    ax.set_title(f"Train vs test error: {payload['model']}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=160)
    plt.close(fig)
