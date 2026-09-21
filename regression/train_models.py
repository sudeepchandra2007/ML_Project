#!/usr/bin/env python3
"""Train Ridge/Lasso or Decision Tree regression on flight arrival delays."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive (1)"
TARGET = "ARRIVAL_DELAY"
FEATURES = [
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_TIME",
    "DISTANCE",
]
CATEGORICAL_FEATURES = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]


def load_data(max_rows):
    columns = FEATURES + [TARGET]
    dtypes = {column: "string" for column in CATEGORICAL_FEATURES}
    chunks = []
    for chunk in pd.read_csv(
        ARCHIVE / "flights.csv", usecols=columns, dtype=dtypes, chunksize=250_000
    ):
        chunk = chunk.dropna(subset=[TARGET])
        if max_rows and len(chunk) > max_rows // 10:
            chunk = chunk.sample(max_rows // 10, random_state=42)
        chunks.append(chunk)

    data = pd.concat(chunks, ignore_index=True)
    if max_rows and len(data) > max_rows:
        data = data.sample(max_rows, random_state=42)

    scheduled_departure = pd.to_numeric(data.pop("SCHEDULED_DEPARTURE"), errors="coerce")
    data["SCHEDULED_DEPARTURE_MINUTES"] = (scheduled_departure // 100) * 60 + scheduled_departure % 100
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    return data.drop(columns=[TARGET]), data[TARGET]


def make_preprocessor(features):
    categorical = features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)]
    )


def make_model(name, preprocessor):
    if name == "ridge":
        regressor = Ridge(alpha=10.0)
    elif name == "lasso":
        regressor = Lasso(alpha=0.01, max_iter=2_000)
    else:
        regressor = DecisionTreeRegressor(max_depth=20, min_samples_leaf=10, random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["ridge", "lasso", "tree"], default="ridge")
    parser.add_argument("--max-rows", type=int, default=250_000)
    parser.add_argument("--output", type=Path, default=ROOT / "validation_predictions.csv")
    args = parser.parse_args()

    features, target = load_data(args.max_rows)
    x_train, x_valid, y_train, y_valid = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = make_model(args.model, make_preprocessor(features))

    print(f"Training {args.model.upper()} on {len(x_train):,} rows; validating on {len(x_valid):,} rows")
    model.fit(x_train, y_train)
    predictions = model.predict(x_valid)
    print(f"Validation MAE: {mean_absolute_error(y_valid, predictions):.3f} minutes")
    print(f"Validation RMSE: {np.sqrt(mean_squared_error(y_valid, predictions)):.3f} minutes")
    print(f"Validation R2: {r2_score(y_valid, predictions):.4f}")

    pd.DataFrame({"actual_arrival_delay": y_valid, "predicted_arrival_delay": predictions}).to_csv(
        args.output, index=False
    )
    print(f"Wrote {len(predictions):,} validation predictions to {args.output}")


if __name__ == "__main__":
    main()