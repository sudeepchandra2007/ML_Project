#!/usr/bin/env python3
"""Train KNN or RBF-SVM on the customer churn dataset."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive"
DROP_COLUMNS = ["id", "CustomerId", "Surname"]
TARGET = "Exited"


def load_data():
    train = pd.read_csv(ARCHIVE / "train.csv")
    test = pd.read_csv(ARCHIVE / "test.csv")
    features = [column for column in train.columns if column not in DROP_COLUMNS + [TARGET]]
    return train[features], train[TARGET], test[features], test["id"]


def make_preprocessor(features):
    categorical = features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = features.select_dtypes(exclude=["object", "string", "category"]).columns.tolist()
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def make_model(name, preprocessor, svm_samples):
    if name == "knn":
        classifier = KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)
    else:
        classifier = SVC(C=3.0, gamma="scale", kernel="rbf", cache_size=2048)
        if svm_samples:
            print(f"RBF-SVM training cap: {svm_samples:,} rows")
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["knn", "svm"], default="knn")
    parser.add_argument("--svm-samples", type=int, default=30000)
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    args = parser.parse_args()

    x, y, test, test_ids = load_data()
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = make_preprocessor(x)
    model = make_model(args.model, preprocessor, args.svm_samples if args.model == "svm" else 0)

    if args.model == "svm" and args.svm_samples < len(x_train):
        x_fit, _, y_fit, _ = train_test_split(
            x_train, y_train, train_size=args.svm_samples, random_state=42, stratify=y_train
        )
    else:
        x_fit, y_fit = x_train, y_train

    print(f"Training {args.model.upper()} on {len(x_fit):,} rows; validating on {len(x_valid):,} rows")
    model.fit(x_fit, y_fit)
    predictions = model.predict(x_valid)
    accuracy = accuracy_score(y_valid, predictions)
    print(f"Validation accuracy: {accuracy:.2%}")
    print(classification_report(y_valid, predictions, digits=4))
    if accuracy < 0.95:
        print("Note: this honest holdout result is below 95%; do not tune against the validation rows.")

    model.fit(x_fit, y_fit)
    test_predictions = model.predict(test)
    pd.DataFrame({"id": test_ids, TARGET: test_predictions}).to_csv(args.output, index=False)
    print(f"Wrote {len(test_predictions):,} predictions to {args.output}")


if __name__ == "__main__":
    main()