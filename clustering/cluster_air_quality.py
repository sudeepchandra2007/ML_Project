"""Cluster Beijing air-quality stations from the PRSA archive.

The default unit of analysis is a station. Each station is represented by
robust distribution summaries of its pollutants and weather variables, then
clustered after imputation, scaling, and PCA. Use --mode hourly to cluster
individual hourly observations instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLUMNS = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"
]
POLLUTANT_COLUMNS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]


def load_archive(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("PRSA_Data_*.csv"))
    if not files:
        raise FileNotFoundError(f"No PRSA CSV files found in {input_dir}")
    frames = []
    for file in files:
        frame = pd.read_csv(file, na_values=["?", "NA", ""])
        frame["source_file"] = file.name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def station_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build compact, interpretable station-level distribution features."""
    aggregations = ["mean", "median", "std", "min", "max"]
    features = data.groupby("station")[NUMERIC_COLUMNS].agg(aggregations)
    features.columns = [f"{column}_{stat}" for column, stat in features.columns]

    # Wind direction is categorical; use each station's direction frequency.
    wind = pd.crosstab(data["station"], data["wd"], normalize="index").add_prefix("wd_share_")
    result = features.join(wind, how="left").reset_index()
    return result.replace([np.inf, -np.inf], np.nan)


def hourly_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return numeric hourly observations with cyclical time features."""
    result = data[["station", *NUMERIC_COLUMNS]].copy()
    result["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    result["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    return result


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    numeric = feature_frame.select_dtypes(include=np.number).columns.tolist()
    categorical = feature_frame.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ],
        remainder="drop",
    )


def evaluate_models(matrix: np.ndarray, max_k: int, random_state: int) -> pd.DataFrame:
    rows = []
    upper_k = min(max_k, len(matrix) - 1)
    for k in range(2, upper_k + 1):
        candidates = {
            "kmeans": KMeans(n_clusters=k, n_init=30, random_state=random_state),
            "agglomerative": AgglomerativeClustering(n_clusters=k, linkage="ward"),
            "gaussian_mixture": GaussianMixture(n_components=k, n_init=10, random_state=random_state),
        }
        for name, estimator in candidates.items():
            labels = estimator.fit_predict(matrix)
            if len(np.unique(labels)) < 2:
                continue
            rows.append({
                "algorithm": name,
                "n_clusters": k,
                "silhouette": silhouette_score(matrix, labels),
                "calinski_harabasz": calinski_harabasz_score(matrix, labels),
                "davies_bouldin": davies_bouldin_score(matrix, labels),
            })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("Could not evaluate a valid clustering configuration.")
    return result.sort_values(
        ["silhouette", "calinski_harabasz", "davies_bouldin"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def make_estimator(algorithm: str, n_clusters: int, random_state: int):
    if algorithm == "kmeans":
        return KMeans(n_clusters=n_clusters, n_init=50, random_state=random_state)
    if algorithm == "agglomerative":
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return GaussianMixture(n_components=n_clusters, n_init=20, random_state=random_state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("archive/PRSA_Data_20130301-20170228"))
    parser.add_argument("--output-dir", type=Path, default=Path("clustering_output"))
    parser.add_argument("--mode", choices=["station", "hourly"], default="station")
    parser.add_argument("--max-clusters", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data = load_archive(args.input_dir)
    observations = station_features(data) if args.mode == "station" else hourly_features(data)
    identifiers = observations[["station"]].copy()
    feature_frame = observations.drop(columns=["station"])

    preprocessor = build_preprocessor(feature_frame)
    transformed = preprocessor.fit_transform(feature_frame)
    transformed = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    pca = PCA(n_components=0.90, svd_solver="full", random_state=args.random_state)
    matrix = pca.fit_transform(transformed)

    metrics = evaluate_models(matrix, args.max_clusters, args.random_state)
    best = metrics.iloc[0]
    estimator = make_estimator(best["algorithm"], int(best["n_clusters"]), args.random_state)
    labels = estimator.fit_predict(matrix)
    identifiers["cluster"] = labels

    args.output_dir.mkdir(parents=True, exist_ok=True)
    identifiers.to_csv(args.output_dir / f"{args.mode}_cluster_assignments.csv", index=False)
    metrics.to_csv(args.output_dir / f"{args.mode}_model_comparison.csv", index=False)
    observations.assign(cluster=labels).to_csv(args.output_dir / f"{args.mode}_features.csv", index=False)
    if args.mode == "station":
        station_clusters = identifiers.set_index("station")["cluster"]
        profile = data.assign(cluster=data["station"].map(station_clusters)).groupby("cluster")[NUMERIC_COLUMNS].mean()
        profile.to_csv(args.output_dir / "station_cluster_profiles.csv")

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(matrix[:, 0], matrix[:, 1] if matrix.shape[1] > 1 else np.zeros(len(matrix)), c=labels, cmap="tab10", s=35)
    if args.mode == "station":
        for index, station in enumerate(identifiers["station"]):
            plt.annotate(station, (matrix[index, 0], matrix[index, 1] if matrix.shape[1] > 1 else 0), fontsize=8)
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title(f"{args.mode.title()} clusters: {best['algorithm']}, k={int(best['n_clusters'])}")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(args.output_dir / f"{args.mode}_clusters.png", dpi=180)
    plt.close()

    joblib.dump({"preprocessor": preprocessor, "pca": pca, "estimator": estimator}, args.output_dir / f"{args.mode}_clustering.joblib")
    print(f"Rows clustered: {len(observations):,}")
    print(f"Selected model: {best['algorithm']} with k={int(best['n_clusters'])}")
    print(f"Silhouette score: {best['silhouette']:.4f}")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()