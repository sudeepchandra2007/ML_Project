"""Compare polynomial feature degrees to illustrate model complexity."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from common import (RESULTS_DIR, load_split, make_preprocessor,
                    save_model_result, save_train_test_chart, scale_target)


def main():
    X_train, X_test, y_train, y_test = load_split()
    rows = []
    for degree in (1, 2, 3):
        print(f"Fitting polynomial regression, degree {degree}...")
        model = scale_target(Pipeline([
            ("preprocess", make_preprocessor(degree=degree)),
            ("regressor", LinearRegression()),
        ]))
        model.fit(X_train, y_train)
        name = f"Polynomial Regression (degree {degree})"
        result = save_model_result(name, model, X_train, y_train, X_test, y_test,
                                   extra={"degree": degree})
        save_train_test_chart(result, f"polynomial_degree_{degree}_train_test_rmse.png")
        rows.append({
            "degree": degree,
            "train_mse": result["train"]["mse"],
            "train_rmse": result["train"]["rmse"],
            "train_r": result["train"]["pearson_r"],
            "train_r2": result["train"]["r2"],
            "test_mse": result["test"]["mse"],
            "test_rmse": result["test"]["rmse"],
            "test_r": result["test"]["pearson_r"],
            "test_r2": result["test"]["r2"],
            "train_test_r2_gap": result["train_test_r2_gap"],
        })

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "polynomial_complexity.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["degree"], table["train_r2"], marker="o", label="Train R²")
    ax.plot(table["degree"], table["test_r2"], marker="o", label="Test R²")
    ax.set_xlabel("Polynomial degree (model complexity)")
    ax.set_ylabel("R² (higher is better)")
    ax.set_title("Underfitting and overfitting diagnostic")
    ax.set_xticks(table["degree"])
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "polynomial_complexity_r2.png", dpi=160)
    plt.close(fig)

    print("\nPolynomial degree comparison:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nInterpretation: low train and test R² suggests underfitting; a much higher train R² than test R² suggests overfitting.")
    print("The measured train/test gap is in polynomial_complexity.csv; judge the pattern from the actual results.")

    # Explicit examples for the assignment discussion. A mean-only model is a
    # deliberately underfit baseline. The cubic model has more predictors than
    # the tiny, spread-out training sample, making it an intentional overfit demo.
    demo_rows = []
    underfit_model = DummyRegressor(strategy="mean")
    underfit_model.fit(X_train, y_train)
    underfit_name = "Underfit Demo (mean predictor)"
    underfit = save_model_result(underfit_name, underfit_model, X_train, y_train,
                                 X_test, y_test)
    demo_rows.append({"example": underfit_name, "train_rmse": underfit["train"]["rmse"],
                      "test_rmse": underfit["test"]["rmse"],
                      "train_r2": underfit["train"]["r2"],
                      "test_r2": underfit["test"]["r2"],
                      "train_test_r2_gap": underfit["train_test_r2_gap"]})

    demo_size = min(12, len(X_train))
    demo_positions = np.linspace(0, len(X_train) - 1, num=demo_size, dtype=int)
    demo_X = X_train.iloc[demo_positions]
    demo_y = y_train.iloc[demo_positions]
    overfit_model = scale_target(Pipeline([
        ("preprocess", make_preprocessor(degree=3)),
        ("regressor", LinearRegression()),
    ]))
    overfit_model.fit(demo_X, demo_y)
    overfit_name = f"Overfit Demo (degree 3 on {demo_size} rows)"
    overfit = save_model_result(overfit_name, overfit_model, demo_X, demo_y,
                                X_test, y_test)
    demo_rows.append({"example": overfit_name, "train_rmse": overfit["train"]["rmse"],
                      "test_rmse": overfit["test"]["rmse"],
                      "train_r2": overfit["train"]["r2"],
                      "test_r2": overfit["test"]["r2"],
                      "train_test_r2_gap": overfit["train_test_r2_gap"]})
    demo_table = pd.DataFrame(demo_rows)
    demo_table.to_csv(RESULTS_DIR / "fit_behavior_demos.csv", index=False)
    print("\nExplicit fit behavior examples:")
    print(demo_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("Saved to:", RESULTS_DIR / "fit_behavior_demos.csv")


if __name__ == "__main__":
    main()
