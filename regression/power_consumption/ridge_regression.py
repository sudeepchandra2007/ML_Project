"""Ridge regression with time-series cross-validation for alpha selection."""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from common import (RESULTS_DIR, load_split, make_preprocessor,
                    save_model_result, save_train_test_chart, scale_target)


def main():
    X_train, X_test, y_train, y_test = load_split()
    model = scale_target(Pipeline([
        ("preprocess", make_preprocessor(degree=2)),
        ("regressor", Ridge(solver="lsqr")),
    ]))
    search = GridSearchCV(
        model,
        {"regressor__regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
        scoring="neg_mean_squared_error",
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=1,
        verbose=2,
        return_train_score=True,
    )
    print("Selecting Ridge alpha with 3-fold time-series cross-validation...")
    search.fit(X_train, y_train)
    cv = pd.DataFrame(search.cv_results_)
    cv["mean_cv_mse"] = -cv["mean_test_score"]
    cv["std_cv_mse"] = cv["std_test_score"]
    cv.to_csv(RESULTS_DIR / "ridge_cv_results.csv", index=False)

    result = save_model_result(
        "Ridge Regression", search.best_estimator_, X_train, y_train, X_test, y_test,
        extra={"best_alpha": float(search.best_params_["regressor__regressor__alpha"]),
               "best_cv_mse": float(-search.best_score_)},
    )
    save_train_test_chart(result, "ridge_train_test_rmse.png")
    print("Best alpha:", search.best_params_["regressor__regressor__alpha"])
    print(f"Best 3-fold CV MSE: {-search.best_score_:.4f}")
    print("Test metrics:", result["test"])
    print("CV results saved to:", RESULTS_DIR / "ridge_cv_results.csv")


if __name__ == "__main__":
    main()
