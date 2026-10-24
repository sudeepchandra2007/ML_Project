"""Linear regression optimized with stochastic gradient descent."""

from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline

from common import (RANDOM_STATE, RESULTS_DIR, load_split, make_preprocessor,
                    save_model_result, save_train_test_chart, scale_target)


def main():
    X_train, X_test, y_train, y_test = load_split()
    model = scale_target(Pipeline([
        ("preprocess", make_preprocessor(degree=1)),
        ("regressor", SGDRegressor(
            loss="squared_error", penalty=None, learning_rate="adaptive",
            eta0=0.01, max_iter=2000, tol=1e-4, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10,
            average=True, random_state=RANDOM_STATE,
        )),
    ]))
    print("Training linear regression with SGD (gradient descent)...")
    model.fit(X_train, y_train)
    result = save_model_result("Linear Regression (SGD)", model, X_train, y_train,
                               X_test, y_test)
    save_train_test_chart(result, "linear_sgd_train_test_rmse.png")
    print("Test metrics:", result["test"])
    print(f"Train/test R2 gap: {result['train_test_r2_gap']:.4f}")
    print("Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
