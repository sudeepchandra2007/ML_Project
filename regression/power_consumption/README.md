# Household Power Consumption Regression

These programs predict `Global_active_power` (kilowatts) from the Household Power Consumption dataset. Put `household_power_consumption.txt` in this folder before running them. The dataset is not included in the repository.

The input is a semicolon-delimited text file. `?` entries are treated as missing; rows without a target are skipped, and remaining missing feature values are imputed inside each model pipeline.

The predictors include the other meter readings from the same timestamp and cyclical time features. This estimates a recorded observation; it is not a future-load forecast.

## Run

From this folder, install the dependencies and run each model:

```powershell
python -m pip install -r requirements.txt
python linear_gradient_descent.py
python polynomial_regression.py
python ridge_regression.py
python lasso_regression.py
```

Each program uses all valid observations by default. At the prompt, enter a smaller row count for a quicker run, or press Enter/type `all` to use the full dataset. You can skip the prompt with an environment variable, for example:

```powershell
$env:REGRESSION_SAMPLE_SIZE = "100000"
python ridge_regression.py
```

Rows stay in chronological order: the first 80% is training data and the final 20% is test data. When using a subset, the loader samples evenly across the full timeline. Use the same row count for all four programs when comparing their results. Ridge and Lasso use three-fold time-series cross-validation on the training portion to select alpha.

## Outputs

Results are written to `results_power_consumption/`. They include a model comparison CSV, per-model metric JSON and prediction CSV files, cross-validation tables, and RMSE charts. The polynomial program compares degrees 1, 2, and 3, plots train/test R², and saves a mean-only underfit example and a small-sample overfit example.

MSE is in kilowatts squared; RMSE is in kilowatts. The programs report Pearson correlation `R` and R². Lower MSE/RMSE is better; higher R/R² is better. A large train/test gap suggests overfitting, while weak train and test fit suggests underfitting.
