# ML Project

## Classification dataset
  https://www.kaggle.com/datasets/rangalamahesh/bank-churn?resource=download

## Regression dataset
  https://www.kaggle.com/datasets/usdot/flight-delays

The Household Power Consumption regression programs are in [`regression/power_consumption/`](regression/power_consumption/). See that folder's README for dataset setup and run instructions; the dataset file itself is not included.

## Clustering dataset
  https://www.kaggle.com/datasets/victorbonilla/beijing-multisite-airquality-data-data-set

# results

## Classification
### KNN Validation Performance

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|--------:|
| 0 | 0.8805 | 0.9500 | 0.9139 | 26,023 |
| 1 | 0.7362 | 0.5195 | 0.6091 | 6,984 |
| **Accuracy** | | | **0.8589** | **33,007** |
| **Macro Avg** | **0.8083** | **0.7348** | **0.7615** | **33,007** |
| **Weighted Avg** | **0.8500** | **0.8589** | **0.8494** | **33,007** |

**Validation Accuracy: 85.89%**
### SVM Validation Performance

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|--------:|
| 0 | 0.8762 | 0.9620 | 0.9171 | 26,023 |
| 1 | 0.7773 | 0.4937 | 0.6039 | 6,984 |
| **Accuracy** | | | **0.8629** | **33,007** |
| **Macro Avg** | **0.8268** | **0.7279** | **0.7605** | **33,007** |
| **Weighted Avg** | **0.8553** | **0.8629** | **0.8508** | **33,007** |

**Validation Accuracy: 86.29%**
