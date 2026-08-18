# Time Series Analysis: Airline Passengers

A compact, notebook-first project for learning core time series analysis and forecasting techniques with monthly airline passenger data. The repository demonstrates how to inspect a time series, evaluate stationarity, transform non-stationary data, and build baseline ARIMA/SARIMAX forecasting models.

## Repository Contents

| Path | Description |
| --- | --- |
| `time_series_air_passengers.ipynb` | Main Jupyter notebook containing the complete exploratory analysis, stationarity checks, transformations, ARIMA modeling, and SARIMAX modeling workflow. |
| `airline-passengers.csv` | Monthly airline passenger counts from January 1949 through December 1960. This file is useful for running the analysis without relying on Seaborn's bundled dataset. |
| `README.md` | Project documentation, setup instructions, workflow notes, and troubleshooting guidance. |

## Project Goals

This project is intended to help you understand and practice:

- Loading and preparing monthly time series data.
- Visualizing trend and seasonality in airline passenger volume.
- Checking whether a time series is stationary.
- Using rolling mean and rolling standard deviation diagnostics.
- Running the Augmented Dickey-Fuller (ADF) stationarity test.
- Applying common transformations such as differencing, logarithms, square roots, and cube roots.
- Interpreting ACF and PACF plots for ARIMA parameter selection.
- Training and evaluating ARIMA models.
- Training and evaluating seasonal SARIMAX models.
- Comparing forecasts against held-out test data with RMSE.

## Dataset

The included `airline-passengers.csv` file contains two columns:

| Column | Description |
| --- | --- |
| `month` | Month in `YYYY-MM` format. |
| `total_passengers` | Number of airline passengers for the month. |

The notebook currently uses Seaborn's built-in `flights` dataset, which covers the same classic monthly airline passenger series. If you prefer to use the CSV file included in this repository, see [Using the local CSV instead of Seaborn](#using-the-local-csv-instead-of-seaborn).

## Analysis Workflow

The notebook walks through the following steps.

### 1. Import dependencies

The workflow uses common Python data science and forecasting libraries:

- `pandas` for tabular data manipulation.
- `numpy` for numerical transformations.
- `matplotlib` and `seaborn` for visualization.
- `statsmodels` for ADF tests, ACF/PACF plots, ARIMA, and SARIMAX.
- `scikit-learn` for model evaluation with RMSE.

### 2. Load and index the data

The notebook loads monthly passenger data, creates a monthly datetime field, and sets it as the DataFrame index. A datetime index is important because time series models and plots depend on chronological ordering.

### 3. Visualize the raw series

A line chart is used to inspect the passenger trend over time. The series shows:

- A strong upward long-term trend.
- Clear yearly seasonality.
- Increasing variance as passenger volume grows.

These characteristics indicate that the raw passenger series is not stationary.

### 4. Check stationarity

The project demonstrates two stationarity checks:

1. **Rolling statistics**: calculates 12-month rolling mean and rolling standard deviation.
2. **Augmented Dickey-Fuller test**: uses `statsmodels.tsa.stattools.adfuller` to test for a unit root.

A reusable `test_stationarity` helper function calculates rolling statistics, runs the ADF test, prints critical values, and plots the original/transformed series alongside rolling diagnostics.

### 5. Make the series more stationary

The notebook explores multiple transformations:

- One-period differencing.
- Log transformation.
- Square-root transformation.
- Cube-root transformation.
- Combined transformation and differencing.

These transformations are used to reduce trend, stabilize variance, and improve suitability for ARIMA-style modeling.

### 6. Build an ARIMA model

The notebook creates differenced columns, plots ACF and PACF charts, selects an example ARIMA order, splits the data into training and test sets, fits an ARIMA model, and plots predictions against actual passenger counts.

The demonstrated ARIMA configuration is:

```python
ARIMA(train['passengers'], order=(1, 1, 3))
```

### 7. Build a SARIMAX model

Because monthly airline passenger data has strong annual seasonality, the notebook also fits a SARIMAX model with a 12-month seasonal period:

```python
SARIMAX(
    train['passengers'],
    order=(1, 1, 3),
    seasonal_order=(1, 1, 3, 12),
)
```

The SARIMAX model is expected to better capture seasonal behavior than the non-seasonal ARIMA baseline.

### 8. Evaluate forecasts

Forecasts are evaluated with root mean squared error (RMSE):

```python
np.sqrt(mean_squared_error(test['passengers'], prediction))
```

RMSE provides an interpretable error value in passenger-count units.

## Getting Started

### Prerequisites

Install Python 3.10 or newer. The notebook metadata was last saved with Python 3.13, but the code should work on recent Python 3 versions supported by the listed dependencies.

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter
```

### Start Jupyter

```bash
jupyter notebook
```

Then open:

```text
time_series_air_passengers.ipynb
```

Run the notebook cells from top to bottom.

## Using the local CSV instead of Seaborn

The notebook currently loads data with:

```python
df = sns.load_dataset('flights')
df['yearMonth'] = pd.to_datetime('01-' + df['month'].astype(str) + '-' + df['year'].astype(str))
df.set_index('yearMonth', inplace=True)
```

To use `airline-passengers.csv` instead, replace the loading cells with:

```python
import pandas as pd

csv_df = pd.read_csv('airline-passengers.csv')
csv_df['month'] = pd.to_datetime(csv_df['month'])
csv_df = csv_df.rename(columns={'total_passengers': 'passengers'})
csv_df = csv_df.set_index('month')

# Match the variable name used throughout the notebook.
df = csv_df
```

## Recommended Notebook Improvements

The current notebook is useful as a learning artifact. If you plan to extend it, consider adding:

- A `requirements.txt` or `pyproject.toml` for reproducible dependency installation.
- Markdown explanations before each major code section.
- Explicit train/test date ranges in the output.
- Model diagnostics such as residual plots and Ljung-Box tests.
- A final future-forecast section after the existing `predict Future Model` heading.
- A comparison table for ARIMA and SARIMAX RMSE values.
- Saved plots or generated reports for easier sharing.

## Troubleshooting

### `ModuleNotFoundError`

If imports fail, install the required package in your active environment:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter
```

### Seaborn dataset download issues

If `sns.load_dataset('flights')` fails because the environment cannot download bundled datasets, use the included `airline-passengers.csv` file instead.

### Pandas chained assignment warnings

Some notebook cells create derived DataFrames and assign columns directly. If you see `SettingWithCopyWarning`, make an explicit copy before assigning new columns:

```python
log_df = df[['passengers']].copy()
log_df['log'] = np.log(log_df['passengers'])
```

### SARIMAX fitting takes a long time

SARIMAX can be slower than ARIMA because it estimates seasonal parameters. If fitting is slow, try a simpler seasonal order first, such as:

```python
seasonal_order=(1, 1, 1, 12)
```

## Suggested Project Structure for Future Work

For a more production-oriented version, consider evolving the repository into this structure:

```text
.
├── data/
│   └── airline-passengers.csv
├── notebooks/
│   └── time_series_air_passengers.ipynb
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

This would separate data loading, feature engineering, modeling, and evaluation logic from notebook exploration.

## License
Felix Amenya
No license file is currently included. Feel free to Clone this project and use it to learn.
