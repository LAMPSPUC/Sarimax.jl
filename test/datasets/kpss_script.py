import json, os
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import kpss

def generate_stationary_series(length=200):
    np.random.seed(42)
    return np.random.randn(length).tolist()

def generate_nonstationary_series(length=200, trend=True, seasonality=False):
    np.random.seed(42)
    time = np.arange(length)
    series = np.cumsum(np.random.randn(length)) if trend else np.random.randn(length)
    if seasonality:
        series += 5 * np.sin(2 * np.pi * time / 50)  # Adding seasonality
    return series.tolist()

def perform_kpss_test(series):
    result = kpss(series, regression='c', nlags='legacy')
    return {
        "KPSS Statistic": result[0],
        "p-value": result[1],
        "Lags Used": result[2],
        "Critical Values": result[3]
    }

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def read_datasets_and_apply_kpss_test():
    # Directory containing the datasets
    DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASETS_DIR = os.path.abspath(os.path.join(DATASETS_DIR, '../../datasets'))

    files = [
        ('GDPC1.csv', ';', 'GDPC1'),
        ('NROU.csv', ';', 'NROU'),
        ('airpassengers.csv', ',', 'value'),
        ('log_airpassengers.csv', ',', 'value'),
    ]

    results = {}

    for fname, delim, value_col in files:
        if "log" in fname:
            use_log = True
            fname = fname.split("_")[1]
        else:
            use_log = False
        path = os.path.join(DATASETS_DIR, fname)
        df = pd.read_csv(path, delimiter=delim)
        # Handle possible quoted headers
        if value_col not in df.columns:
            # Try stripping quotes
            df.columns = [c.strip('"') for c in df.columns]
        series = df[value_col].astype(float)
        if use_log:
            series = np.log(series)
        result = kpss(series)
        if use_log:
            fname = "log_" + fname
        results[fname] = {
            'test_stat': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'critical_values': result[3]
        }

    # Save the results as json
    save_json(results, "kpss_results_datasets.json")

if __name__ == "__main__":
    # series_data = {}
    # kpss_results_data = {}
    # for i in range(1, 11):
    #     if i % 2 == 0:
    #         series = generate_nonstationary_series(trend=True, seasonality=(i % 4 == 0))
    #         label = f"nonstationary_series_{i}"
    #     else:
    #         series = generate_stationary_series()
    #         label = f"stationary_series_{i}"
    # series_data[label] = series
    # kpss_results_data[label] = perform_kpss_test(series)
    # save_json(series_data, "time_series.json")
    # save_json(kpss_results_data, "kpss_results.json")
    read_datasets_and_apply_kpss_test()
print("Files saved: time_series.json, kpss_results.json")