import json
import numpy as np
import pandas as pd
from pmdarima.arima.seasonality import OCSBTest
import os

def generate_stationary_series(length=200):
    np.random.seed(42)
    return np.random.randn(length).tolist()

def generate_nonstationary_series(length=200, trend=True, seasonality=False):
    np.random.seed(42)
    time = np.arange(length)
    series = np.cumsum(np.random.randn(length)) if trend else np.random.randn(length)
    if seasonality:
        series += 5 * np.sin(2 * np.pi * time / 12)  # Monthly seasonality
    return series.tolist()

def perform_ocsb_test(series):
    ocsb = OCSBTest(m=12)
    series = np.array(series)
    return {"D": ocsb.estimate_seasonal_differencing_term(series), "OCSB test statistic": ocsb._compute_test_statistic(series)}

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def read_datasets_and_apply_ocsb_test():
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
        ocsb = OCSBTest(m=12)
        D = ocsb.estimate_seasonal_differencing_term(series.values)
        test_stat = ocsb._compute_test_statistic(series.values)
        if use_log:
            fname = "log_" + fname
        results[fname] = {
            'test_stat': test_stat,
            'D': D
        }

    # Save the results as json
    save_json(results, "ocsb_results_datasets.json")

if __name__ == "__main__":
    # series_data = {}
    # ocsb_results_data = {}

    # for i in range(1, 11):
    #     if i % 2 == 0:
    #         series = generate_nonstationary_series(trend=True, seasonality=(i % 4 == 0))
    #         label = f"nonstationary_series_{i}"
    #     else:
    #         series = generate_stationary_series()
    #         label = f"stationary_series_{i}"
    #     series_data[label] = series
    #     ocsb_results_data[label] = perform_ocsb_test(series)

    # save_json(series_data, "ocsb_time_series.json")
    # save_json(ocsb_results_data, "ocsb_results.json")

    read_datasets_and_apply_ocsb_test()

    print("Files saved: ocsb_time_series.json, ocsb_results.json")
