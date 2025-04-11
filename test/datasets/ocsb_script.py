import json
import numpy as np
import pandas as pd
from pmdarima.arima.seasonality import OCSBTest

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
    p_value = ocsb.is_stationary(series)
    return {"OCSB p-value": p_value}

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
series_data = {}
ocsb_results_data = {}

for i in range(1, 11):
    if i % 2 == 0:
        series = generate_nonstationary_series(trend=True, seasonality=(i % 4 == 0))
        label = f"nonstationary_series_{i}"
    else:
        series = generate_stationary_series()
        label = f"stationary_series_{i}"
    series_data[label] = series
    ocsb_results_data[label] = perform_ocsb_test(series)

save_json(series_data, "ocsb_time_series.json")
save_json(ocsb_results_data, "ocsb_results.json")

print("Files saved: ocsb_time_series.json, ocsb_results.json")
