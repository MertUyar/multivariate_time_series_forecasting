import pandas as pd
from scipy.signal import detrend
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks


def analyze_candidate_cycles():
    df = pd.read_csv(
        "hf://datasets/AIML-TUDA/dlam-ts-project-data-2026/train.csv"
    )

    df = df.sort_values(["series_id", "timestamp"])

    candidate_lags = [24, 48, 72, 168, 168 *2, 168*4, 168*4*2, 168*4*3, 168*4*4]

    results = []

    for series_id, g in df.groupby("series_id"):

        y = g["target"].astype(float)

        y = (
            y.interpolate()
             .bfill()
             .ffill()
             .values
        )

        y = detrend(y)

        acf_values = acf(
            y,
            nlags=max(candidate_lags),
            fft=True
        )

        row = {"series_id": series_id}

        for lag in candidate_lags:
            row[f"acf_{lag}"] = acf_values[lag]

        results.append(row)

    return pd.DataFrame(results)

acf_df = analyze_candidate_cycles()

print(acf_df.mean(numeric_only=True))

