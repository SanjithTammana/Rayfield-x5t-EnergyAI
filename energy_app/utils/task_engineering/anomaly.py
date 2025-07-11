import pandas as pd
import numpy as np
from energy_app.utils.task_engineering.forecast import engineer as base_forecast

def engineer(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    1) Start with hydro-based forecast features (lags, seasonal, trend)
    2) Add rolling stats, diffs, pct_changes, z_score, deviation_from_trend
    3) Add year_normalized = (year - min_year) / (max_year - min_year)
    4) Map to bundle['features']
    """
    # 1) Get the base features
    feat = base_forecast(df, "hydro")

    # 2) Rolling windows & stats
    for w in (4, 8):
        feat[f"roll_mean_{w}"] = feat["y"].rolling(w).mean()
    feat["roll_std_4"] = feat["y"].rolling(4).std()
    feat["roll_max_4"] = feat["y"].rolling(4).max()
    feat["roll_min_4"] = feat["y"].rolling(4).min()

    # 3) Diffs & percent-changes
    feat["diff_1"]       = feat["y"].diff(1)
    feat["diff_4"]       = feat["y"].diff(4)
    feat["pct_change_1"] = feat["y"].pct_change(1)
    feat["pct_change_4"] = feat["y"].pct_change(4)

    # 4) Trend deviation
    feat["z_score"]             = (feat["y"] - feat["y"].mean()) / feat["y"].std()
    feat["deviation_from_trend"] = feat["y"] - feat["time_trend"]

    # 5) Quarter dummies Q_2, Q_3, Q_4
    for q in (2, 3, 4):
        feat[f"Q_{q}"] = (feat["q"] == q).astype(int)

    # 6) Year normalization
    years = feat["date"].dt.year
    feat["year_normalized"] = (years - years.min()) / (years.max() - years.min())

    # 7) Final cleanup & map to bundle features
    feat = feat.dropna().reset_index(drop=True)
    return feat[bundle["features"]]
