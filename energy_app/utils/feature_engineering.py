import pandas as pd
import numpy as np

# ——— Mapping for quarterly aggregation (forecasts) ———
COLUMN_MAP = {
    "solar":      "solar energy",
    "wind":       "wind energy",
    "hydro":      "total renewable energy",
    "geothermal": "geothermal energy",
    "wood":       "wood energy",
    "waste":      "waste energy",
}

def aggregate_and_engineer(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Quarterly aggregation + lags, diffs, seasonal dummies, trend, plus
    geothermal’s capacity_expansion, rolling stats for anomaly if needed.
    """
    df = df.rename(columns=str.lower)
    # verify Year & Quarter exist
    if not {"year", "quarter"}.issubset(df.columns):
        raise ValueError("Your data must include 'Year' and 'Quarter' columns")
    raw_col = COLUMN_MAP.get(model_name)
    if raw_col not in df.columns:
        raise ValueError(f"Missing energy column '{raw_col}'")

    # 1) aggregate
    agg = (
        df.groupby(["year", "quarter"])[raw_col]
          .sum()
          .reset_index()
          .sort_values(["year","quarter"])
    )
    # 2) timestamp index + target rename
    agg["date"] = pd.PeriodIndex(year=agg.year, quarter=agg.quarter, freq="Q").to_timestamp()
    agg = agg[["date", raw_col]].rename(columns={raw_col: "y"}).reset_index(drop=True)

    # 3) lags & diffs
    for lag in range(1, 9):
        agg[f"lag_{lag}"] = agg.y.shift(lag)
    agg["lag_12"]      = agg.y.shift(12)
    agg["lag_4_diff"]  = agg.y.diff(4)
    agg["lag_12_diff"] = agg.y.diff(12)

    # 4) seasonal
    agg["q"]         = agg.date.dt.quarter
    agg["is_spring"] = (agg.q == 2).astype(int)
    agg["is_summer"] = (agg.q == 3).astype(int)
    agg["is_fall"]   = (agg.q == 4).astype(int)
    agg["is_winter"] = (agg.q == 1).astype(int)

    # 5) trend
    agg["time_trend"]    = np.arange(len(agg))
    agg["time_trend_sq"] = agg.time_trend ** 2

    # 6) geothermal‐specific
    if model_name == "geothermal":
        # capacity_expansion = y / cumulative sum of previous y's
        cum = agg.y.cumsum().shift(1).fillna(0)
        agg["capacity_expansion"] = agg.y / cum.replace(0, np.nan)
        agg["capacity_expansion"] = agg.capacity_expansion.fillna(0)

    return agg.dropna().reset_index(drop=True)

def classification_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    From 02_classification_regression.ipynb:
    - Year, Quarter
    - Sector_Encoded (one-hot)
    - lag-1 and lag-4 on 'total renewable energy'
    """
    df = df.rename(columns=str.lower)
    if not {"year","quarter","sector","total renewable energy"}.issubset(df.columns):
        raise ValueError("Need 'Year','Quarter','Sector','Total Renewable Energy' columns")
    # pivot to one row per (year,quarter,sector)
    tmp = df.groupby(["year","quarter","sector"])["total renewable energy"]\
            .sum().reset_index()
    tmp["lag_1"] = tmp["total renewable energy"].shift(1)
    tmp["lag_4"] = tmp["total renewable energy"].shift(4)
    # one-hot sector
    dummies = pd.get_dummies(tmp.sector, prefix="sector")
    out = pd.concat([tmp[["year","quarter","lag_1","lag_4"]], dummies], axis=1)
    return out.dropna().reset_index(drop=True)

def anomaly_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds on aggregate_and_engineer, then adds:
    - rolling mean/std (window=4)
    - rolling mean (8), max/min (4)
    - diff-1, diff-4, pct_change-1, pct_change-4
    - z_score, deviation_from_trend
    - quarter dummies Q_2, Q_3, Q_4
    """
    agg = aggregate_and_engineer(df, "hydro")  # hydro has all base features
    # rolling stats
    for w in [4,8]:
        agg[f"roll_mean_{w}"] = agg.y.rolling(window=w).mean()
    agg["roll_std_4"] = agg.y.rolling(4).std()
    agg["roll_max_4"] = agg.y.rolling(4).max()
    agg["roll_min_4"] = agg.y.rolling(4).min()
    # diffs and pct
    agg["diff_1"]       = agg.y.diff(1)
    agg["diff_4"]       = agg.y.diff(4)
    agg["pct_change_1"] = agg.y.pct_change(1)
    agg["pct_change_4"] = agg.y.pct_change(4)
    # trend‐deviation
    agg["z_score"]             = (agg.y - agg.y.mean()) / agg.y.std()
    agg["deviation_from_trend"] = agg.y - agg.time_trend

    # quarter dummies
    for q in [2,3,4]:
        agg[f"Q_{q}"] = (agg.q == q).astype(int)

    return agg.dropna().reset_index(drop=True)
