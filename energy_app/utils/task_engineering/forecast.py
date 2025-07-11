# energy_app/utils/task_engineering/forecast.py
import pandas as pd
import numpy as np
from energy_app.utils.feature_engineering import COLUMN_MAP

def engineer(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    df = df.rename(columns=str.lower)
    if not {"year", "quarter"}.issubset(df.columns):
        raise ValueError("Your data must include 'Year' and 'Quarter' columns")
    raw_col = COLUMN_MAP.get(model_name)
    if raw_col not in df.columns:
        raise ValueError(f"Missing energy column '{raw_col}'")

    agg = (
        df.groupby(["year", "quarter"])[raw_col]
          .sum().reset_index()
          .sort_values(["year", "quarter"])
    )
    agg["date"] = pd.PeriodIndex(year=agg.year, quarter=agg.quarter, freq="Q")\
                  .to_timestamp()
    agg = agg[["date", raw_col]].rename(columns={raw_col: "y"}).reset_index(drop=True)

    # lags, diffs
    for lag in range(1, 9):
        agg[f"lag_{lag}"] = agg.y.shift(lag)
    agg["lag_12"]      = agg.y.shift(12)
    agg["lag_4_diff"]  = agg.y.diff(4)
    agg["lag_12_diff"] = agg.y.diff(12)

    # seasonal
    agg["q"]         = agg.date.dt.quarter
    agg["is_spring"] = (agg.q == 2).astype(int)
    agg["is_summer"] = (agg.q == 3).astype(int)
    agg["is_fall"]   = (agg.q == 4).astype(int)
    agg["is_winter"] = (agg.q == 1).astype(int)

    # trend
    agg["time_trend"]    = np.arange(len(agg))
    agg["time_trend_sq"] = agg.time_trend ** 2

    # geothermal extra
    if model_name == "geothermal":
        cum = agg.y.cumsum().shift(1).fillna(0)
        agg["capacity_expansion"] = (agg.y / cum.replace(0, np.nan)).fillna(0)

    return agg.dropna().reset_index(drop=True)
