import pandas as pd
import numpy as np

def clean_user_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Basic auto-cleanup:
    • make column names lowercase & strip spaces
    • parse first datetime-like column (if any) to pd.DatetimeIndex
    • forward-fill, then back-fill remaining gaps
    """
    df = raw.copy()
    df.columns = df.columns.str.strip().str.lower()

    # Attempt to parse a date column
    for c in df.columns:
        if ("date" in c or "time" in c) and not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_datetime(df[c], errors="ignore")
            df = df.set_index(c)
            break

    # Normalise numeric dtypes
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Fill missing values
    df = df.ffill().bfill()

    return df
