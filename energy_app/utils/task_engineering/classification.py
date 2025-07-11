import pandas as pd

def engineer(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    Build exactly bundle['features']:
      ['Year','Quarter','Sector_Encoded',
       'Total_Renewable_Lag1','Total_Renewable_Lag4']
    from the cleaned, lower-cased df.
    """
    # 1) Ensure the raw columns exist (clean_user_df lowercases everything)
    needed = ["year", "quarter", "sector", "total renewable energy"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Classification needs these raw columns: {needed}")

    # 2) Find the LabelEncoder in the bundle
    encoder = bundle.get("encoder") \
           or bundle.get("preprocessing", {}).get("label_encoder")
    if encoder is None:
        raise ValueError(
            "Classification bundle missing encoder under 'encoder' "
            "or 'preprocessing.label_encoder'"
        )

    # 3) Build the interim, lower-case feature DataFrame
    out = pd.DataFrame({
        "year":           df["year"],
        "quarter":        df["quarter"],
        "sector_encoded": encoder.transform(df["sector"]),
    })

    # 4) Add the two lag features
    tre = df["total renewable energy"]
    out["total_renewable_lag1"] = tre.shift(1)
    out["total_renewable_lag4"] = tre.shift(4)

    # 5) Drop initial NaN rows and reset the index
    out = out.dropna().reset_index(drop=True)

    # 6) Rename to exactly the names in bundle['features']
    col_map = dict(zip(
        ["year", "quarter", "sector_encoded",
         "total_renewable_lag1", "total_renewable_lag4"],
        bundle["features"]
    ))
    return out.rename(columns=col_map)
