import pandas as pd

def engineer(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    Select exactly the raw columns listed in bundle['features']
    (or bundle['feature_columns']) from the cleaned DataFrame.
    """
    features = bundle.get("features") or bundle.get("feature_columns")
    if features is None:
        raise ValueError(
            "Dimensionality bundle missing 'features' or 'feature_columns' metadata"
        )

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Dimensionality needs these columns: {missing}")

    return df[features].dropna().reset_index(drop=True)
