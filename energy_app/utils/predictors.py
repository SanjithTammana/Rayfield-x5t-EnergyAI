import pandas as pd
from energy_app.utils.loaders import load_bundle
from energy_app.utils.preprocess import clean_user_df

from energy_app.utils.task_engineering.forecast       import engineer as forecast_engineer
from energy_app.utils.task_engineering.classification import engineer as classification_engineer
from energy_app.utils.task_engineering.anomaly        import engineer as anomaly_engineer
from energy_app.utils.task_engineering.dimensionality import engineer as dimensionality_engineer

# List your forecasting model names exactly as in load_bundle
FORECAST_MODELS = {"solar","wind","hydro","geothermal","wood","waste"}

def _prep(bundle: dict, user_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    # Always start by cleaning & lower-casing columns
    df_clean = clean_user_df(user_df)

    if model_name == "classification":
        # classification_engineer expects cleaned-lower-case df & bundle encoder
        df_feat = classification_engineer(df_clean, bundle)
    elif model_name == "anomaly_detection":
        df_feat = anomaly_engineer(df_clean, bundle)
    elif model_name == "dimensionality_reduction":
        df_feat = dimensionality_engineer(df_clean, bundle)
    elif model_name in FORECAST_MODELS:
        df_feat = forecast_engineer(df_clean, model_name)
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    # Final alignment & scaling
    missing = [c for c in bundle["features"] if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Still missing features: {missing}")

    X = df_feat[bundle["features"]]
    return bundle["scaler"].transform(X)

def forecast(model_name: str, user_df: pd.DataFrame, horizon: int | None = None):
    b = load_bundle(model_name)
    Xs = _prep(b, user_df, model_name)
    h  = horizon or b.get("horizon", len(Xs))
    return b["model"].predict(Xs[-h:])

def classify(user_df: pd.DataFrame):
    b   = load_bundle("classification")
    Xs  = _prep(b, user_df, "classification")
    probs = b["model"].predict_proba(Xs)
    preds = b["model"].classes_[probs.argmax(axis=1)]
    return preds, probs

def detect_anomaly(user_df: pd.DataFrame):
    b   = load_bundle("anomaly_detection")
    Xs  = _prep(b, user_df, "anomaly_detection")
    flags = b["model"].predict(Xs)
    return flags == -1

def reduce_dim(user_df: pd.DataFrame):
    b   = load_bundle("dimensionality_reduction")
    Xs  = _prep(b, user_df, "dimensionality_reduction")
    return b["model"].transform(Xs)
