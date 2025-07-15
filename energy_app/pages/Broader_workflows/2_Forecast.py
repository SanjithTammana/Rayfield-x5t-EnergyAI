# energy_app/pages/Broader_workflows/2_Forecast.py
import streamlit as st
from energy_app.utils.predictors import forecast
from energy_app.utils.summariser import summarise
from energy_app.utils.zapier import fire_event

def show_forecast(df):
    """
    Forecasting page. Expects df passed in from Home.
    """
    if df is None:
        st.warning("Upload data in **Main** first.")
        return

    st.header("📈 Forecasting")
    model_name = st.selectbox(
        "Select model",
        ["solar", "wind", "hydro", "geothermal", "wood", "waste"],
    )
    horizon = st.slider("Forecast horizon (rows)", 1, 168, 24)

    # run forecast and catch any missing‐feature errors
    try:
        preds = forecast(model_name, df, horizon)
    except ValueError as e:
        st.error(f"Forecast error: {e}")
        return

    st.subheader("Forecast Plot")
    st.line_chart(preds, height=300)

    if st.button("Generate AI summary"):
        stats = {"mean": float(preds.mean()), "max": float(preds.max()), "min": float(preds.min())}
        summary_text = summarise(model_name, preds, stats)
        st.info(summary_text)
        fire_event("forecast_complete", {"model": model_name, "rows": len(preds)})
