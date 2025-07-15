# energy_app/Home.py

import streamlit as st
import pandas as pd
from pathlib import Path
import importlib.util, sys

# Ensure root directory is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_page_module(path: Path):
    """Dynamically load a .py as a module."""
    if not path.exists():
        raise FileNotFoundError(f"Page file not found: {path}")
    spec = importlib.util.spec_from_file_location(str(path), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(path)] = module
    spec.loader.exec_module(module)
    return module

# Fallbacks for the broader workflows only
def show_basic_visualise(df):
    st.header("📊 Basic Visualisation")
    if df is not None and not df.empty:
        st.dataframe(df.head())
    else:
        st.warning("No data available for visualization")

def show_basic_forecast(df):
    st.header("🔮 Basic Forecast")
    if df is not None and not df.empty:
        st.dataframe(df.describe())
    else:
        st.warning("No data available for forecasting")

st.set_page_config(page_title="EnergyAI", layout="wide")
st.title("⚡ EnergyAI")
st.markdown(
    """
- **Broader_workflows**: upload an energy‐consumption CSV, then Visualise & Forecast.  
- **Localized_workflows**: run the all-in-one inverter/efficiency/anomaly app.
"""
)

st.sidebar.title("🔍 Select Workflow")
workflow = st.sidebar.radio(
    "Workflow Type",
    ["Broader_workflows", "Localized_workflows"],
    key="workflow_selector"
)

BASE = Path(__file__).parent  # energy_app/

if workflow == "Broader_workflows":
    uploaded = st.sidebar.file_uploader("Upload energy CSV", type="csv", key="csv_uploader")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.user_df = df
        except:
            st.sidebar.error("❌ Could not read CSV.")
    if "user_df" not in st.session_state:
        st.sidebar.warning("Please upload a CSV to proceed.")
        st.stop()
    df = st.session_state.user_df

    page = st.sidebar.radio("Page", ["Visualise", "Forecast"], key="page_selector")
    if page == "Visualise":
        mod = load_page_module(
            BASE / "pages" / "Broader_workflows" / "1_Visualise.py"
        )
        try:
            mod.show_visualise(df)
        except Exception:
            st.error("⚠️ Visualise failed—showing fallback.")
            show_basic_visualise(df)
    else:
        mod = load_page_module(
            BASE / "pages" / "Broader_workflows" / "2_Forecast.py"
        )
        try:
            mod.show_forecast(df)
        except Exception:
            st.error("⚠️ Forecast failed—showing fallback.")
            show_basic_forecast(df)

else:  # Localized_workflows
    # Always load & run your one-file localized app (no fallback)
    mod = load_page_module(
        BASE / "pages" / "Localized_workflows" / "streamlit_app.py"
    )
    # show_localized() contains its own uploader and tabs
    mod.show_localized()
