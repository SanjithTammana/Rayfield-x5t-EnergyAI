# energy_app/pages/Broader_workflows/1_Visualise.py
import streamlit as st
from energy_app.components.charts import line

def show_visualise(df):
    """
    Data‐visualisation page. Expects df passed in from Home.
    """
    if df is None:
        st.warning("Upload data in **Main** first.")
        return

    st.header("📊 Data Visualisation")
    # pick only numeric columns
    num_cols = df.select_dtypes(include="number").columns
    if not len(num_cols):
        st.info("No numeric columns found for visualization.")
        return

    col = st.selectbox("Pick a numeric column", num_cols)
    # draw chart via your helper
    line(df, col, f"Trend of {col}")
    # show summary stats
    st.subheader("Descriptive Statistics")
    st.dataframe(df[num_cols].describe())
