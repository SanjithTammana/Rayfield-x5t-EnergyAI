import streamlit as st
import pandas as pd
from utils.preprocess import clean_user_df

def upload_widget():
    uploaded = st.file_uploader("📤 Upload a CSV file", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        cleaned = clean_user_df(df)
        st.session_state["user_df"] = cleaned
        st.success(f"Loaded {cleaned.shape[0]:,} rows × {cleaned.shape[1]} cols")