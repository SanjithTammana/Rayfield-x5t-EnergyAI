import plotly.express as px
import streamlit as st
import pandas as pd

def line(df: pd.DataFrame, y: str, title: str):
    fig = px.line(df, y=y, title=title)
    st.plotly_chart(fig, use_container_width=True)