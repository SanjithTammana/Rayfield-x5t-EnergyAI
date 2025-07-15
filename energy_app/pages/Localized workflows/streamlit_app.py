import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import date

# ---------- Inverter-Specific Preprocessing ----------
def preprocess_inverter_data(df: pd.DataFrame) -> pd.DataFrame:
    if 'SOURCE_KEY' not in df.columns or 'DATE_TIME' not in df.columns:
        return None

    df = df.copy()
    df['SOURCE_KEY'] = df['SOURCE_KEY'].fillna('UNKNOWN')
    unique_ids = df['SOURCE_KEY'].unique()
    id_mapping = {uid: f"S{i+1}" for i, uid in enumerate(unique_ids)}
    df['SOURCE_ID'] = df['SOURCE_KEY'].map(id_mapping)
    df['SOURCE_ID_NUMBER'] = df['SOURCE_ID'].str.extract(r'(\d+)').astype(int)

    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], errors='coerce')
    df = df.dropna(subset=['DATE_TIME'])

    df = df.rename(columns={
        'AC_POWER': 'AC_POWER_OUTPUT',
        'DC_POWER': 'DC_POWER_INPUT'
    })

    # ensure numeric
    df['AC_POWER_OUTPUT'] = pd.to_numeric(df['AC_POWER_OUTPUT'], errors='coerce')
    df['DC_POWER_INPUT']  = pd.to_numeric(df['DC_POWER_INPUT'],  errors='coerce')
    df['AC_POWER_FIXED']  = df['AC_POWER_OUTPUT'] * 10
    df['EFFICIENCY']      = df['AC_POWER_FIXED'] / df['DC_POWER_INPUT']
    df['EFFICIENCY_%']    = df['EFFICIENCY'] * 100
    df['Value']           = df['AC_POWER_FIXED']

    df = df.loc[:, ~df.columns.duplicated()]
    df = df.rename(columns={'DATE_TIME': 'TIME_STAMP'})
    df = df[['TIME_STAMP', 'SOURCE_ID', 'SOURCE_ID_NUMBER',
             'Value', 'EFFICIENCY_%', 'AC_POWER_FIXED']]
    df = df.sort_values("TIME_STAMP").reset_index(drop=True)
    df["time_index"] = df.index
    return df

# ---------- Load & Clean Data ----------
def load_clean_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    st.write("📄 Uploaded Columns:", df.columns.tolist())

    inv = preprocess_inverter_data(df)
    if inv is not None:
        st.success("☀️ Detected inverter data. Preprocessed automatically.")
        return inv

    # generic timestamp detection
    timestamp_col = None
    for col in df.columns:
        if any(k in col.lower() for k in ("date","time","year")):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if df[col].notna().sum() > 0:
                    timestamp_col = col
                    break
            except:
                continue

    if not timestamp_col:
        st.error("❌ Could not detect a timestamp column. Please include a date-like column.")
        return pd.DataFrame()

    df = df.dropna(subset=[timestamp_col])
    df = df.rename(columns={timestamp_col: "TIME_STAMP"})
    df["TIME_STAMP"] = pd.to_datetime(df["TIME_STAMP"])

    # numeric value detection
    value_col = None
    for col in df.columns:
        if col.lower() in ("value","output","consumption","energy","acpower_kw"):
            value_col = col
            break
    if value_col is None:
        nums = df.select_dtypes(include="number").columns
        if len(nums) > 0:
            value_col = nums[0]
    if value_col is None:
        st.error("❌ Could not detect a numeric 'Value' column.")
        return pd.DataFrame()

    df = df.dropna(subset=["TIME_STAMP", value_col])
    df["Value"] = pd.to_numeric(df[value_col], errors='coerce')
    df = df.dropna(subset=["Value"])
    df = df.sort_values("TIME_STAMP").reset_index(drop=True)
    df["time_index"] = df.index
    return df

# ---------- Anomaly Detection ----------
def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    model = IsolationForest(contamination=0.01, random_state=42)
    df["anomaly"] = model.fit_predict(df[["Value"]]) == -1
    return df

# ---------- Forecasting ----------
def run_forecast(df: pd.DataFrame):
    df = df.copy()
    if "TIME_STAMP" in df.columns:
        df["TIME_STAMP"] = pd.to_datetime(df["TIME_STAMP"], errors='coerce')
        df = df.dropna(subset=["TIME_STAMP"])

    y_col = "AC_POWER_FIXED" if "AC_POWER_FIXED" in df.columns else "Value"
    if y_col not in df.columns:
        st.error(f"❌ Could not find '{y_col}' for forecasting.")
        return pd.DataFrame(), None

    X = df[["time_index"]]
    y = df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    forecast_df = pd.DataFrame({
        "time_index": X_test["time_index"],
        "Actual":     y_test.values,
        "Predicted":  y_pred
    }).sort_index()

    mse = mean_squared_error(y_test, y_pred)
    return forecast_df, mse

# ---------- AI Summary ----------
def generate_summary(df: pd.DataFrame) -> str:
    total     = df["Value"].sum()
    mean      = df["Value"].mean()
    anomalies = int(df.get("anomaly", pd.Series()).sum())
    return (
        f"- Total: **{total:.2f}**  \n"
        f"- Average: **{mean:.2f}**  \n"
        f"- Anomalies: **{anomalies}**"
    )

# ---------- Export Alerts ----------
def export_alerts(df: pd.DataFrame):
    today = pd.Timestamp(date.today()).date()
    df["DateOnly"] = pd.to_datetime(df["TIME_STAMP"]).dt.date
    alerts = df[df.get("anomaly", False) & (df["DateOnly"] == today)]
    if not alerts.empty:
        path = "alerts_today.csv"
        alerts[["TIME_STAMP","Value"]].to_csv(path, index=False)
        st.success(f"✅ Exported {len(alerts)} alert(s) to {path}")
    else:
        st.info("No new anomalies today.")

# ---------- Groupings & Summary ----------
def grouping_data_with_summary(df: pd.DataFrame):
    df_eff = df[df["EFFICIENCY_%"].between(0.01,100)].copy()

    res = (
        df_eff
        .groupby("SOURCE_ID")
        ["AC_POWER_FIXED"]
        .mean()
        .reset_index(name="avg_output")
    )

    q25, q50, q75 = res["avg_output"].quantile([0.25,0.5,0.75])

    high     = res[ res["avg_output"] >  q75 ]
    med_high = res[(res["avg_output"] >  q50) & (res["avg_output"] <= q75)]
    med_low  = res[(res["avg_output"] >  q25) & (res["avg_output"] <= q50)]
    low      = res[ res["avg_output"] <= q25 ]

    summary = {
        "High (4th Quartile)":     len(high),
        "Med-High (3rd Quartile)": len(med_high),
        "Med-Low (2nd Quartile)":  len(med_low),
        "Low (1st Quartile)":      len(low)
    }
    if len(low)>0:
        summary["⚠️ Low-performers need review"] = len(low)

    best  = res.loc[res["avg_output"].idxmax()]
    worst = res.loc[res["avg_output"].idxmin()]
    summary["Top Inverter"]    = {"id": best["SOURCE_ID"],  "avg_output": best["avg_output"]}
    summary["Bottom Inverter"] = {"id": worst["SOURCE_ID"], "avg_output": worst["avg_output"]}

    return high, med_high, med_low, low, summary

# ---------- Plotly Efficiency Anomaly ----------
def anomaly_detect(df: pd.DataFrame) -> go.Figure:
    df_clean = df[df['EFFICIENCY_%'] > 0].copy()
    df_clean['TIME_STAMP'] = pd.to_datetime(df_clean['TIME_STAMP'])
    df_clean = df_clean.sort_values(['SOURCE_ID', 'TIME_STAMP'])

    full_range = pd.date_range(
        start=df_clean['TIME_STAMP'].min(),
        end=df_clean['TIME_STAMP'].max(),
        freq='15T'
    )
    all_data = []
    inverter_list = sorted(df_clean['SOURCE_ID'].unique())

    for inv in inverter_list:
        inv_df = df_clean[df_clean['SOURCE_ID'] == inv].copy()
        inv_df = inv_df.set_index('TIME_STAMP').reindex(full_range)
        inv_df['SOURCE_ID'] = inv
        inv_df = inv_df.rename_axis('TIME_STAMP').reset_index()

        inv_df['anomaly'] = False
        mask = inv_df['EFFICIENCY_%'] > 0
        if mask.sum() > 10:
            model = IsolationForest(contamination=0.01, random_state=42)
            inv_df.loc[mask, 'anomaly'] = model.fit_predict(inv_df.loc[mask, ['EFFICIENCY_%']]) == -1

        all_data.append(inv_df)

    final_df = pd.concat(all_data)
    final_df['Status'] = final_df['anomaly'].map({True: 'Anomaly', False: 'Normal'})

    fig = go.Figure()
    dropdowns = []
    for i, inv in enumerate(inverter_list):
        temp = final_df[final_df['SOURCE_ID'] == inv]
        fig.add_trace(go.Scatter(
            x=temp['TIME_STAMP'],
            y=temp['EFFICIENCY_%'],
            mode='lines', name=f'{inv} Efficiency', visible=(i == 0)
        ))
        fig.add_trace(go.Scatter(
            x=temp[temp['anomaly']]['TIME_STAMP'],
            y=temp[temp['anomaly']]['EFFICIENCY_%'],
            mode='markers', name=f'{inv} Anomaly', visible=(i == 0), marker=dict(color='red', size=6)
        ))
        vis = [False] * (2 * len(inverter_list))
        vis[2*i] = True
        vis[2*i+1] = True
        dropdowns.append({
            'label': inv,
            'method': 'update',
            'args': [{'visible': vis}, {'title': f'Efficiency for {inv}'}]
        })

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=dropdowns, x=1.05, xanchor='left', y=1.15, yanchor='top')],
        title=f'Efficiency for {inverter_list[0]}',
        xaxis_title='Time',
        yaxis_title='Efficiency (%)',
        height=600,
        template='plotly_white'
    )
    return fig

# ---------- Main Localized App ----------
def show_localized():
    st.set_page_config(page_title="EnergyAI Localized", layout="wide")
    st.sidebar.markdown("### ⚡ EnergyAI Localized")
    st.sidebar.info("Inverter & efficiency dashboard—upload your CSV to begin.")
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")

    tab = st.sidebar.radio("Navigation", [
        "Home", "Energy & Summary", "Anomalies & Groupings", "Forecasting"
    ])

    if tab == "Home":
        st.title("⚡ EnergyAI Localized")
        st.markdown("Welcome to the localized workflows for inverter and efficiency analysis.")
        return

    if not uploaded:
        st.info("Please upload a CSV file to continue.")
        return

    df = load_clean_data(uploaded)
    if df.empty:
        return

    if tab == "Energy & Summary":
        st.header("Energy & Summary")
        df2 = detect_anomalies(df)
        st.line_chart(df2.set_index("TIME_STAMP")["Value"])
        st.markdown(generate_summary(df2))

    elif tab == "Anomalies & Groupings":
        st.header("Anomalies & Groupings")
        df2 = detect_anomalies(df)
        # updated graphing logic
        fig = anomaly_detect(df2)
        st.plotly_chart(fig, use_container_width=True)

        anomalies = df2[df2["anomaly"]]
        if not anomalies.empty:
            st.subheader("Detected Anomalies")
            st.dataframe(anomalies[["TIME_STAMP", "Value"]])

        high, med_high, med_low, low, summary = grouping_data_with_summary(df2)
        st.subheader("🔋 High-Efficiency Inverters")
        st.dataframe(high)
        st.subheader("⚡ Medium-High Efficiency")
        st.dataframe(med_high)
        st.subheader("🔆 Medium-Low Efficiency")
        st.dataframe(med_low)
        st.subheader("🪫 Low-Efficiency Inverters")
        st.dataframe(low)
        st.markdown("### 📊 Inverter Performance Summary")
        st.json(summary)

        if st.button("Export Anomaly Alerts"):
            export_alerts(df2)

    else:  # Forecasting
        st.header("Forecasting")
        df2 = detect_anomalies(df)
        fc, mse = run_forecast(df2)

        st.write(f"Mean Squared Error: {mse:.2f}")
        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.plot(fc["time_index"], fc["Actual"],    label="Actual",    alpha=0.6)
        ax2.plot(fc["time_index"], fc["Predicted"], label="Predicted", alpha=0.8)
        ax2.set_xlabel("Time Index")
        ax2.set_ylabel("Value")
        ax2.legend()
        st.pyplot(fig2)
