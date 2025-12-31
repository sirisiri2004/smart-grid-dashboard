import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Smart Grid ML Dashboard", layout="wide")

st.title("⚡ Smart Grid ML Dashboard")
st.markdown("Upload a DER-based Smart Grid CSV to validate, diagnose, and train the ML model.")

# CONFIGURATION
SEQ_LEN = 24
MIN_ROWS_REQUIRED = 500

FEATURES = [
    "load_demand_MW", "solar_MW", "wind_MW", "hydro_MW",
    "bess_soc_pct", "grid_import_MW", "grid_export_MW"
]

REQUIRED_COLUMNS = FEATURES + [
    "voltage_pu", "frequency_Hz", "generation_cost_$",
    "constraint_violation", "optimal_dispatch_flag"
]

def check_viability(df):
    issues = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
    if df.isnull().sum().sum() > 0:
        issues.append("Dataset contains missing values")
    if len(df) < MIN_ROWS_REQUIRED:
        issues.append("Dataset too small for temporal ML")
    if "voltage_pu" in df.columns:
        if df["voltage_pu"].min() < 0.85 or df["voltage_pu"].max() > 1.15:
            issues.append("Voltage outside grid-safe range")
    if "frequency_Hz" in df.columns:
        if df["frequency_Hz"].min() < 49 or df["frequency_Hz"].max() > 51:
            issues.append("Frequency instability detected")
    return len(issues) == 0, issues or ["Dataset viable for smart-grid ML"]

def problem_brief(df):
    brief = []
    if "constraint_violation" in df.columns and df["constraint_violation"].mean() > 0.05:
        brief.append("High grid constraint violation frequency")
    if "bess_soc_pct" in df.columns and df["bess_soc_pct"].mean() > 90:
        brief.append("BESS frequently saturated")
    if all(col in df.columns for col in ["solar_MW", "wind_MW", "hydro_MW", "load_demand_MW"]):
        renewable_ratio = (df["solar_MW"] + df["wind_MW"] + df["hydro_MW"]).mean() / df["load_demand_MW"].mean()
        if renewable_ratio < 0.3:
            brief.append("Low renewable penetration")
        if renewable_ratio > 0.85:
            brief.append("Risk of renewable over-curtailment")
    return brief or ["No major operational issues detected"]

uploaded_file = st.file_uploader("Upload Smart Grid CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully!")
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    
    viable, messages = check_viability(df)
    st.subheader("✅ Dataset Viability Check")
    if viable:
        st.success("Dataset is viable for Smart Grid ML")
    else:
        st.error("Dataset NOT viable for Smart Grid ML")
    for m in messages:
        st.write("•", m)
    
    st.subheader("🧠 Automated Problem Diagnosis")
    for issue in problem_brief(df):
        st.warning(issue)
    
    if viable:
        if st.button("🚀 Simulate ML Training"):
            with st.spinner("Training model..."):
                st.success("✅ Model training completed!")
                st.balloons()
else:
    st.info("Please upload a Smart Grid CSV dataset to begin.")
