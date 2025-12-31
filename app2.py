import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Smart Grid ML Dashboard", layout="wide")

st.title("⚡ Smart Grid ML Dashboard")
st.markdown("Upload ANY CSV - Auto-detects power grid features")

# Universal column mapping
POWER_COLS = {
    'load': ['load_demand_MW', 'load_MW', 'demand_MW', 'power_demand', 'P_load'],
    'solar': ['solar_MW', 'solar_power', 'pv_MW', 'solar_generation'],
    'wind': ['wind_MW', 'wind_power', 'wind_generation'],
    'voltage': ['voltage_pu', 'voltage', 'V_pu', 'bus_voltage'],
    'frequency': ['frequency_Hz', 'freq', 'frequency']
}

def detect_power_columns(df):
    """Auto-detect power columns"""
    mapping = {}
    for key, cols in POWER_COLS.items():
        for col in cols:
            if col in df.columns:
                mapping[key] = col
                break
    return mapping

uploaded_file = st.file_uploader("📁 Upload ANY CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ CSV loaded! {len(df):,} rows × {len(df.columns)} columns")
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10))
    
    # Auto-detect features
    mapping = detect_power_columns(df)
    st.subheader("🔍 Auto-Detected Features")
    for feature, col in mapping.items():
        st.write(f"✅ {feature.title()}: `{col}`")
    
    if not mapping:
        st.warning("No standard power columns detected - Basic analysis only")
    
    # Universal viability check
    st.subheader("✅ Dataset Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing %", f"{df.isnull().sum().sum()/len(df)/len(df.columns)*100:.1f}%")
    
    # Smart Grid specific checks
    st.subheader("🧠 Smart Grid Diagnostics")
    issues = []
    
    # Voltage check
    if 'voltage' in mapping:
        v_col = mapping['voltage']
        v_min, v_max = df[v_col].min(), df[v_col].max()
        if v_min < 0.85 or v_max > 1.15:
            issues.append(f"⚠️ Voltage unstable: {v_min:.2f}-{v_max:.2f} pu")
    
    # Size check
    if len(df) < 100:
        issues.append("⚠️ Dataset too small for ML")
    
    if issues:
        for issue in issues:
            st.warning(issue)
    else:
        st.success("✅ Perfect for Smart Grid ML!")
    
    # Download cleaned data
    csv = df.to_csv(index=False)
    st.download_button("💾 Download Cleaned CSV", csv, "smart_grid_data.csv")
    
    st.balloons()

else:
    st.info("👆 Upload ANY CSV to analyze power grid data")
