import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# =========================================================
# CONFIGURATION
# =========================================================
SEQ_LEN = 24
MIN_ROWS_REQUIRED = 500

FEATURES = [
    "load_demand_MW", "solar_MW", "wind_MW", "hydro_MW",
    "bess_soc_pct", "grid_import_MW", "grid_export_MW"
]

TARGETS = [
    "load_demand_MW", "generation_cost_$",
    "constraint_violation", "optimal_dispatch_flag"
]

REQUIRED_COLUMNS = FEATURES + [
    "voltage_pu", "frequency_Hz", "generation_cost_$",
    "constraint_violation", "optimal_dispatch_flag"
]

# =========================================================
# FUNCTIONS
# =========================================================
def check_viability(df):
    issues = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
    if df.isnull().sum().sum() > 0:
        issues.append("Dataset contains missing values")
    if len(df) < MIN_ROWS_REQUIRED:
        issues.append("Dataset too small for temporal ML")
    if "voltage_pu" in df.columns and (df["voltage_pu"].min() < 0.85 or df["voltage_pu"].max() > 1.15):
        issues.append("Voltage values outside grid-safe range")
    if "frequency_Hz" in df.columns and (df["frequency_Hz"].min() < 49 or df["frequency_Hz"].max() > 51):
        issues.append("Frequency instability detected")
    if issues:
        return False, issues
    return True, ["Dataset viable for smart-grid ML"]

def problem_brief(df):
    brief = []
    if "constraint_violation" in df.columns:
        violation_rate = df["constraint_violation"].mean()
        if violation_rate > 0.05:
            brief.append("High grid constraint violation frequency")
    if "bess_soc_pct" in df.columns:
        avg_soc = df["bess_soc_pct"].mean()
        if avg_soc > 90:
            brief.append("BESS frequently saturated")
    if all(col in df.columns for col in ["solar_MW", "wind_MW", "hydro_MW", "load_demand_MW"]):
        renewable_ratio = (df["solar_MW"] + df["wind_MW"] + df["hydro_MW"]).mean() / df["load_demand_MW"].mean()
        if renewable_ratio < 0.3:
            brief.append("Low renewable penetration")
        if renewable_ratio > 0.85:
            brief.append("Risk of renewable over-curtailment")
    if not brief:
        brief.append("No major operational issues detected")
    return brief

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(32)(x)
    load_out = layers.Dense(1, name="load_forecast")(x)
    cost_out = layers.Dense(1, name="cost_forecast")(x)
    constraint_out = layers.Dense(1, activation="sigmoid", name="constraint_flag")(x)
    optimal_out = layers.Dense(1, activation="sigmoid", name="optimal_flag")(x)
    model = models.Model(inputs, [load_out, cost_out, constraint_out, optimal_out])
    model.compile(
        optimizer="adam",
        loss={"load_forecast": "mse", "cost_forecast": "mse", "constraint_flag": "binary_crossentropy", "optimal_flag": "binary_crossentropy"},
        metrics=["accuracy"]
    )
    return model

def train_smart_grid_model(df):
    os.makedirs('models', exist_ok=True)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    y = df[TARGETS].values
    X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LEN)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
    model = build_model((SEQ_LEN, len(FEATURES)))
    model.fit(
        X_train, [y_train[:, 0], y_train[:, 1], y_train[:, 2], y_train[:, 3]],
        epochs=5, batch_size=32, validation_split=0.1, verbose=0
    )
    model.save("models/smart_grid_unified_model.h5")
    return model

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Smart Grid ML Dashboard", layout="wide")

st.title("⚡ Smart Grid ML Dashboard")
st.markdown("Upload a DER-based Smart Grid CSV to validate, diagnose, and train the ML model.")

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
        if st.button("🚀 Train Smart Grid ML Model"):
            with st.spinner("Training model..."):
                train_smart_grid_model(df)
            st.success("Model trained and saved successfully!")
            
            if os.path.exists("models/smart_grid_unified_model.h5"):
                with open("models/smart_grid_unified_model.h5", "rb") as file:
                    st.download_button("💾 Download Trained Model", file.read(), "smart_grid_model.h5")

else:
    st.info("Please upload a Smart Grid CSV dataset to begin.")
