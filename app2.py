# ADD THIS FUNCTION after problem_brief()
def calculate_der_losses(df):
    """Real DER optimization loss calculations"""
    losses = {}
    
    # Renewable curtailment loss
    if all(col in df for col in ["solar_MW", "wind_MW", "load_demand_MW"]):
        total_renew = (df["solar_MW"] + df["wind_MW"]).sum()
        avg_demand = df["load_demand_MW"].mean()
        losses["renew_curtail_pct"] = max(0, total_renew - avg_demand * len(df)) / total_renew * 100
    
    # BESS efficiency loss
    if "bess_soc_pct" in df:
        losses["bess_ineff_pct"] = 100 - df["bess_soc_pct"].mean()
    
    # Constraint violation cost ($)
    if "constraint_violation" in df:
        losses["violation_cost"] = df["constraint_violation"].sum() * 1000  # $1k per violation
    
    return losses

# REPLACE "Simulate ML Training" button with:
if viable:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Calculate DER Losses"):
            losses = calculate_der_losses(df)
            st.subheader("💰 DER Optimization Losses")
            for metric, value in losses.items():
                st.metric(metric.replace("_", " ").title(), f"{value:.1f}")
    
    with col2:
        if st.button("🤖 Train LSTM Model"):
            st.info("Full ML training requires TensorFlow - use Colab Pro")
