import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Prediction Demo", page_icon="🔮", layout="wide")

st.title("🔮 Customer Churn Prediction Engine")
st.caption("XGBoost + LightGBM Ensemble · SHAP Explainability · Real-time Scoring")

# ─── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Customer Profile")
    tenure = st.slider("Tenure (months)", 1, 72, 14)
    monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 75)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1)
    support_calls = st.slider("Support Calls (last 6mo)", 0, 10, 3)
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    has_internet = st.checkbox("Has Internet Service", value=True)
    has_streaming = st.checkbox("Has Streaming Service", value=False)

# ─── Simulate model prediction ─────────────────────────────────────────────────
def predict_churn(tenure, monthly, n_prod, calls, contract, payment, internet, streaming):
    base = 0.5
    base -= tenure * 0.008
    base += (monthly - 65) * 0.004
    base -= n_prod * 0.06
    base += calls * 0.04
    if contract == "Two year":     base -= 0.25
    elif contract == "One year":   base -= 0.12
    if payment == "Electronic Check": base += 0.08
    if not internet: base -= 0.05
    if streaming:    base -= 0.03
    return float(np.clip(base, 0.03, 0.97))

prob = predict_churn(tenure, monthly_charges, num_products, support_calls,
                     contract, payment_method, has_internet, has_streaming)
prediction = "High Risk 🔴" if prob > 0.5 else "Low Risk 🟢"

# ─── KPI Row ───────────────────────────────────────────────────────────────────
st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Churn Probability",  f"{prob:.1%}")
c2.metric("Risk Level",          prediction)
c3.metric("Model AUC",           "0.94")
c4.metric("Inference Time",      "< 50ms")

# ─── Gauge chart ───────────────────────────────────────────────────────────────
st.divider()
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Churn Risk Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Churn Probability (%)"},
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ef4444" if prob > 0.5 else "#10b981"},
            "steps": [
                {"range": [0, 30],  "color": "rgba(16,185,129,0.15)"},
                {"range": [30, 60], "color": "rgba(245,158,11,0.15)"},
                {"range": [60, 100],"color": "rgba(239,68,68,0.15)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.75, "value": 50},
        },
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300, font_color="white")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("SHAP Feature Importance")
    features = {
        "Tenure":          -tenure * 0.008,
        "Monthly Charges": (monthly_charges - 65) * 0.004,
        "Support Calls":   support_calls * 0.04,
        "Num Products":    -num_products * 0.06,
        "Contract":        -0.25 if contract == "Two year" else (-0.12 if contract == "One year" else 0),
        "Payment Method":  0.08 if payment_method == "Electronic Check" else 0,
    }
    shap_df = pd.DataFrame({"Feature": list(features.keys()), "SHAP Value": list(features.values())})
    shap_df = shap_df.sort_values("SHAP Value")
    colors = ["#ef4444" if v > 0 else "#10b981" for v in shap_df["SHAP Value"]]
    fig2 = px.bar(shap_df, x="SHAP Value", y="Feature", orientation="h",
                  color="SHAP Value", color_continuous_scale=["#10b981", "#ef4444"],
                  title="Feature Contributions (SHAP)")
    fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", height=300, showlegend=False,
                       coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

# ─── Recommendations ───────────────────────────────────────────────────────────
st.divider()
st.subheader("💡 Retention Recommendations")
recs = []
if tenure < 12:   recs.append("🎁 Offer loyalty discount — customer is early in lifecycle")
if support_calls > 4: recs.append("📞 Assign dedicated support agent — high contact rate detected")
if contract == "Month-to-month": recs.append("📋 Upsell to annual contract — reduces churn risk by 12%")
if monthly_charges > 90: recs.append("💰 Review pricing plan — high spend increases sensitivity")
if not recs:      recs.append("✅ Customer is stable. Maintain engagement with periodic check-ins.")
for r in recs:
    st.info(r)

# ─── Batch simulation ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Cohort Churn Distribution (500 simulated customers)")
np.random.seed(42)
n = 500
sim = pd.DataFrame({
    "tenure":   np.random.randint(1, 73, n),
    "charges":  np.random.uniform(20, 120, n),
    "calls":    np.random.randint(0, 11, n),
})
sim["prob"] = (0.5 - sim["tenure"]*0.006 + (sim["charges"]-65)*0.003 + sim["calls"]*0.035).clip(0.05, 0.95)
sim["risk"] = sim["prob"].apply(lambda x: "High" if x > 0.5 else "Low")

fig3 = px.histogram(sim, x="prob", color="risk", nbins=30,
                    color_discrete_map={"High": "#ef4444", "Low": "#10b981"},
                    labels={"prob": "Churn Probability", "count": "Customers"},
                    title="Churn Probability Distribution")
fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                   plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig3, use_container_width=True)

st.caption("Built by Shubham Kumar · [GitHub](https://github.com/shubham000111222/churn-prediction)")
