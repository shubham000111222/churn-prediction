"""
FastAPI inference endpoint for churn prediction.
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time churn scoring with SHAP-based reason codes",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path("models")

# ── Load models at startup ────────────────────────────────────────────────────
xgb_model = None
lgb_model = None
feature_cols = None
explainer = None


@app.on_event("startup")
async def load_models():
    global xgb_model, lgb_model, feature_cols, explainer
    try:
        with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
            xgb_model = pickle.load(f)
        with open(MODELS_DIR / "lgb_model.pkl", "rb") as f:
            lgb_model = pickle.load(f)
        with open(MODELS_DIR / "feature_cols.json") as f:
            feature_cols = json.load(f)
        explainer = shap.TreeExplainer(xgb_model)
        print("✅ Models loaded successfully")
    except FileNotFoundError:
        print("⚠️  Model files not found. Run src/models/train.py first.")


# ── Request / Response schemas ────────────────────────────────────────────────
class CustomerFeatures(BaseModel):
    customer_id: str
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0, le=500)
    total_charges: float = Field(..., ge=0)
    num_support_tickets: int = Field(0, ge=0)
    days_since_last_login: int = Field(0, ge=0, le=365)
    payment_failures: int = Field(0, ge=0)
    num_products: int = Field(1, ge=1, le=10)
    has_phone_service: int = Field(1, ge=0, le=1)
    has_streaming: int = Field(0, ge=0, le=1)
    is_senior_citizen: int = Field(0, ge=0, le=1)
    contract_type: str = Field("month-to-month")  # month-to-month | one-year | two-year
    internet_service: str = Field("fiber")         # fiber | dsl | no
    payment_method: str = Field("credit_card")     # credit_card | bank_transfer | electronic_check | mailed_check


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_tier: str
    top_reasons: list
    latency_ms: float


def engineer_single(req: CustomerFeatures) -> np.ndarray:
    contract_enc = {"month-to-month": 1, "one-year": 0, "two-year": 2}.get(req.contract_type, 1)
    internet_enc = {"dsl": 0, "fiber": 1, "no": 2}.get(req.internet_service, 1)
    payment_enc = {"bank_transfer": 0, "credit_card": 1, "electronic_check": 2, "mailed_check": 3}.get(req.payment_method, 1)

    revenue_per_month = req.total_charges / (req.tenure_months + 1)
    avg_tickets_per_month = req.num_support_tickets / (req.tenure_months + 1)
    charge_to_tenure_ratio = req.monthly_charges / (req.tenure_months + 1)
    payment_failure_rate = req.payment_failures / (req.tenure_months + 1)
    is_high_spender = int(req.monthly_charges > 89)
    is_long_tenure = int(req.tenure_months > 24)
    is_new_customer = int(req.tenure_months <= 3)
    engagement_score = req.has_phone_service + req.has_streaming + int(req.num_products > 2)

    return np.array([[
        req.tenure_months, req.monthly_charges, req.total_charges,
        req.num_support_tickets, req.days_since_last_login, req.payment_failures,
        req.num_products, req.has_phone_service, req.has_streaming, req.is_senior_citizen,
        revenue_per_month, avg_tickets_per_month, charge_to_tenure_ratio,
        payment_failure_rate, is_high_spender, is_long_tenure, is_new_customer,
        engagement_score, contract_enc, internet_enc, payment_enc,
    ]])


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: CustomerFeatures):
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training first.")

    t0 = time.perf_counter()
    X = engineer_single(req)

    xgb_prob = float(xgb_model.predict_proba(X)[0, 1])
    lgb_prob = float(lgb_model.predict_proba(X)[0, 1])
    prob = round((xgb_prob + lgb_prob) / 2, 4)

    # SHAP reasons
    shap_vals = explainer.shap_values(X)[0]
    cols = feature_cols or [f"f{i}" for i in range(X.shape[1])]
    reasons = sorted(
        [{"feature": c, "impact": round(float(v), 4)} for c, v in zip(cols, shap_vals)],
        key=lambda x: abs(x["impact"]),
        reverse=True,
    )[:5]

    risk = "LOW" if prob < 0.3 else "MEDIUM" if prob < 0.6 else "HIGH"
    latency = round((time.perf_counter() - t0) * 1000, 2)

    return PredictionResponse(
        customer_id=req.customer_id,
        churn_probability=prob,
        churn_prediction=prob >= 0.5,
        risk_tier=risk,
        top_reasons=reasons,
        latency_ms=latency,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": xgb_model is not None}


@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs"}
