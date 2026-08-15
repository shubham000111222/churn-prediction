# SaaS Customer Churn Prediction Engine

**[Live Demo](https://churn-prediction-grph4xyczphtcaaqfwdh3d.streamlit.app/)**
> XGBoost + LightGBM ensemble with SHAP explainability, served via FastAPI in <50ms.  
> Built on a synthetic dataset modelling SaaS telemetry — end-to-end from EDA to deployment.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## Problem

Customer churn is one of the highest-leverage problems in SaaS. This project simulates a
retention team's ML workflow: given 30 days of behavioural signals, identify which customers
are likely to churn — and explain *why* — so retention teams can intervene proactively.

> **Note:** This project uses a synthetic dataset generated to mirror real-world SaaS 
> telemetry (RFM scores, payment failures, login activity). All metrics reflect 
> model performance on held-out synthetic data.

---

## Model Performance (Held-out Test Set)

| Metric              | Value     |
|---------------------|-----------|
| ROC-AUC             | **0.94**  |
| Precision @ 0.5     | **0.89**  |
| Recall @ 0.5        | **0.87**  |
| Inference latency   | **<50ms** |

---

## What I Built

- **Feature engineering** — RFM scoring, behavioural telemetry, payment failure flags 
  from 3M+ synthetic records
- **Ensemble model** — XGBoost + LightGBM with Bayesian hyperparameter tuning via Optuna
- **Explainability** — SHAP TreeExplainer returns per-customer reason codes with each prediction
- **Production API** — FastAPI endpoint with Redis caching, fully containerised with Docker
- **Experiment tracking** — MLflow for model versioning and metric comparison

---

## Demo

[Add a GIF or screenshot of the Streamlit dashboard here]

```bash
docker-compose up --build
# API docs → http://localhost:8000/docs
# Dashboard → http://localhost:8501
```

---

## Sample API Response

```json
{
  "customer_id": "C12345",
  "churn_probability": 0.78,
  "risk_tier": "HIGH",
  "top_reasons": [
    {"feature": "payment_failures", "impact": 0.23},
    {"feature": "days_since_last_login", "impact": 0.18},
    {"feature": "tenure_months", "impact": -0.12}
  ],
  "latency_ms": 38
}
```

---

## Tech Stack

`Python` · `XGBoost` · `LightGBM` · `SHAP` · `Optuna` · `FastAPI` · `Redis` · `Docker` · `MLflow`

---

## Author

**Shubham Kumar** · NIT Delhi, CSE (3rd Year)  
[GitHub](https://github.com/shubham000111222) · [Portfolio](https://data-science-portfolio-three-olive.vercel.app)
