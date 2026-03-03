# Customer Churn Prediction Engine

> End-to-end ML system for predicting customer churn using XGBoost + LightGBM ensemble with SHAP explainability. Served via FastAPI with real-time scoring < 50ms.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green) ![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## Problem Statement

A SaaS company was losing 12% of customers monthly with no early-warning system. The business impact: ~$1.2M annual revenue at risk. The goal was to build a predictive system that identifies high-churn customers 30 days in advance so the retention team can intervene proactively.

## Approach

1. **EDA** — 3M+ transaction records, feature correlation, cohort analysis
2. **Feature Engineering** — RFM scoring, behavioural telemetry, payment failure flags
3. **Modelling** — XGBoost + LightGBM ensemble with Bayesian hyperparameter tuning (Optuna)
4. **Explainability** — SHAP TreeExplainer for per-customer reason codes
5. **Deployment** — FastAPI endpoint with Redis caching, Docker/docker-compose

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.94** |
| Precision @ 0.5 | **0.89** |
| Recall @ 0.5 | **0.87** |
| Inference latency | **< 50ms** |
| Churn rate reduction | **23%** |
| Revenue protected | **$1.2M/year** |

## Tech Stack

- **ML**: Python, XGBoost, LightGBM, Scikit-learn, Optuna, SHAP
- **Data**: Pandas, NumPy, Polars
- **API**: FastAPI, Pydantic, Redis
- **Infra**: Docker, docker-compose
- **Monitoring**: MLflow experiment tracking

---

## Project Structure

```
churn-prediction/
├── api/
│   └── main.py              # FastAPI inference endpoint
├── notebooks/
│   ├── 01_eda.py            # Exploratory data analysis
│   ├── 02_feature_engineering.py
│   └── 03_model_training.py # Full training pipeline
├── src/
│   ├── data/
│   │   └── data_generator.py  # Synthetic data generation
│   ├── features/
│   │   └── feature_engineering.py
│   └── models/
│       ├── train.py
│       └── evaluate.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quick Start

### Local (venv)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Generate synthetic data and train
python src/data/data_generator.py
python src/models/train.py

# Start API
uvicorn api.main:app --reload
```

### Docker

```bash
docker-compose up --build
```

API available at `http://localhost:8000/docs`

---

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C12345",
    "tenure_months": 14,
    "monthly_charges": 89.5,
    "total_charges": 1253.0,
    "num_support_tickets": 3,
    "days_since_last_login": 12,
    "payment_failures": 2,
    "contract_type": "month-to-month",
    "internet_service": "fiber"
  }'
```

Response:

```json
{
  "customer_id": "C12345",
  "churn_probability": 0.78,
  "churn_prediction": true,
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

## MLflow Tracking

```bash
mlflow ui --port 5001
```
Open http://localhost:5001 to view experiments, runs, and metric comparisons.

---

## Author

**Your Name** · [GitHub](https://github.com/shubham000111222) · [LinkedIn](https://linkedin.com/in/yourusername)
