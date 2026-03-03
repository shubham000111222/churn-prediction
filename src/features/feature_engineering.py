import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_engineer(path: str = "data/customers.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # RFM-style features
    df["revenue_per_month"] = df["total_charges"] / (df["tenure_months"] + 1)
    df["avg_tickets_per_month"] = df["num_support_tickets"] / (df["tenure_months"] + 1)
    df["charge_to_tenure_ratio"] = df["monthly_charges"] / (df["tenure_months"] + 1)
    df["payment_failure_rate"] = df["payment_failures"] / (df["tenure_months"] + 1)
    df["is_high_spender"] = (df["monthly_charges"] > df["monthly_charges"].quantile(0.75)).astype(int)
    df["is_long_tenure"] = (df["tenure_months"] > 24).astype(int)
    df["is_new_customer"] = (df["tenure_months"] <= 3).astype(int)
    df["engagement_score"] = (
        df["has_phone_service"] + df["has_streaming"] + (df["num_products"] > 2).astype(int)
    )

    # Encode categoricals
    le = LabelEncoder()
    for col in ["contract_type", "internet_service", "payment_method"]:
        df[f"{col}_enc"] = le.fit_transform(df[col])

    return df


FEATURE_COLS = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_support_tickets", "days_since_last_login", "payment_failures",
    "num_products", "has_phone_service", "has_streaming", "is_senior_citizen",
    "revenue_per_month", "avg_tickets_per_month", "charge_to_tenure_ratio",
    "payment_failure_rate", "is_high_spender", "is_long_tenure", "is_new_customer",
    "engagement_score", "contract_type_enc", "internet_service_enc", "payment_method_enc",
]
TARGET_COL = "churn"
