import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
N = 50_000

contract_types = ["month-to-month", "one-year", "two-year"]
internet_types = ["fiber", "dsl", "no"]
payment_methods = ["credit_card", "bank_transfer", "electronic_check", "mailed_check"]

data = {
    "customer_id": [f"C{i:06d}" for i in range(N)],
    "tenure_months": np.random.randint(1, 73, N),
    "monthly_charges": np.round(np.random.uniform(20, 120, N), 2),
    "num_support_tickets": np.random.poisson(1.5, N),
    "days_since_last_login": np.random.randint(0, 90, N),
    "payment_failures": np.random.poisson(0.8, N),
    "contract_type": np.random.choice(contract_types, N, p=[0.55, 0.25, 0.20]),
    "internet_service": np.random.choice(internet_types, N, p=[0.45, 0.40, 0.15]),
    "payment_method": np.random.choice(payment_methods, N),
    "num_products": np.random.randint(1, 6, N),
    "has_phone_service": np.random.choice([0, 1], N, p=[0.1, 0.9]),
    "has_streaming": np.random.choice([0, 1], N, p=[0.5, 0.5]),
    "is_senior_citizen": np.random.choice([0, 1], N, p=[0.84, 0.16]),
}

df = pd.DataFrame(data)
df["total_charges"] = np.round(df["tenure_months"] * df["monthly_charges"] * np.random.uniform(0.9, 1.1, N), 2)

# Engineer churn probability based on realistic signals
churn_score = (
    0.30 * (df["contract_type"] == "month-to-month").astype(int)
    + 0.20 * np.clip(df["payment_failures"] / 5, 0, 1)
    + 0.15 * np.clip(df["days_since_last_login"] / 90, 0, 1)
    + 0.15 * np.clip(df["num_support_tickets"] / 10, 0, 1)
    - 0.15 * np.clip(df["tenure_months"] / 72, 0, 1)
    - 0.10 * np.clip(df["num_products"] / 5, 0, 1)
    + np.random.normal(0, 0.08, N)
)
churn_prob = 1 / (1 + np.exp(-5 * (churn_score - 0.3)))
df["churn"] = (np.random.uniform(0, 1, N) < churn_prob).astype(int)

out_path = Path(__file__).parent.parent.parent / "data"
out_path.mkdir(exist_ok=True)
df.to_csv(out_path / "customers.csv", index=False)

print(f"Generated {N} records — churn rate: {df['churn'].mean():.1%}")
print(f"Saved to {out_path / 'customers.csv'}")
