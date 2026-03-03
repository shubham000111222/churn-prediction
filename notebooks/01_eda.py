# %% [markdown]
# # 01 — Exploratory Data Analysis: Customer Churn
# **Goal**: Understand the dataset structure, distributions, and early signals of churn.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

df = pd.read_csv("../../data/customers.csv")
print(f"Shape: {df.shape}")
df.head()

# %%
# Dataset overview
print(df.info())
print("\nChurn distribution:")
print(df["churn"].value_counts(normalize=True).round(3))

# %%
# Numerical distributions
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
num_cols = ["tenure_months", "monthly_charges", "total_charges",
            "num_support_tickets", "days_since_last_login", "payment_failures"]

for ax, col in zip(axes.flatten(), num_cols):
    df[col].hist(ax=ax, bins=40, color="#6366f1", edgecolor="white", alpha=0.85)
    ax.set_title(col.replace("_", " ").title())
plt.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../../reports/eda_distributions.png", dpi=150)
plt.show()

# %%
# Churn rate by contract type
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, col in zip(axes, ["contract_type", "internet_service", "payment_method"]):
    churn_rate = df.groupby(col)["churn"].mean().sort_values(ascending=False)
    churn_rate.plot(kind="bar", ax=ax, color="#6366f1", alpha=0.8, edgecolor="white")
    ax.set_title(f"Churn Rate by {col.replace('_', ' ').title()}")
    ax.set_ylabel("Churn Rate")
    ax.tick_params(axis="x", rotation=25)
plt.suptitle("Churn Rate by Categorical Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../../reports/eda_churn_by_category.png", dpi=150)
plt.show()

# %%
# Correlation heatmap (numerical only)
num_df = df.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
sns.heatmap(num_df.corr(), mask=mask, cmap="coolwarm", annot=True,
            fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../../reports/eda_correlation.png", dpi=150)
plt.show()

# %%
# Tenure vs Monthly Charges coloured by churn
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    df["tenure_months"], df["monthly_charges"],
    c=df["churn"], cmap="coolwarm", alpha=0.4, s=8
)
plt.colorbar(scatter, ax=ax, label="Churn")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Monthly Charges ($)")
ax.set_title("Tenure vs Monthly Charges — Coloured by Churn")
plt.savefig("../../reports/eda_scatter_churn.png", dpi=150)
plt.show()

print("✅ EDA complete. Reports saved to reports/")
