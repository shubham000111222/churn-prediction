"""
Full training pipeline: XGBoost + LightGBM ensemble with Optuna tuning.
Run: python src/models/train.py
"""
import json
import pickle
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.features.feature_engineering import load_and_engineer, FEATURE_COLS, TARGET_COL

optuna.logging.set_verbosity(optuna.logging.WARNING)
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def objective_xgb(trial, X_tr, y_tr, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5),
        "use_label_encoder": False,
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    preds = clf.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


def objective_lgb(trial, X_tr, y_tr, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        "is_unbalance": True,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    preds = clf.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, preds)


def train():
    print("Loading and engineering features...")
    df = load_and_engineer("data/customers.csv")
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)

    mlflow.set_experiment("churn_prediction")

    with mlflow.start_run(run_name="xgb_lgb_ensemble"):
        # ── XGBoost ──────────────────────────────────────────────────────────
        print("Tuning XGBoost (20 trials)...")
        study_xgb = optuna.create_study(direction="maximize")
        study_xgb.optimize(lambda t: objective_xgb(t, X_tr, y_tr, X_val, y_val), n_trials=20)

        best_xgb = xgb.XGBClassifier(**study_xgb.best_params, use_label_encoder=False,
                                       eval_metric="auc", random_state=42, n_jobs=-1)
        best_xgb.fit(X_train, y_train, verbose=False)

        # ── LightGBM ──────────────────────────────────────────────────────────
        print("Tuning LightGBM (20 trials)...")
        study_lgb = optuna.create_study(direction="maximize")
        study_lgb.optimize(lambda t: objective_lgb(t, X_tr, y_tr, X_val, y_val), n_trials=20)

        best_lgb_params = {k: v for k, v in study_lgb.best_params.items()}
        best_lgb_params.update({"is_unbalance": True, "random_state": 42, "n_jobs": -1, "verbose": -1})
        best_lgb = lgb.LGBMClassifier(**best_lgb_params)
        best_lgb.fit(X_train, y_train)

        # ── Ensemble (average probabilities) ──────────────────────────────────
        xgb_proba = best_xgb.predict_proba(X_test)[:, 1]
        lgb_proba = best_lgb.predict_proba(X_test)[:, 1]
        ensemble_proba = 0.5 * xgb_proba + 0.5 * lgb_proba

        auc = roc_auc_score(y_test, ensemble_proba)
        preds = (ensemble_proba >= 0.5).astype(int)
        report = classification_report(y_test, preds, output_dict=True)

        print(f"\n✅ Ensemble AUC: {auc:.4f}")
        print(classification_report(y_test, preds))

        mlflow.log_metric("auc", auc)
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall", report["1"]["recall"])
        mlflow.log_metric("f1", report["1"]["f1-score"])

        # ── SHAP explainability ───────────────────────────────────────────────
        print("Computing SHAP values...")
        explainer = shap.TreeExplainer(best_xgb)
        shap_vals = explainer.shap_values(X_test[:500])
        mean_abs_shap = dict(zip(FEATURE_COLS, np.abs(shap_vals).mean(axis=0)))
        top_features = dict(sorted(mean_abs_shap.items(), key=lambda x: x[1], reverse=True)[:10])
        print("Top 10 SHAP features:", top_features)

        # ── Save artefacts ────────────────────────────────────────────────────
        with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
            pickle.dump(best_xgb, f)
        with open(MODELS_DIR / "lgb_model.pkl", "wb") as f:
            pickle.dump(best_lgb, f)
        with open(MODELS_DIR / "shap_feature_importance.json", "w") as f:
            json.dump(top_features, f, indent=2)
        with open(MODELS_DIR / "feature_cols.json", "w") as f:
            json.dump(FEATURE_COLS, f)

        mlflow.log_artifacts(str(MODELS_DIR))
        print(f"\nModels saved to {MODELS_DIR}/")


if __name__ == "__main__":
    train()
