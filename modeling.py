# src/modeling.py
"""
Task 4 - Modeling pipeline
- Builds: Claim Severity (regression), Claim Probability (classification)
- Models: Linear Regression, RandomForest, XGBoost (if available)
- Evaluates and saves metrics, models, and SHAP summary plot for best model
- Inputs: data/processed/cleaned_data_sample.csv (preferred) or cleaned_data.csv
- Outputs: models/*.pkl, results/metrics.csv, results/shap_summary.png
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# try import xgboost
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    print("xgboost not available — XGBoost models will be skipped.")

# try import shap
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False
    print("shap not available — SHAP explanations will be skipped.")

# -----------------------
# Config
# -----------------------
PROJECT_ROOT = Path.cwd()
INPUT_PATHS = [
    PROJECT_ROOT / "data" / "processed" / "cleaned_data_sample.csv",
    PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_SIZE = None  # set to int for extra sampling; None uses file as-is

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------
# Load data (prefer sample)
# -----------------------
df = None
for p in INPUT_PATHS:
    if p.exists():
        print(f"Loading {p}")
        # try automatic parsing
        try:
            df = pd.read_csv(p)
        except Exception:
            df = pd.read_csv(p, sep="|", engine="python")
        break

if df is None:
    raise FileNotFoundError("No processed data found. Run preprocess.py first.")

# -----------------------
# Quick cleaning & dtypes
# -----------------------
df.columns = df.columns.str.strip()
for col in ["TotalPremium", "TotalClaims", "CustomValueEstimate", "SumInsured", "CalculatedPremiumPerTerm"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# create target metrics
df["ClaimFrequency"] = (df["TotalClaims"] > 0).astype(int)
df["Severity"] = df["TotalClaims"].where(df["TotalClaims"] > 0, np.nan)
df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

# optional sub-sample if huge (uncomment to use)
if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
    df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

print("Rows after loading:", len(df))

# -----------------------
# Feature selection
# -----------------------
# Candidate features (adjust if your dataset has different names)
CANDIDATE_FEATURES = [
    "Province", "VehicleType", "Gender", "make", "Model",
    "RegistrationYear", "CustomValueEstimate", "SumInsured", "TermFrequency"
]
# Keep only features that exist
features = [f for f in CANDIDATE_FEATURES if f in df.columns]
print("Using features:", features)

# split categorical vs numeric
categorical_feats = [c for c in features if df[c].dtype == "object" or c in ["Province","VehicleType","Gender","make","Model"]]
numeric_feats = [c for c in features if c not in categorical_feats]

print("Categorical:", categorical_feats)
print("Numeric:", numeric_feats)

# -----------------------
# Helper: preprocessing pipeline
# -----------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_feats),
    ("cat", categorical_transformer, categorical_feats)
], remainder="drop", sparse_threshold=0)

# -----------------------
# 1) CLAIM SEVERITY MODEL (regression) - train only on rows with claims
# -----------------------
severity_df = df.dropna(subset=["Severity"])
if len(severity_df) < 50:
    print("Not enough rows with claims to train severity model.")
else:
    X_sev = severity_df[features]
    y_sev = severity_df["Severity"]

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_sev, y_sev, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    models_reg = {}

    # Linear Regression baseline
    pipe_lr = Pipeline([
        ("preproc", preprocessor),
        ("model", LinearRegression())
    ])
    print("Training Linear Regression (Severity)...")
    pipe_lr.fit(X_train_s, y_train_s)
    models_reg["LinearRegression"] = pipe_lr

    # Random Forest
    pipe_rf = Pipeline([
        ("preproc", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    print("Training RandomForestRegressor (Severity)...")
    pipe_rf.fit(X_train_s, y_train_s)
    models_reg["RandomForest"] = pipe_rf

    # XGBoost if available
    if XGBOOST_AVAILABLE:
        pipe_xgb = Pipeline([
            ("preproc", preprocessor),
            ("model", XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0))
        ])
        print("Training XGBoost Regressor (Severity)...")
        pipe_xgb.fit(X_train_s, y_train_s)
        models_reg["XGBoost"] = pipe_xgb

    # Evaluate regression models
    reg_results = []
    for name, model in models_reg.items():
        preds = model.predict(X_test_s)
        rmse = mean_squared_error(y_test_s, preds, squared=False)
        r2 = r2_score(y_test_s, preds)
        reg_results.append({"model":name, "rmse":rmse, "r2":r2})
        # save model
        joblib.dump(model, MODELS_DIR / f"severity_{name}.pkl")
        print(f"{name} -> RMSE: {rmse:.2f}, R2: {r2:.3f}")

    reg_df = pd.DataFrame(reg_results)
    reg_df.to_csv(RESULTS_DIR / "severity_model_metrics.csv", index=False)

# -----------------------
# 2) CLAIM PROBABILITY MODEL (classification)
# -----------------------
# Use whole dataset, predict ClaimFrequency
X = df[features].copy()
y = df["ClaimFrequency"].copy()

# if class imbalance, you might use class_weight or sampling; here we proceed straightforwardly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if y.nunique()>1 else None)

models_clf = {}

# Logistic Regression baseline
pipe_log = Pipeline([
    ("preproc", preprocessor),
    ("model", LogisticRegression(max_iter=500, class_weight="balanced"))
])
print("Training Logistic Regression (Claim Probability)...")
pipe_log.fit(X_train, y_train)
models_clf["LogisticRegression"] = pipe_log

# Random Forest classifier
pipe_rf_clf = Pipeline([
    ("preproc", preprocessor),
    ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"))
])
print("Training RandomForestClassifier (Claim Probability)...")
pipe_rf_clf.fit(X_train, y_train)
models_clf["RandomForest"] = pipe_rf_clf

# XGBoost classifier
if XGBOOST_AVAILABLE:
    pipe_xgb_clf = Pipeline([
        ("preproc", preprocessor),
        ("model", XGBClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, use_label_encoder=False, eval_metric="logloss"))
    ])
    print("Training XGBoost Classifier (Claim Probability)...")
    pipe_xgb_clf.fit(X_train, y_train)
    models_clf["XGBoost"] = pipe_xgb_clf

# Evaluate classification models
clf_results = []
for name, model in models_clf.items():
    preds = model.predict(X_test)
    probs = None
    try:
        probs = model.predict_proba(X_test)[:,1]
    except Exception:
        pass
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc = roc_auc_score(y_test, probs) if probs is not None else np.nan
    clf_results.append({"model":name, "accuracy":acc, "precision":prec, "recall":rec, "f1":f1, "roc_auc":roc})
    joblib.dump(model, MODELS_DIR / f"probability_{name}.pkl")
    print(f"{name} -> acc:{acc:.3f} prec:{prec:.3f} rec:{rec:.3f} f1:{f1:.3f} roc:{roc:.3f}")

pd.DataFrame(clf_results).to_csv(RESULTS_DIR / "probability_model_metrics.csv", index=False)

# -----------------------
# 3) PREMIUM OPTIMIZATION (conceptual)
# Premium = P(claim) * predicted_severity + expense + margin
# We'll compute predicted premium using best available classifier and regressor
# -----------------------
# choose best models by simple metric (lowest RMSE for reg, highest ROC for clf)
best_reg_name = None
best_clf_name = None
if os.path.exists(RESULTS_DIR / "severity_model_metrics.csv"):
    rdf = pd.read_csv(RESULTS_DIR / "severity_model_metrics.csv")
    best_reg_name = rdf.sort_values("rmse").iloc[0]["model"]
if os.path.exists(RESULTS_DIR / "probability_model_metrics.csv"):
    cdf = pd.read_csv(RESULTS_DIR / "probability_model_metrics.csv")
    # prefer highest roc_auc if available else f1
    if cdf["roc_auc"].notnull().any():
        best_clf_name = cdf.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    else:
        best_clf_name = cdf.sort_values("f1", ascending=False).iloc[0]["model"]

print("Best reg:", best_reg_name, "Best clf:", best_clf_name)

# load best models
best_reg = None
best_clf = None
if best_reg_name:
    best_reg = joblib.load(MODELS_DIR / f"severity_{best_reg_name}.pkl")
if best_clf_name:
    best_clf = joblib.load(MODELS_DIR / f"probability_{best_clf_name}.pkl")

if best_reg is not None and best_clf is not None:
    X_full = df[features]
    p_claim = best_clf.predict_proba(X_full)[:,1] if hasattr(best_clf, "predict_proba") else best_clf.predict(X_full)
    pred_sev = best_reg.predict(X_full)
    # Basic premium: expected claim cost
    df["Predicted_Expected_Claim"] = p_claim * pred_sev
    # Add a simple expense loading and target margin (example constants - replace with business numbers)
    EXPENSE_LOADING = 0.10  # 10% of expected claim
    TARGET_MARGIN = 0.15    # 15% profit margin
    df["Predicted_Premium"] = df["Predicted_Expected_Claim"] * (1 + EXPENSE_LOADING + TARGET_MARGIN)
    df[["Predicted_Expected_Claim","Predicted_Premium"]].head().to_csv(RESULTS_DIR / "predicted_premiums_head.csv", index=False)
    print("Predicted premiums computed and saved.")

# -----------------------
# 4) SHAP interpretability for the best regressor (if available)
# -----------------------
if SHAP_AVAILABLE and best_reg is not None:
    print("Computing SHAP values for best regression model...")
    # Need to get preprocessed feature names after ColumnTransformer
    # Create small function to transform X_train for explainer
    # We'll use a sample of data to compute shap values
    try:
        # get preprocessor and model
        preproc = best_reg.named_steps["preproc"]
        model = best_reg.named_steps["model"]
        # transform sample
        X_sample = X_train_s.sample(n=min(1000, len(X_train_s)), random_state=RANDOM_STATE)
        X_trans = preproc.transform(X_sample)
        # shap explainer for tree models or linear
        if hasattr(model, "predict") and model.__class__.__name__.lower().startswith("xgb"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
        elif hasattr(model, "feature_importances_") or hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
        else:
            explainer = shap.LinearExplainer(model, X_trans, feature_dependence="independent")
            shap_values = explainer.shap_values(X_trans)
        # try to recover feature names
        # get column names from preprocessor
        cat_cols = []
        if categorical_feats:
            ohe = preproc.named_transformers_["cat"].named_steps["onehot"]
            cat_cols = list(ohe.get_feature_names_out(categorical_feats))
        num_cols = numeric_feats
        feature_names = list(num_cols) + cat_cols
        # summary plot
        shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "shap_summary_regressor.png", dpi=150)
        plt.close()
        print("Saved SHAP summary plot to results/shap_summary_regressor.png")
    except Exception as e:
        print("SHAP explainability failed:", e)

# -----------------------
# 5) Save sample diagnostics and feature importances
# -----------------------
# If RandomForestRegression was trained, save feature importances (approx via pipeline)
if 'RandomForest' in models_reg:
    rf = models_reg['RandomForest']
    try:
        model_rf = rf.named_steps["model"]
        pre = rf.named_steps["preproc"]
        # get feature names
        cat_cols = []
        if categorical_feats:
            ohe = pre.named_transformers_['cat'].named_steps['onehot']
            cat_cols = list(ohe.get_feature_names_out(categorical_feats))
        feat_names = numeric_feats + cat_cols
        importances = model_rf.feature_importances_
        fi = pd.DataFrame({"feature":feat_names, "importance":importances}).sort_values("importance", ascending=False)
        fi.to_csv(RESULTS_DIR / "feature_importances_randomforest_regression.csv", index=False)
    except Exception as e:
        print("Saving feature importances failed:", e)

print("Modeling pipeline completed. Check results/ and models/ folders for outputs.")
