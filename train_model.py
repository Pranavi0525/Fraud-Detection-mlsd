"""
IEEE-CIS Fraud Detection — XGBoost Training Pipeline
Real-Time Fraud Detection Project
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix
)
import xgboost as xgb

print("=" * 60)
print("  IEEE-CIS Fraud Detection — XGBoost Training Pipeline")
print("=" * 60)

# ── 1. LOAD DATA ──────────────────────────────────────────────
DATA_PATH = r"C:\Users\New\Downloads\train_transaction.csv\train_transaction.csv"
MODEL_DIR = r"D:\SaiU\semester-6\Porjects\Fraud_Detection\models"

print(f"\n[1/6] Loading dataset from:\n      {DATA_PATH}")
t0 = time.time()
df = pd.read_csv(DATA_PATH)
print(f"      Loaded in {time.time()-t0:.1f}s")
print(f"      Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"      Fraud rate: {df['isFraud'].mean()*100:.2f}%  "
      f"({df['isFraud'].sum():,} fraud / {len(df):,} total)")

# ── 2. FEATURE SELECTION ──────────────────────────────────────
print("\n[2/6] Selecting features...")

# Drop columns with >90% missing or identifier columns
drop_cols = ['TransactionID', 'TransactionDT']

# Keep numeric columns with <90% missing
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in drop_cols + ['isFraud']]

# Filter by missing rate
missing_rate = df[numeric_cols].isnull().mean()
keep_cols = missing_rate[missing_rate < 0.90].index.tolist()

print(f"      Total numeric columns: {len(numeric_cols)}")
print(f"      Columns kept (<90% missing): {len(keep_cols)}")

feature_cols = keep_cols
X = df[feature_cols].copy()
y = df['isFraud'].copy()

# ── 3. MISSING VALUE STRATEGY ─────────────────────────────────
print("\n[3/6] Handling missing values...")
print("      Strategy: -999 sentinel (IEEE-CIS features missing by design)")
X = X.fillna(-999)
print(f"      Missing values after fill: {X.isnull().sum().sum()}")

# ── 4. TRAIN / TEST SPLIT ─────────────────────────────────────
print("\n[4/6] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Training samples : {len(X_train):,}")
print(f"      Testing samples  : {len(X_test):,}")

# Class imbalance ratio
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"      Fraud in train   : {pos:,} ({pos/len(y_train)*100:.2f}%)")
print(f"      scale_pos_weight : {scale_pos_weight:.1f}")

# ── 5. TRAIN XGBOOST ──────────────────────────────────────────
print("\n[5/6] Training XGBoost model...")
print("      (This will take 30-60 minutes on a laptop — please wait)")
print()

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1,
    tree_method='hist',   # fast histogram method
    verbosity=1
)

t1 = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50   # print every 50 trees
)
training_time = time.time() - t1
print(f"\n      Training completed in {training_time/60:.1f} minutes")

# ── 6. EVALUATION ─────────────────────────────────────────────
print("\n[6/6] Evaluating model...")

y_proba = model.predict_proba(X_test)[:, 1]

# Find optimal threshold (maximise F1 on fraud class)
thresholds = np.arange(0.1, 0.95, 0.01)
f1_scores = []
from sklearn.metrics import f1_score
for t in thresholds:
    y_pred_t = (y_proba >= t).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_t, pos_label=1))
optimal_threshold = thresholds[np.argmax(f1_scores)]

y_pred = (y_proba >= optimal_threshold).astype(int)

auc_roc   = roc_auc_score(y_test, y_proba)
pr_auc    = average_precision_score(y_test, y_proba)
cm        = confusion_matrix(y_test, y_pred)

print()
print("=" * 60)
print("  MODEL PERFORMANCE SUMMARY")
print("=" * 60)
print(f"  AUC-ROC          : {auc_roc:.4f}")
print(f"  PR-AUC           : {pr_auc:.4f}")
print(f"  Optimal Threshold: {optimal_threshold:.2f}")
print(f"  Training Time    : {training_time/60:.1f} min")
print()
print("  Confusion Matrix:")
print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")
print()
print("  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Legitimate', 'Fraud']))

# ── 7. SAVE MODEL + METADATA ──────────────────────────────────
os.makedirs(MODEL_DIR, exist_ok=True)

model_path    = os.path.join(MODEL_DIR, "champion_model.joblib")
metadata_path = os.path.join(MODEL_DIR, "champion_metadata.json")
features_path = os.path.join(MODEL_DIR, "feature_names.json")

joblib.dump(model, model_path)
print(f"\n  Model saved      → {model_path}")

metadata = {
    "model_type"         : "XGBoost",
    "dataset"            : "IEEE-CIS Fraud Detection",
    "training_samples"   : int(len(X_train)),
    "testing_samples"    : int(len(X_test)),
    "n_features"         : int(len(feature_cols)),
    "auc_roc"            : round(auc_roc, 4),
    "pr_auc"             : round(pr_auc, 4),
    "optimal_threshold"  : round(float(optimal_threshold), 2),
    "scale_pos_weight"   : round(float(scale_pos_weight), 1),
    "fraud_rate_pct"     : round(float(df['isFraud'].mean() * 100), 2),
    "n_estimators"       : 300,
    "max_depth"          : 6,
    "learning_rate"      : 0.05,
    "training_time_min"  : round(training_time / 60, 1),
    "confusion_matrix"   : cm.tolist()
}

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Metadata saved   → {metadata_path}")

with open(features_path, "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"  Feature names    → {features_path}")

print()
print("=" * 60)
print("  TRAINING COMPLETE — Model ready for inference")
print("=" * 60)