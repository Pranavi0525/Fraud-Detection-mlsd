"""
IEEE-CIS Fraud Detection — XGBoost Training with MLflow Tracking
Logs parameters, metrics, and model artifact to MLflow
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import time
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix, f1_score
)
import xgboost as xgb

print("=" * 60)
print("  IEEE-CIS Fraud Detection — MLflow Training Run")
print("=" * 60)

# ── CONFIG ────────────────────────────────────────────────────
DATA_PATH  = r"C:\Users\New\Downloads\train_transaction.csv\train_transaction.csv"
MODEL_DIR  = r"D:\SaiU\semester-6\Porjects\Fraud_Detection\models"
EXPERIMENT = "IEEE-CIS-Fraud-Detection"

# Hyperparameters — change these to trigger Iteration 1
PARAMS = {
    "n_estimators"     : 150,
    "max_depth"        : 6,
    "learning_rate"    : 0.05,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "random_state"     : 42,
}

# ── MLFLOW SETUP ─────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///D:/SaiU/semester-6/Porjects/Fraud_Detection/mlflow.db")
mlflow.set_experiment(EXPERIMENT)

print(f"\n[MLflow] Experiment : {EXPERIMENT}")
print(f"[MLflow] Tracking   : ./mlruns")

# ── LOAD DATA ─────────────────────────────────────────────────
print(f"\n[1/6] Loading dataset...")
t0 = time.time()
df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"      Fraud rate: {df['isFraud'].mean()*100:.2f}%")

# ── FEATURES ─────────────────────────────────────────────────
print("\n[2/6] Selecting features...")
drop_cols   = ['TransactionID', 'TransactionDT']
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in drop_cols + ['isFraud']]
missing_rate = df[numeric_cols].isnull().mean()
feature_cols = missing_rate[missing_rate < 0.90].index.tolist()
print(f"      Features kept: {len(feature_cols)}")

X = df[feature_cols].fillna(-999)
y = df['isFraud']

# ── SPLIT ─────────────────────────────────────────────────────
print("\n[3/6] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
PARAMS["scale_pos_weight"] = round(float(scale_pos_weight), 1)

print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"      scale_pos_weight: {scale_pos_weight:.1f}")

# ── MLFLOW RUN ───────────────────────────────────────────────
print("\n[4/6] Starting MLflow run...")
with mlflow.start_run() as run:

    run_id = run.info.run_id
    print(f"      Run ID: {run_id}")

    # Log parameters
    mlflow.log_params(PARAMS)
    mlflow.log_param("n_features", len(feature_cols))
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("fraud_rate_pct", round(float(y.mean() * 100), 2))
    mlflow.log_param("missing_strategy", "-999 sentinel")
    mlflow.log_param("dataset", "IEEE-CIS Fraud Detection (590K transactions)")

    # Train
    print("\n[5/6] Training XGBoost...")
    model = xgb.XGBClassifier(
        **{k: v for k, v in PARAMS.items()},
        use_label_encoder=False,
        eval_metric='auc',
        n_jobs=-1,
        tree_method='hist',
        verbosity=1
    )
    t1 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    training_time = time.time() - t1
    print(f"\n      Training time: {training_time/60:.1f} min")

    # Evaluate
    print("\n[6/6] Evaluating and logging metrics...")
    y_proba = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold
    thresholds = np.arange(0.1, 0.95, 0.01)
    f1_scores = [f1_score(y_test, (y_proba >= t).astype(int), pos_label=1) for t in thresholds]
    optimal_threshold = float(thresholds[np.argmax(f1_scores)])
    y_pred = (y_proba >= optimal_threshold).astype(int)

    auc_roc = roc_auc_score(y_test, y_proba)
    pr_auc  = average_precision_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred,
                target_names=['Legitimate', 'Fraud'], output_dict=True)

    # Log metrics to MLflow
    mlflow.log_metric("auc_roc",           round(auc_roc, 4))
    mlflow.log_metric("pr_auc",            round(pr_auc, 4))
    mlflow.log_metric("optimal_threshold", round(optimal_threshold, 2))
    mlflow.log_metric("accuracy",          round(report['accuracy'], 4))
    mlflow.log_metric("fraud_precision",   round(report['Fraud']['precision'], 4))
    mlflow.log_metric("fraud_recall",      round(report['Fraud']['recall'], 4))
    mlflow.log_metric("fraud_f1",          round(report['Fraud']['f1-score'], 4))
    mlflow.log_metric("training_time_min", round(training_time / 60, 2))
    mlflow.log_metric("true_positives",    int(cm[1, 1]))
    mlflow.log_metric("false_positives",   int(cm[0, 1]))
    mlflow.log_metric("false_negatives",   int(cm[1, 0]))
    mlflow.log_metric("true_negatives",    int(cm[0, 0]))

    # Log model
    mlflow.xgboost.log_model(model, "xgboost_model")

    # Save locally too
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "champion_model.joblib"))
    with open(os.path.join(MODEL_DIR, "champion_metadata.json"), "w") as f:
        json.dump({
            "run_id": run_id,
            "auc_roc": round(auc_roc, 4),
            "pr_auc": round(pr_auc, 4),
            "optimal_threshold": round(optimal_threshold, 2),
            "params": PARAMS,
            "training_samples": len(X_train),
        }, f, indent=2)

    print()
    print("=" * 60)
    print("  MLFLOW RUN COMPLETE")
    print("=" * 60)
    print(f"  AUC-ROC          : {auc_roc:.4f}")
    print(f"  PR-AUC           : {pr_auc:.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.2f}")
    print(f"  Fraud Precision  : {report['Fraud']['precision']:.4f}")
    print(f"  Fraud Recall     : {report['Fraud']['recall']:.4f}")
    print(f"  Run ID           : {run_id}")
    print()
    print("  To view MLflow UI, run:")
    print("  mlflow ui --port 5000")
    print("  Then open: http://localhost:5000")
    print("=" * 60)