from fastapi import FastAPI
from pydantic import BaseModel
import joblib, json, numpy as np, pandas as pd

app = FastAPI(title="Fraud Detection API", description="IEEE-CIS Fraud Detection using XGBoost", version="1.0.0")

model = joblib.load("models/champion_model.joblib")
with open("models/champion_metadata.json") as f: meta = json.load(f)
with open("models/feature_names.json") as f: FEATURE_NAMES = json.load(f)
THRESHOLD = meta.get("optimal_threshold", 0.5)

class TransactionFeatures(BaseModel):
    TransactionAmt: float = 250.0
    card1: float = 9999.0
    card2: float = 100.0
    card3: float = 150.0
    card5: float = 226.0
    addr1: float = 299.0
    addr2: float = 87.0
    dist1: float = 500.0

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running", "model": "XGBoost", "features": len(FEATURE_NAMES)}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "threshold": THRESHOLD}

@app.post("/predict")
def predict(tx: TransactionFeatures):
    row = {f: -999 for f in FEATURE_NAMES}
    for k, v in tx.model_dump().items():
        if k in row: row[k] = v
    X = pd.DataFrame([row])[FEATURE_NAMES]
    proba = float(model.predict_proba(X)[0][1])
    prediction = int(proba >= THRESHOLD)
    return {
        "prediction": prediction,
        "label": "Fraud Detected" if prediction == 1 else "Legitimate Transaction",
        "probability": round(proba, 4),
        "risk_level": "High" if proba > 0.7 else "Medium" if proba > 0.3 else "Low"
    }
