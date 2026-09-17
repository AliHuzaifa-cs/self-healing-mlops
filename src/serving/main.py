"""
FastAPI app serving the LUCK direction-prediction model.
"""

import pickle
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException

from schemas import PredictionRequest, PredictionResponse, HealthResponse
from prometheus_fastapi_instrumentator import Instrumentator

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.pkl"

FEATURE_ORDER = [
    "return_lag1", "return_lag2", "return_lag3",
    "price_vs_ma5", "price_vs_ma20",
    "volatility_5", "volatility_10",
    "volume_change", "volume_vs_ma5",
    "hl_range",
]

app = FastAPI(title="LUCK Direction Prediction API")
Instrumentator().instrument(app).expose(app)

model = None


@app.on_event("startup")
def load_model():
    global model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = None


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = pd.DataFrame([[getattr(request, col) for col in FEATURE_ORDER]], columns=FEATURE_ORDER)

    pred = int(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    confidence = float(max(proba))

    return PredictionResponse(
        prediction=pred,
        prediction_label="up" if pred == 1 else "down",
        confidence=confidence,
    )