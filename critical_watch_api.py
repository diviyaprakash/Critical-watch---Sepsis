    import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from critical_watch_backend import (
    SepsisLSTM, VITALS, WINDOW_HOURS, SEQ_CHANNELS,
    engineered_features
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

artifacts = joblib.load("critical_watch_models.joblib")
preprocessor = artifacts["preprocessor"]
xgb_model = artifacts["xgboost"]
calibrator = artifacts["calibrator"]
lstm_weight, xgb_weight = artifacts["ensemble_weights"]
tabular_columns = artifacts["tabular_columns"]

lstm_model = SepsisLSTM(input_size=18).to(DEVICE)
lstm_model.load_state_dict(
    torch.load("critical_watch_lstm.pt", map_location=DEVICE)
)
lstm_model.eval()

app = FastAPI(
    title="Critical Watch API",
    description="Educational sepsis-risk inference prototype. Not for clinical use."
)


class HourReading(BaseModel):
    hour: float
    heart_rate: float | None = None
    temperature: float | None = None
    respiratory_rate: float | None = None
    systolic_bp: float | None = None
    spo2: float | None = None
    wbc: float | None = None


class PredictionRequest(BaseModel):
    patient_id: str = "demo-patient"
    readings: List[HourReading] = Field(min_length=24, max_length=24)


@app.get("/")
def home():
    return {
        "service": "Critical Watch",
        "status": "running",
        "notice": "Educational prototype only; not for clinical decisions."
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        rows = []
        for reading in request.readings:
            row = reading.model_dump()
            row["patient_id"] = request.patient_id
            # Onset is unknown at prediction time; this field is only needed
            # by the shared preprocessing structure, not by inference.
            row["sepsis_onset_hour"] = np.nan
            rows.append(row)

        raw = pd.DataFrame(rows).sort_values("hour")
        processed = preprocessor.transform(raw).reset_index(drop=True)

        if len(processed) != WINDOW_HOURS:
            raise ValueError("Exactly 24 hourly readings are required.")

        sequence = processed[SEQ_CHANNELS].to_numpy(dtype=np.float32)
        if not np.isfinite(sequence).all():
            raise ValueError("Input contains invalid or non-finite values.")

        features = engineered_features(processed)
        tabular = pd.DataFrame([features]).reindex(
            columns=tabular_columns, fill_value=0
        ).fillna(0)

        with torch.no_grad():
            x = torch.tensor(sequence[None, :, :], dtype=torch.float32).to(DEVICE)
            p_lstm = torch.sigmoid(lstm_model(x)).item()

        p_xgb = float(xgb_model.predict_proba(tabular)[:, 1][0])
        raw_ensemble = lstm_weight * p_lstm + xgb_weight * p_xgb
        risk = float(calibrator.predict([raw_ensemble])[0])
        risk = max(0.0, min(1.0, risk))

        return {
            "patient_id": request.patient_id,
            "sepsis_risk_probability": round(risk, 4),
            "risk_percent": round(risk * 100, 1),
            "model_probabilities": {
                "lstm": round(p_lstm, 4),
                "xgboost": round(p_xgb, 4)
            },
            "ensemble_weights": {
                "lstm": lstm_weight,
                "xgboost": xgb_weight
            },
            "note": (
                "Demonstration output only. A validated clinical alert "
                "threshold and clinician workflow are not configured."
            )
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
