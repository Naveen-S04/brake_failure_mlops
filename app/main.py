from fastapi import FastAPI
from pydantic import BaseModel, Field
import os, joblib, numpy as np
from pathlib import Path

class Instance(BaseModel):
    speed: float
    brake_pedal_pressure: float = Field(ge=0.0, le=1.0)
    rotor_temp: float
    vibration: float
    battery_voltage: float
    pad_thickness: float
    mileage: float
    ambient_temp: float
    humidity: float

class RequestBody(BaseModel):
    instances: list[Instance]

app = FastAPI(title="Brake Failure Prediction API", version="1.0.0")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_FILE = MODEL_DIR / "brake_failure_xgb.pkl"
SCALER_PATH = Path("artifacts/scaler.pkl")

model = None
scaler = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    if MODEL_FILE.exists():
        model = joblib.load(MODEL_FILE)
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)

@app.get("/health")
def health():
    status = (model is not None) and (scaler is not None)
    return {"status": "ok" if status else "booting", "model_loaded": model is not None, "scaler_loaded": scaler is not None}

@app.post("/predict")
def predict(body: RequestBody):
    assert model is not None and scaler is not None, "Model/Scaler not loaded. Train pipeline first."
    X = np.array([[
        x.speed, x.brake_pedal_pressure, x.rotor_temp, x.vibration, x.battery_voltage,
        x.pad_thickness, x.mileage, x.ambient_temp, x.humidity
    ] for x in body.instances], dtype=float)
    X = scaler.transform(X)
    proba = model.predict_proba(X)[:,1].tolist()
    preds = (np.array(proba) >= 0.5).astype(int).tolist()
    return {"predictions": preds, "probabilities": proba}