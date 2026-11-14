# api/main.py
# --------------------------------------------------------------
# FastAPI inference service for NASA SpaceGuard AI
# --------------------------------------------------------------

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List


import torch
import numpy as np
import joblib
from pathlib import Path
import logging

logger = logging.getLogger("uvicorn.error")


# ------------------------------------------------------------------
# 1) Load model checkpoint and scaler at startup
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CHECKPOINT = Path(__file__).parent.parent / "models" / "checkpoints" / "P-1_best.pt"
SCALER_PATH = Path(__file__).parent.parent / "data" / "processed" / "scaler.joblib"

# Model class – same as src/spaceguard/models/lstm.py
# (imported to avoid circular path issues)
from src.spaceguard.models.lstm import TelemetryLSTM
from src.spaceguard.detection.smoothing import exponential_moving_average
from src.spaceguard.detection.threshold import compute_dynamic_threshold, detect_anomalies_from_errors

# Load scaler (MinMaxScaler saved during training)
try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load scaler from {SCALER_PATH}: {e}")

# Load model checkpoint
try:
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    # Try to read explicit input dimension saved during training
    in_dim = checkpoint.get("input_dim")
    if in_dim is None:
        # LSTM weight shape: (4 * hidden, input_dim)
        # We need the original input dimension directly (no division)
        in_dim = checkpoint["model_state_dict"]["lstm.weight_ih_l0"].size(1)
    model = TelemetryLSTM(input_dim=in_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    raise RuntimeError(f"Failed to load model from {MODEL_CHECKPOINT}: {e}")

# ------------------------------------------------------------------
# 2) FastAPI app definition
# ------------------------------------------------------------------
app = FastAPI(
    title="NASA SpaceGuard AI",
    description="LSTM‑based spacecraft telemetry anomaly detection API",
    version="1.0.0",
)

# ------------------------------------------------------------------
# 3) Pydantic request schema
# ------------------------------------------------------------------
class TelemetryWindow(BaseModel):
    """
    250‑step sliding window for a single telemetry channel.
    `data` must be a list of 250 rows, each row containing `in_dim` features.
    """
    data: List[List[float]] = Field(..., description="2‑D list with shape (250, in_dim)")
    true_next: float = Field(..., description="Raw telemetry value for the timestep immediately following the window")

    @validator("data")
    def check_shape(cls, v):
        if len(v) != 250:
            raise ValueError("Window must contain exactly 250 time steps")
        # Ensure consistent inner length
        inner_len = len(v[0])
        if any(len(row) != inner_len for row in v):
            raise ValueError("All rows must have the same number of features")
        return v

# ------------------------------------------------------------------
# 4) Health‑check endpoint (now reports model status)
# ------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "device": str(DEVICE),
    }

# ------------------------------------------------------------------
# 5) Prediction endpoint
# ------------------------------------------------------------------
@app.post("/api/predict")
def predict(window: TelemetryWindow):
    """Predict next telemetry value for a single 250‑step window.

    Returns a JSON payload containing the scaled prediction, the true scaled
    value, absolute error, a placeholder EMA‑smoothed error, a dynamic
    threshold (computed from the scaler‑stored statistics) and a boolean flag
    indicating whether the error exceeds the threshold.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to NumPy array → torch tensor
        raw_np = np.array(window.data, dtype=np.float32)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid input payload: {exc}")

    # ------------------------------------------------------------------
    # Scaling – we scale only the first column (telemetry) exactly as during training
    # (the scaler was fit on the first column of the raw data)
    # ------------------------------------------------------------------
    try:
        scaled_np = raw_np.copy()
        scaled_np[:, 0] = scaler.transform(raw_np[:, [0]]).flatten()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scaling failed: {exc}")

    # Verify input dimensions match model's expectation
    if scaled_np.shape[1] != model.input_dim:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input feature dimension mismatch: received {scaled_np.shape[1]}, "
                f"model expects {model.input_dim}."
            ),
        )

    tensor = torch.from_numpy(scaled_np).unsqueeze(0).to(DEVICE)  # shape (1, 250, in_dim)

    # ------------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------------
    try:
        with torch.no_grad():
            pred = model(tensor)                       # shape (1, 1)
            pred_val = pred.squeeze().cpu().numpy()    # scalar
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}")

    # ------------------------------------------------------------------
    # Compute absolute error using the provided true_next value
    # Scale the true_next telemetry value using the same scaler as during training
    # ------------------------------------------------------------------
    true_next_raw = np.array([[window.true_next]], dtype=np.float32)
    true_next_scaled = scaler.transform(true_next_raw).flatten()[0]
    error = float(abs(true_next_scaled - pred_val))

    # ------------------------------------------------------------------
    # EMA smoothing (window size 10) – placeholder for client aggregation
    # ------------------------------------------------------------------
    smoothed_error = error

    # ------------------------------------------------------------------
    # Dynamic threshold (z‑score based) using scaler‑stored statistics
    # ------------------------------------------------------------------
    error_mean = getattr(scaler, "error_mean", 0.0)
    error_std = getattr(scaler, "error_std", 1.0)
    z_thresh = 3.0
    epsilon = error_mean + z_thresh * error_std


    is_anomaly = error > epsilon

    return {
        "prediction": float(pred_val),          # scaled prediction
        "true_value": float(true_val),          # scaled ground‑truth
        "error": error,
        "smoothed_error": smoothed_error,
        "threshold": float(epsilon),
        "is_anomaly": is_anomaly,
        "z_thresh": z_thresh,
    }

    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert input to NumPy array → torch tensor
    raw_np = np.array(window.data, dtype=np.float32)

    # ------------------------------------------------------------------
    # Scaling – we scale only the first column (telemetry) exactly as during training
    # (the scaler was fit on the first column of the raw data)
    # ------------------------------------------------------------------
    scaled_np = raw_np.copy()
    # Assuming scaler was fitted on column 0 only (as in training script)
    scaled_np[:, 0] = scaler.transform(raw_np[:, [0]]).flatten()
    tensor = torch.from_numpy(scaled_np).unsqueeze(0).to(DEVICE)  # shape (1, 250, in_dim)

    # ------------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------------
    with torch.no_grad():
        pred = model(tensor)                       # shape (1, 1)
        # We predict the *next* telemetry value (column 0)
        pred_val = pred.squeeze().cpu().numpy()    # scalar

    # ------------------------------------------------------------------
    # Compute absolute error for the *last* timestep of the window
    # (ground‑truth = real telemetry at step 250)
    # ------------------------------------------------------------------
    true_val = scaled_np[-1, 0]                     # already scaled
    error = float(abs(true_val - pred_val))

    # ------------------------------------------------------------------
    # EMA smoothing (window size 10) – same as evaluation script
    # ------------------------------------------------------------------
    # For a single point we cannot really smooth, so we just return the raw error.
    # In a real streaming scenario you would keep a buffer; here we expose the
    # smoothing function for consistency.
    smoothed_error = error  # placeholder – downstream client can apply EMA over a sequence

    # ------------------------------------------------------------------
    # Dynamic threshold (z‑score based)
    # ------------------------------------------------------------------
    # We need mean & std of the error distribution. For a single request we
    # cannot compute them, so we use the statistics computed on the training
    # set and persisted alongside the scaler (stored in the same .joblib file).
    # For simplicity we embed them in the scaler file as attributes `error_mean`
    # and `error_std`. If they are missing we fall back to z=3.0 with mean=0.
    error_mean = getattr(scaler, "error_mean", 0.0)
    error_std = getattr(scaler, "error_std", 1.0)
    z_thresh = 3.0
    epsilon = compute_dynamic_threshold(np.array([error]), z_threshold=z_thresh)

    is_anomaly = error > epsilon

    return {
        "prediction": float(pred_val),          # scaled prediction
        "true_value": float(true_val),          # scaled ground‑truth
        "error": error,
        "smoothed_error": smoothed_error,
        "threshold": float(epsilon),
        "is_anomaly": is_anomaly,
        "z_thresh": z_thresh,
    }

# ------------------------------------------------------------------
# Optional: expose OpenAPI docs at /docs (already default)
# ------------------------------------------------------------------