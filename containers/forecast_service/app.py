import os
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
import uvicorn
import traceback

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Forecast Service")

model = None
scaler_X = None
scaler_y = None
SEQ_LENGTH = None

FEATURES = [
    'temperature_2m_max (°C)',
    'temperature_2m_min (°C)',
    'wind_speed_10m_max (km/h)'
]

DATA_DIR = os.getenv("DATA_DIR", "/data")
MODEL_PATH = os.getenv("MODEL_PATH", "/data/lstm_weather.pt")

# -------------------------
# LSTM Model Definition
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# -------------------------
# Startup: load model
# -------------------------
@app.on_event("startup")
async def load_model():
    global model, scaler_X, scaler_y, SEQ_LENGTH

    checkpoint = torch.load(MODEL_PATH, weights_only=False)

    model = LSTMModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler_X = checkpoint["scaler_X"]
    scaler_y = checkpoint["scaler_y"]
    SEQ_LENGTH = checkpoint["seq_length"]

# -------------------------
# Helpers
# -------------------------
def load_latest_sequence(csv_path: str):
    # Skip 2 metadata lines — Line 3 becomes header
    df = pd.read_csv(csv_path, skiprows=2, encoding="ISO-8859-1")
    
    # Exact FEATURES from Open-Meteo CSV
    FEATURES = [
        'temperature_2m_max (°C)',
        'temperature_2m_min (°C)',
        'wind_speed_10m_max (km/h)'
    ]
    
    df = df[FEATURES].dropna()
    
    if len(df) < SEQ_LENGTH:
        raise ValueError(f"Not enough data rows ({len(df)}) for SEQ_LENGTH {SEQ_LENGTH}")
    
    values = df.values[-SEQ_LENGTH:]
    values_scaled = scaler_X.transform(values)

    x = torch.tensor(values_scaled, dtype=torch.float32)
    return x.unsqueeze(0)  # (1, seq_length, 3)

def forecast_recursive(hours: int, csv_path: str):
    x = load_latest_sequence(csv_path)
    preds = []

    with torch.no_grad():
        for _ in range(hours):
            y_scaled = model(x)
            y = scaler_y.inverse_transform(y_scaled.numpy())[0, 0]
            preds.append(y)

            last_step = x[:, -1, :].clone()
            last_step[0, 2] = y_scaled  # update wind speed only

            x = torch.cat([x[:, 1:, :], last_step.unsqueeze(1)], dim=1)

    return preds

# -------------------------
# Health endpoints (K8s)
# -------------------------
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    if model is not None:
        return {"status": "ready", "model_loaded": True}
    raise HTTPException(status_code=503, detail="Model not loaded")

# -------------------------
# Prediction endpoint
# -------------------------
@app.get("/predict")
async def predict(
    hours: int = 24,
    csv_path: str | None = None  # Now optional
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    # if csv_path is None:
    #     # Auto-find latest weather_*.csv in /data
    #     csv_files = glob.glob(f"{DATA_DIR}/weather_*.csv")
    #     if not csv_files:
    #         raise HTTPException(status_code=400, detail="No weather CSV files found")
    #     csv_path = max(csv_files, key=os.path.getctime)

    # # Support relative filenames (e.g., ?csv_path=weather_20260106.csv)
    # full_path = csv_path if os.path.isabs(csv_path) else os.path.join(DATA_DIR, csv_path)

    # if not os.path.exists(full_path):
    #     raise HTTPException(status_code=400, detail=f"CSV file not found: {full_path}")

    # preds = forecast_recursive(hours, full_path)

    # return {
    #     "hours": hours,
    #     "csv_used": os.path.basename(full_path),
    #     "forecast_wind_speed_kmh": preds
    # }
    try:
        if csv_path is None:
            csv_files = glob.glob(f"{DATA_DIR}/weather_*.csv")
            if not csv_files:
                raise HTTPException(status_code=400, detail="No weather CSV files found")
            csv_path = max(csv_files, key=os.path.getctime)

        full_path = csv_path if os.path.isabs(csv_path) else os.path.join(DATA_DIR, csv_path)

        if not os.path.exists(full_path):
            raise HTTPException(status_code=400, detail=f"CSV file not found: {full_path}")

        # Temporary debug: load and inspect
        df = pd.read_csv(full_path, skiprows=2, encoding="ISO-8859-1")  # Try without skiprows first
        print("CSV Columns:", list(df.columns))  # Will appear in pod logs
        print("CSV Rows:", len(df))
        print("First few rows:\n", df.head())

        df = df[FEATURES].dropna()
        
        if len(df) < SEQ_LENGTH:
            raise ValueError(f"Not enough rows after dropna: {len(df)} < {SEQ_LENGTH}")

        preds = forecast_recursive(hours, full_path)

        np.savetxt(f"{DATA_DIR}/predictions.txt", np.array(preds))

        return {
            "hours": hours,
            "csv_used": os.path.basename(full_path),
            "forecast_wind_speed_kmh": preds
        }

    except Exception as e:
        error_detail = traceback.format_exc()  # Full traceback
        print("Prediction Error:\n", error_detail)  # Logs full error
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}\n{error_detail}")

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
