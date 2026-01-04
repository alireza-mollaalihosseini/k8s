import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
import uvicorn

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

    # model_path = os.getenv("MODEL_PATH", "lstm_weather.pt")

    # checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load("lstm_weather.pt", weights_only=False)

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
    df = pd.read_csv(csv_path, skiprows=2)
    df = df[FEATURES].dropna()

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
    csv_path: str = "weather_20260103.csv"
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail="CSV file not found")

    preds = forecast_recursive(hours, csv_path)

    return {
        "hours": hours,
        "forecast_wind_speed_kmh": preds
    }

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
