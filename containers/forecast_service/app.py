import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

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
# Load model checkpoint
# -------------------------
# MODEL_PATH = os.getenv("MODEL_PATH", "lstm_weather.pt")

# checkpoint = torch.load("lstm_weather.pt", map_location="cpu")
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

FEATURES = [
    'temperature_2m_max (°C)',
    'temperature_2m_min (°C)',
    'wind_speed_10m_max (km/h)'
]

# -------------------------
# Load latest sequence
# -------------------------
def load_latest_sequence(csv_path):
    df = pd.read_csv(csv_path, skiprows=2)
    df = df[FEATURES].dropna()

    values = df.values[-SEQ_LENGTH:]
    values_scaled = scaler_X.transform(values)

    x = torch.tensor(values_scaled, dtype=torch.float32)
    return x.unsqueeze(0)  # (1, seq_length, 3)


# -------------------------
# One-step forecast
# -------------------------
def forecast_next_step(csv_path):
    x = load_latest_sequence(csv_path)

    with torch.no_grad():
        y_scaled = model(x).numpy()

    return float(scaler_y.inverse_transform(y_scaled)[0, 0])


# -------------------------
# Multi-step forecast (recursive)
# -------------------------
def forecast(hours=24, csv_path="weather_latest.csv"):
    x = load_latest_sequence(csv_path)

    preds = []

    with torch.no_grad():
        for _ in range(hours):
            y_scaled = model(x)           # (1,1)
            y = scaler_y.inverse_transform(y_scaled.numpy())[0, 0]
            preds.append(y)

            # Build next input step
            last_step = x[:, -1, :].clone()

            # Keep temperatures constant
            last_step[0, 2] = y_scaled    # update wind speed only

            x = torch.cat(
                [x[:, 1:, :], last_step.unsqueeze(1)],
                dim=1
            )

    return np.array(preds)


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    preds = forecast(hours=24, csv_path="weather_20260103.csv")
    print("Next 24h wind speed forecast (km/h):")
    print(preds)