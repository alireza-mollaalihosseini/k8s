import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

# Load (handle gzip if .gz)
df = pd.read_csv("time_series_60min_singleindex.csv", index_col=0, parse_dates=True)

# Focus on Germany wind onshore (or offshore/solar/load)
df_de = df.filter(like='DE_')  # All DE columns

# Target: Wind onshore actual generation (MW)
data = df_de['DE_wind_onshore_generation_actual'].dropna()

# Normalize (LSTM sensitive to scale)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

# Create sequences (lookback = 168 hours ~1 week)
def create_sequences(data, seq_length=168):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

seq_length = 168
X, y = create_sequences(data_scaled, seq_length)

# Train/test split (80/20 time-based)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# To PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# DataLoaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# LSTm model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Last time step

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the Loader
epochs = 20  # Increase for better (50-100, monitor loss)

for epoch in range(epochs):
    model.train()
    for seq, labels in train_loader:
        optimizer.zero_grad()
        y_pred = model(seq)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")


# Evaluate and Forcast
model.eval()
with torch.no_grad():
    preds = model(X_test).numpy()

# Inverse scale
preds = scaler.inverse_transform(preds)
y_test_actual = scaler.inverse_transform(y_test.numpy())

# Plot sample
plt.figure(figsize=(12,6))
plt.plot(y_test_actual[:500], label="Actual")
plt.plot(preds[:500], label="Predicted")
plt.title("LSTM Wind Power Forecast vs Actual")
plt.legend()
# plt.show()
plt.savefig("LSTM-wind-forcast.png", dpi=300)

# RMSE
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test_actual, preds))
print(f"Test RMSE: {rmse:.2f} MW")