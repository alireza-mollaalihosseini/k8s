import requests
import pandas as pd
import os
from datetime import datetime

def ingest_weather(lat=52.52, lon=13.41, days=30):
    # Open-Meteo API (free)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": (datetime.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
        "end_date": datetime.now().strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,temperature_2m_min,wind_speed_10m_max",
        "format": "csv"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    # Save CSV
    output_dir = os.getenv("DATA_DIR", "./data")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/weather_{datetime.now().strftime('%Y%m%d')}.csv"
    with open(filename, "w") as f:
        f.write(response.text)
    print(f"Ingested data saved to {filename}")

if __name__ == "__main__":
    ingest_weather()