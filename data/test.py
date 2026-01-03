import pandas as pd
import matplotlib.pyplot as plt

# Load (handle gzip if .gz)
df = pd.read_csv("time_series_60min_singleindex.csv", index_col=0, parse_dates=True)

# Focus on Germany wind onshore (or offshore/solar/load)
df_de = df.filter(like='DE_')  # All DE columns
print(df_de.columns)  # e.g., 'DE_load_actual_entsoe_transparency', 'DE_wind_onshore_generation_actual'

# Target: Wind onshore actual generation (MW)
data = df_de['DE_wind_onshore_generation_actual'].dropna()

# Explore
print(data.head())
data.plot(figsize=(12,6), title="Germany Onshore Wind Power Production (Hourly)")
# plt.show()
plt.savefig("germany-onshore-wind.png", dpi=300)

# Stats
print(data.describe())