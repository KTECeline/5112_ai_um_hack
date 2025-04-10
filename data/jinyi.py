import pandas as pd
import numpy as np
import requests
import json
import os
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# Load API key
API_KEY = os.getenv('CRYPTOQUANT_API_KEY')  # Replace with a string if not using .env

headers = {
    'X-API-Key': API_KEY
}

# API endpoints
cryptoquant_url = "https://api.datasource.cybotrade.rs/cryptoquant/btc/exchange-flows/inflow"
glassnode_url = "https://api.datasource.cybotrade.rs/glassnode/blockchain/utxo_created_value_median"
coinglass_url = "https://api.datasource.cybotrade.rs/coinglass/futures/openInterest/ohlc-history"

# Parameters
start_time = "1743575763000"

cryptoquant_params = {
    "exchange": "okx",
    "window": "hour",
    "start_time": start_time,
    "limit": "1000"
}

glassnode_params = {
    "a": "BTC",
    "c": "usd",
    "i": "1h",
    "start_time": int(start_time),
    "limit": 1000,
    "flatten": False
}

coinglass_params = {
    "exchange": "Binance",
    "symbol": "BTCUSDT",
    "interval": "1h",
    "start_time": start_time,
    "limit": "1000"
}

# Function to fetch API data
def fetch_data_from_api(url, params, headers):
    try:
        response = requests.get(url, headers=headers, params=params)
        print(f"\nRequest URL: {response.url}")
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print("Response Body:", response.text)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Fetch data
cryptoquant_data = fetch_data_from_api(cryptoquant_url, cryptoquant_params, headers)
glassnode_data = fetch_data_from_api(glassnode_url, glassnode_params, headers)
coinglass_data = fetch_data_from_api(coinglass_url, coinglass_params, headers)

# Check all datasets
if not all([cryptoquant_data, glassnode_data, coinglass_data]):
    raise ValueError("Error: One or more datasets could not be fetched.")

# Convert to DataFrames
cryptoquant_df = pd.DataFrame(cryptoquant_data['data'])
glassnode_df = pd.DataFrame(glassnode_data['data'])
coinglass_df = pd.DataFrame(coinglass_data['data'])

# Convert and unify timestamps
cryptoquant_df['timestamp'] = pd.to_datetime(cryptoquant_df['start_time'], unit='ms')
glassnode_df['timestamp'] = pd.to_datetime(glassnode_df['start_time'], unit='ms')
coinglass_df['timestamp'] = pd.to_datetime(coinglass_df['start_time'], unit='ms')

# Merge dataframes on timestamp
try:
    combined_df = pd.merge(cryptoquant_df, glassnode_df, on='timestamp', how='inner')
    combined_df = pd.merge(combined_df, coinglass_df, on='timestamp', how='inner')
except KeyError as e:
    raise ValueError(f"Error merging data: {e}")

# Drop rows with any NaNs
combined_df.dropna(inplace=True)

# Select numeric features only for HMM input (excluding 'start_time' columns)
numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
features = combined_df[numeric_columns]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train HMM
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
model.fit(features_scaled)

# Predict hidden states
hidden_states = model.predict(features_scaled)
combined_df['hidden_states'] = hidden_states

# Output preview
print("\nModel trained successfully!")
print(combined_df[['timestamp', 'hidden_states']].head())

# Save to CSV
combined_df.to_csv("combined_crypto_data_with_hmm.csv", index=False)

# ========================
# 🔍 OPTIONAL: Plot example feature and hidden states
# ========================

# Pick a feature to visualize
feature_name = numeric_columns[1]  # You can change this to 'inflow', 'median_value', etc.

plt.figure(figsize=(15, 6))
plt.plot(combined_df['timestamp'], combined_df[feature_name], label=feature_name, alpha=0.6)
plt.scatter(combined_df['timestamp'], 
            combined_df[feature_name], 
            c=combined_df['hidden_states'], cmap='viridis', label='Hidden States', marker='o')

plt.title(f'{feature_name} with Hidden States Over Time')
plt.xlabel('Timestamp')
plt.ylabel(feature_name)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
