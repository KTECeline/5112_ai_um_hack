import pandas as pd
import numpy as np
import requests
import json
import os
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# ============== 🔐 API KEY =================
API_KEY = os.getenv('CRYPTOQUANT_API_KEY')  # Or just use your key as a string
headers = {'X-API-Key': API_KEY}

# ============== 📅 Parameters =================
start_time = "1743575763000"

# ============== 🌐 API URLs & Params =================
cryptoquant_url = "https://api.datasource.cybotrade.rs/cryptoquant/btc/exchange-flows/inflow"
glassnode_url = "https://api.datasource.cybotrade.rs/glassnode/blockchain/utxo_created_value_median"
coinglass_url = "https://api.datasource.cybotrade.rs/coinglass/futures/openInterest/ohlc-history"

# Updated Binance OHLC URL
binance_ohlc_url = "https://api.binance.com/api/v3/klines"

params = {
    'cryptoquant': {
        "exchange": "okx",
        "window": "hour",
        "start_time": start_time,
        "limit": "1000"
    },
    'glassnode': {
        "a": "BTC",
        "c": "usd",
        "i": "1h",
        "start_time": int(start_time),
        "limit": 1000,
        "flatten": False
    },
    'coinglass': {
        "exchange": "Binance",
        "symbol": "BTCUSDT",
        "interval": "1h",
        "start_time": start_time,
        "limit": "1000"
    },
    'binance': {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "startTime": start_time,  # Use startTime instead of start_time
        "limit": 1000
    }
}

# ============== 🔁 Helper: Fetch =================
def fetch_data(url, params, headers):
    try:
        print(f"⏳ Fetching data from {url}...")
        response = requests.get(url, headers=headers, params=params)
        print(f"Request URL: {response.url}")
        print(f"Status Code: {response.status_code}")
        if response.status_code != 200:
            print("Response Body:", response.text)
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

# ============== 🧲 Get All Data =================
print("\n🔄 Fetching data from all sources...")
cryptoquant_data = fetch_data(cryptoquant_url, params['cryptoquant'], headers)
glassnode_data = fetch_data(glassnode_url, params['glassnode'], headers)
coinglass_data = fetch_data(coinglass_url, params['coinglass'], headers)
binance_data = fetch_data(binance_ohlc_url, params['binance'], headers)

if not all([cryptoquant_data, glassnode_data, coinglass_data, binance_data]):
    raise ValueError("❌ Error: One or more datasets could not be fetched.")
print("✅ All data fetched successfully.")

# ============== 📊 Convert to DataFrames =================
print("\n🔄 Converting data to DataFrames...")
cryptoquant_df = pd.DataFrame(cryptoquant_data['data'])
glassnode_df = pd.DataFrame(glassnode_data['data'])
coinglass_df = pd.DataFrame(coinglass_data['data'])
binance_df = pd.DataFrame(binance_data)
print("✅ DataFrames created.")

# ============== 🕒 Timestamps =================
print("\n🕒 Converting timestamps to datetime...")
cryptoquant_df['timestamp'] = pd.to_datetime(cryptoquant_df['start_time'], unit='ms')
glassnode_df['timestamp'] = pd.to_datetime(glassnode_df['start_time'], unit='ms')
coinglass_df['timestamp'] = pd.to_datetime(coinglass_df['start_time'], unit='ms')
binance_df['timestamp'] = pd.to_datetime(binance_df[0], unit='ms')
print("✅ Timestamps converted.")

# ============== 🧹 Drop Duplicates =================
print("\n🧹 Dropping unnecessary columns...")
cryptoquant_df = cryptoquant_df.drop(columns=['start_time'], errors='ignore')
glassnode_df = glassnode_df.drop(columns=['start_time'], errors='ignore')
coinglass_df = coinglass_df.drop(columns=['start_time'], errors='ignore')
binance_df = binance_df.drop(columns=[0, 6, 7, 8, 9, 10])  # Removing unnecessary columns from Binance DataFrame
print("✅ Unnecessary columns dropped.")

# ============== 🔗 Merge All =================
print("\n🔗 Merging all datasets...")
combined_df = cryptoquant_df.merge(glassnode_df, on='timestamp', how='inner') \
                            .merge(coinglass_df, on='timestamp', how='inner') \
                            .merge(binance_df, on='timestamp', how='inner')
print("✅ Datasets merged.")

# ============== 📅 Add readable date =================
print("\n📅 Adding readable date column...")
combined_df['datetime'] = combined_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
print("✅ Readable date added.")

# ============== 🧽 Clean & Prepare =================
print("\n🧽 Cleaning and preparing the data...")
combined_df.dropna(inplace=True)
numeric_columns = combined_df.select_dtypes(include=[np.number]).columns
features = combined_df[numeric_columns]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
print("✅ Data cleaned and features scaled.")

# ============== 🧠 Train HMM =================
print("\n🧠 Training the Hidden Markov Model (HMM)...")
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
model.fit(features_scaled)
combined_df['hidden_states'] = model.predict(features_scaled)
print("✅ HMM trained.")

# ============== 📊 Evaluate Model Accuracy =================
print("\n📊 Evaluating model accuracy...")

# Log-likelihood: higher values are better (indicating a better fit to the data)
log_likelihood = model.score(features_scaled)
print(f"✅ Log-Likelihood: {log_likelihood}")

# AIC (Akaike Information Criterion): Lower is better
aic = model.score(features_scaled) * 2 - 2 * len(features_scaled)
print(f"✅ AIC: {aic}")

# BIC (Bayesian Information Criterion): Lower is better
bic = model.score(features_scaled) * np.log(len(features_scaled)) - 2 * len(features_scaled)
print(f"✅ BIC: {bic}")

# ============== 💾 Export =================
print("\n💾 Exporting data to CSV...")
combined_df.to_csv("combined_crypto_data_with_hmm.csv", index=False)
print("✅ Exported to combined_crypto_data_with_hmm.csv")

# ============== 📈 Plot Optional =================
# feature_name = numeric_columns[1]
# print(f"\n📈 Plotting the feature {feature_name} with Hidden States...")
# plt.figure(figsize=(15, 6))
# plt.plot(combined_df['timestamp'], combined_df[feature_name], label=feature_name, alpha=0.6)
# plt.scatter(combined_df['timestamp'], 
#             combined_df[feature_name], 
#             c=combined_df['hidden_states'], cmap='viridis', label='Hidden States', marker='o')
# plt.title(f'{feature_name} with Hidden States Over Time')
# plt.xlabel('Timestamp')
# plt.ylabel(feature_name)
# plt.xticks(rotation=45)
# plt.legend()
# plt.tight_layout()
# plt.show()
# print("✅ Plotting complete.")
