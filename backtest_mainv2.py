import requests
import pandas as pd
import os
from datetime import datetime, timedelta

# ============== 🔐 API KEY =================
API_KEY = "7zVOILbXdb1kMsf5NTQ2FI5BTVcmJglQaE0EK9YQybxWvHZP"
headers = {'X-API-Key': API_KEY}

# ============== 📅 Parameters =================
# Set the start time to get data from 4 years ago
start_time = int((datetime.now() - timedelta(days=4*365)).timestamp() * 1000)  # in milliseconds

# ============== 🌐 API URLs & Params =================
binance_ohlc_url = "https://api.binance.com/api/v3/klines"
coinglass_url = "https://api.datasource.cybotrade.rs/coinglass/futures/openInterest/ohlc-history"

params = {
    'binance': {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "startTime": start_time,
        "limit": 1000
    },
    'coinglass': {
        "exchange": "Binance",
        "symbol": "BTCUSDT",
        "interval": "1h",
        "start_time": start_time,
        "limit": "1000"
    }
}

# ============== 🔁 Helper: Fetch =================
def fetch_data(url, params, headers=None):
    try:
        print(f"⏳ Fetching data from {url}...")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

# ============== 🧲 Get Data =================
print("\n🔄 Fetching data from Binance...")
binance_data = fetch_data(binance_ohlc_url, params['binance'])

print("\n🔄 Fetching data from Coinglass...")
coinglass_data = fetch_data(coinglass_url, params['coinglass'], headers)

# Check if data was fetched successfully
if not binance_data or not coinglass_data:
    raise ValueError("❌ Error: One or more datasets could not be fetched.")
print("✅ Data fetched successfully.")

# ============== 📊 Convert to DataFrames =================
print("\n🔄 Converting data to DataFrames...")
binance_df = pd.DataFrame(binance_data, columns=[
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base', 'taker_buy_quote', 'ignore'
])
binance_df['timestamp'] = pd.to_datetime(binance_df['open_time'], unit='ms')
binance_df[['open', 'high', 'low', 'close', 'volume']] = binance_df[['open', 'high', 'low', 'close', 'volume']].astype(float)
binance_df = binance_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

coinglass_df = pd.DataFrame(coinglass_data['data'])
coinglass_df['timestamp'] = pd.to_datetime(coinglass_df['start_time'], unit='ms')
coinglass_df = coinglass_df[['timestamp', 'spot_price', 'futures_price']]

# ============== 🔗 Merge Data =================
print("\n🔗 Merging datasets...")
combined_df = pd.merge(binance_df, coinglass_df, on='timestamp', how='inner')

# ============== 💾 Export =================
output_file = "historical_crypto_data.csv"
print(f"\n💾 Exporting data to {output_file}...")
combined_df.to_csv(output_file, index=False)
print(f"✅ Data exported to {output_file}")
