import requests
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import ccxt

# Configuration
API_KEY = os.getenv('CRYPTOQUANT_API_KEY')
EXCHANGE = 'binance'
SYMBOLS = ['BTC/USDT', 'ETH/USDT']
TIMEFRAME = '1h'
YEARS_OF_DATA = 3

# API URLs
CRYPTOQUANT_BASE_URL = "https://api.datasource.cybotrade.rs/cryptoquant"
GLASSNODE_URL = "https://api.datasource.cybotrade.rs/glassnode/blockchain/utxo_created_value_median"
COINGLASS_URL = "https://api.datasource.cybotrade.rs/coinglass/futures/openInterest/ohlc-history"

# Generic fetch function
def fetch_data_from_api(url, params, headers, source_type='cryptoquant'):
    end_time = int(time.time() * 1000)
    start_time = end_time - (YEARS_OF_DATA * 365 * 24 * 60 * 60 * 1000)
    params["start_time"] = str(start_time)
    print(f"Fetching {source_type} data from {url}...")
    all_data = []
    current_start = start_time
    
    while True:
        params["start_time"] = str(int(current_start)) if source_type == 'glassnode' else str(current_start)
        response = requests.get(url, headers=headers, params=params)
        print(f"{source_type} status: {response.status_code}, Response: {response.text[:200]}")
        if response.status_code == 200:
            data = response.json()
            if source_type == 'cryptoquant' and "data" in data and isinstance(data["data"], list) and data["data"]:
                all_data.extend(data["data"])
                limit = int(params.get("limit", 10000))  # Convert to int
                if len(data["data"]) < limit:
                    break
                current_start = data["data"][-1]['start_time'] + 1
            elif source_type == 'glassnode' and "data" in data and isinstance(data["data"], list) and data["data"]:
                all_data.extend(data["data"])
                limit = int(params.get("limit", 1000))  # Convert to int
                if len(data["data"]) < limit:
                    break
                current_start = data["data"][-1]['start_time'] + 1
            elif source_type == 'coinglass' and "data" in data and isinstance(data["data"], list) and data["data"]:
                all_data.extend(data["data"])
                limit = int(params.get("limit", 1000))  # Convert to int
                if len(data["data"]) < limit:
                    break
                current_start = data["data"][-1][0] + 1
            else:
                print(f"{source_type}: Empty or invalid data")
                break
        else:
            print(f"{source_type} fetch failed: {response.status_code}")
            break
        time.sleep(1)
    
    if all_data:
        df = pd.DataFrame(all_data)
        if source_type == 'cryptoquant':
            df['datetime'] = pd.to_datetime(df['start_time'], unit='ms')
            df = df.set_index('datetime').sort_index()
        elif source_type == 'glassnode':
            df['datetime'] = pd.to_datetime(df['start_time'], unit='ms')
            df = df.set_index('datetime').sort_index()
            df = df[['v']].rename(columns={'v': 'utxo_created_value_median'})
        elif source_type == 'coinglass':
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('datetime').sort_index()
            df = df[['open', 'high', 'low', 'close']].add_prefix('oi_')
        print(f"{source_type} data shape: {df.shape}")
        return df
    print(f"{source_type}: No data fetched")
    return pd.DataFrame()

def fetch_ohlcv(symbol, timeframe=TIMEFRAME):
    exchange = ccxt.binance()
    since = exchange.parse8601((datetime.now() - timedelta(days=YEARS_OF_DATA*365)).isoformat())
    all_ohlcv = []
    limit = 1000
    last_timestamp = int(since)
    
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, last_timestamp, limit)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        last_timestamp = ohlcv[-1][0] + 1
        if len(ohlcv) < limit:
            break
        time.sleep(1)
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def collect_all_data():
    btc_ohlcv = fetch_ohlcv('BTC/USDT')
    eth_ohlcv = fetch_ohlcv('ETH/USDT')
    print(f"BTC OHLCV shape: {btc_ohlcv.shape}")
    print(f"ETH OHLCV shape: {eth_ohlcv.shape}")
    
    headers = {'X-API-Key': API_KEY}
    cryptoquant_params = {"exchange": EXCHANGE, "window": "hour", "limit": 10000}  # Integer limit
    glassnode_params = {"a": "BTC", "c": "usd", "i": "1h", "limit": 1000, "flatten": False}
    coinglass_params = {"exchange": "Binance", "symbol": "BTCUSDT", "interval": "1h", "limit": 1000}
    
    # Fetch data
    cryptoquant_data = {
        'inflow': fetch_data_from_api(f"{CRYPTOQUANT_BASE_URL}/btc/exchange-flows/inflow", cryptoquant_params, headers, 'cryptoquant'),
        'outflow': fetch_data_from_api(f"{CRYPTOQUANT_BASE_URL}/btc/exchange-flows/outflow", cryptoquant_params, headers, 'cryptoquant'),
        'netflow': fetch_data_from_api(f"{CRYPTOQUANT_BASE_URL}/btc/exchange-flows/netflow", cryptoquant_params, headers, 'cryptoquant')
    }
    glassnode_data = fetch_data_from_api(GLASSNODE_URL, glassnode_params, headers, 'glassnode')
    coinglass_data = fetch_data_from_api(COINGLASS_URL, coinglass_params, headers, 'coinglass')
    
    btc_data = btc_ohlcv.set_index('datetime')
    for metric, df in cryptoquant_data.items():
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            df_resampled = df.resample('1h').last().ffill()
            col_prefix = metric
            df_resampled = df_resampled.add_prefix(f'{col_prefix}_')
            btc_data = btc_data.merge(df_resampled, left_index=True, right_index=True, how='left')
    
    if not glassnode_data.empty and isinstance(glassnode_data.index, pd.DatetimeIndex):
        glassnode_resampled = glassnode_data.resample('1h').last().ffill()
        btc_data = btc_data.merge(glassnode_resampled, left_index=True, right_index=True, how='left')
    
    if not coinglass_data.empty and isinstance(coinglass_data.index, pd.DatetimeIndex):
        coinglass_resampled = coinglass_data.resample('1h').last().ffill()
        btc_data = btc_data.merge(coinglass_resampled, left_index=True, right_index=True, how='left')
    
    eth_data = eth_ohlcv.set_index('datetime')
    
    print(f"Merged BTC data shape: {btc_data.shape}")
    print(f"ETH data shape: {eth_data.shape}")
    return btc_data, eth_data

def clean_and_enhance_data(df):
    df = df.drop_duplicates()
    df = df.sort_index()
    df = df.ffill()
    df = df.dropna(subset=['close'])
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(24).std()
    df['volume_ma'] = df['volume'].rolling(24).mean()
    if 'inflow_total' in df.columns:
        df['netflow_ratio'] = df['inflow_total'] / (df['outflow_total'] + 1e-6)
    lookahead = 6
    df['target'] = (df['close'].shift(-lookahead) > df['close']).astype(int)
    df = df.dropna(subset=['target'])
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    return df

def validate_data(df):
    time_diffs = df.index.diff()
    time_gaps = time_diffs.total_seconds() / 3600
    if any(time_gaps > 1.1):
        print(f"Warning: Time gaps detected - max {time_gaps.max():.1f} hours")
    if len(df) < 365 * 24 * 0.8:
        raise ValueError("Insufficient data points")
    if 'returns' in df.columns:
        extreme_returns = (df['returns'].abs() > 0.1).sum()
        if extreme_returns > len(df) * 0.01:
            print(f"Warning: {extreme_returns} extreme returns detected")
    return True

def build_ml_ready_dataset():
    btc_data, eth_data = collect_all_data()
    btc_enhanced = clean_and_enhance_data(btc_data)
    eth_enhanced = clean_and_enhance_data(eth_data)
    validate_data(btc_enhanced)
    validate_data(eth_enhanced)
    btc_enhanced.to_csv('btc_ml_ready.csv', index=True)
    eth_enhanced.to_csv('eth_ml_ready.csv', index=True)
    print(f"Data collection complete. BTC shape: {btc_enhanced.shape}, ETH shape: {eth_enhanced.shape}")
    return btc_enhanced, eth_enhanced

if __name__ == "__main__":
    btc_df, eth_df = build_ml_ready_dataset()