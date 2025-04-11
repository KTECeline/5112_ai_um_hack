import pandas as pd
import numpy as np
import requests
import json
import os
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import time

# ============== 🔐 API KEY =================
API_KEY = os.getenv('CRYPTOQUANT_API_KEY')
if not API_KEY:
    raise ValueError("❌ CRYPTOQUANT_API_KEY not set.")
headers = {'X-API-Key': API_KEY}

# ============== 📅 Parameters =================
start_time = int(datetime(2022, 10, 1).timestamp() * 1000)  # October 1, 2022
end_time = int(datetime(2025, 4, 10).timestamp() * 1000)   # April 10, 2025
symbols = ['BTCUSDT', 'ETHUSDT']
timeframe = '1h'
limit_per_call = 1000

# ============== 🌐 API URLs & Params =================
cryptoquant_inflow_url = "https://api.datasource.cybotrade.rs/cryptoquant/{coin}/exchange-flows/inflow"
cryptoquant_outflow_url = "https://api.datasource.cybotrade.rs/cryptoquant/{coin}/exchange-flows/outflow"
glassnode_url = "https://api.datasource.cybotrade.rs/glassnode/blockchain/utxo_created_value_median"
coinglass_url = "https://api.datasource.cybotrade.rs/coinglass/futures/openInterest/ohlc-history"
kraken_ohlc_url = "https://api.kraken.com/0/public/OHLC"

params = {
    'cryptoquant': {
        "exchange": "okx",
        "window": "hour",
        "limit": str(limit_per_call)
    },
    'glassnode': {
        "a": "{coin}",
        "c": "usd",
        "i": "1h",
        "limit": limit_per_call,
        "flatten": False
    },
    'coinglass': {
        "exchange": "Binance",
        "symbol": "{symbol}",
        "interval": "1h",
        "limit": str(limit_per_call)
    },
    'kraken': {
        "pair": "{pair}",
        "interval": 60,  # 1 hour in minutes
        "since": "{since}"
    }
}

# ============== 🔁 Helper: Fetch Data =================
def fetch_data(url, params, headers=None, max_retries=3, timeout=30):
    for attempt in range(max_retries):
        try:
            print(f"⏳ Fetching data from {url} (Attempt {attempt + 1}/{max_retries})...")
            response = requests.get(url, headers=headers if headers else None, params=params, timeout=timeout)
            print(f"Request URL: {response.url}")
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if 'result' in data:  # Kraken
                    pair = params.get('pair', list(data['result'].keys())[0])
                    sample = data['result'].get(pair, [])[:2]
                elif 'data' in data:  # CryptoQuant, Glassnode, Coinglass
                    sample = data['data'][:2]
                else:
                    sample = data[:2] if isinstance(data, list) else data
                print(f"Raw response sample: {json.dumps(sample, indent=2)}")
                return data
            else:
                print(f"Response Body: {response.text}")
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching data: {e}")
        time.sleep(1)
    print(f"❌ Failed to fetch data from {url} after {max_retries} attempts.")
    return None

# ============== 🔁 Helper: Fetch Kraken OHLC =================
def fetch_kraken_ohlc(pair, start_time, end_time):
    print(f"⏳ Fetching OHLC from Kraken for {pair}...")
    kraken_dfs = []
    current_time = start_time // 1000  # Kraken uses seconds
    kraken_params = params['kraken'].copy()
    kraken_params['pair'] = pair
    
    while current_time * 1000 < end_time:
        kraken_params['since'] = str(current_time)
        data = fetch_data(kraken_ohlc_url, kraken_params, headers=None)
        if not data or 'result' not in data or pair not in data['result']:
            print(f"⚠️ Stopping Kraken OHLC fetch for {pair} at {current_time}")
            break
        ohlc_data = data['result'][pair]
        if not ohlc_data:
            print(f"⚠️ No more OHLC data for {pair} at {current_time}")
            break
        df = pd.DataFrame(ohlc_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype({
            'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
        })
        kraken_dfs.append(df)
        current_time = int(df['timestamp'].iloc[-1].timestamp()) + 3600  # Next hour after last timestamp
        print(f"Fetched {len(df)} rows, new current_time: {current_time}")
        time.sleep(0.5)
    
    if not kraken_dfs:
        return None
    df = pd.concat(kraken_dfs).drop_duplicates(subset=['timestamp'])
    df = df[(df['timestamp'] >= pd.to_datetime(start_time, unit='ms')) & 
            (df['timestamp'] <= pd.to_datetime(end_time, unit='ms'))]
    print(f"✅ Fetched {len(df)} rows from Kraken for {pair}")
    return df

# ============== 🧼 Clean and Enhance Data =================
def clean_and_enhance_data(df):
    print("\n🧼 Cleaning and enhancing data...")
    df = df.drop_duplicates(subset=['timestamp'])
    df = df.sort_values('timestamp')
    df = df.ffill().bfill()
    df = df.dropna(subset=['close'])
    
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=24).std()
    df['volume_ma'] = df['volume'].rolling(window=24).mean() if 'volume' in df else np.nan
    df['netflow_ratio'] = df['inflow'].astype(float) / df['outflow'].astype(float).replace(0, np.nan) if 'inflow' in df and 'outflow' in df else np.nan
    df['future_close'] = df['close'].shift(-6)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df = df.dropna(subset=['close', 'open', 'high', 'low'])
    
    print("✅ Data cleaned and enhanced.")
    return df

# ============== ✅ Validate Data =================
def validate_data(df, symbol):
    print(f"\n✅ Validating data for {symbol}...")
    time_diff = df['timestamp'].diff().dt.total_seconds() / 3600
    gaps = time_diff[time_diff > 1.0]
    if len(gaps) > 0:
        print(f"⚠️ Found {len(gaps)} time gaps larger than 1 hour.")
    
    expected_points = 3 * 365 * 24 * 0.8
    if len(df) < expected_points:
        print(f"⚠️ Insufficient data points: {len(df)} (expected ~{int(expected_points)})")
    else:
        print(f"✅ Sufficient data points: {len(df)}")
    
    extreme_returns = df['returns'].abs() > 0.1
    if extreme_returns.sum() > 0:
        print(f"⚠️ Found {extreme_returns.sum()} extreme returns (>10% hourly).")
    
    print("✅ Validation complete.")

# ============== 🧲 Collect All Data for a Symbol =================
def collect_all_data(symbol):
    coin_short = 'btc' if 'BTC' in symbol else 'eth'
    kraken_pair = 'XXBTZUSD' if 'BTC' in symbol else 'XETHZUSD'
    
    print(f"\n🔄 Collecting data for {symbol}...")
    
    # Fetch Kraken OHLC
    ohlc_df = fetch_kraken_ohlc(kraken_pair, start_time, end_time)
    if ohlc_df is None or ohlc_df.empty:
        print(f"❌ Failed to fetch OHLC for {symbol} from Kraken")
        return pd.DataFrame()
    
    # Fetch CryptoQuant Inflow
    inflow_dfs = []
    current_time = start_time
    cryptoquant_params = params['cryptoquant'].copy()
    while current_time < end_time:
        cryptoquant_params['start_time'] = str(current_time)
        inflow_url = cryptoquant_inflow_url.format(coin=coin_short)
        inflow_data = fetch_data(inflow_url, cryptoquant_params, headers)
        if not inflow_data or 'data' not in inflow_data:
            print(f"⚠️ Stopping CryptoQuant inflow for {coin_short} at {current_time}")
            break
        inflow_df = pd.DataFrame(inflow_data['data'])
        inflow_df['timestamp'] = pd.to_datetime(inflow_df['start_time'], unit='ms')
        inflow_df = inflow_df[['timestamp', 'inflow_total']].rename(columns={'inflow_total': 'inflow'})
        inflow_dfs.append(inflow_df)
        current_time = int(inflow_df['timestamp'].iloc[-1].timestamp() * 1000) + 3600 * 1000
        time.sleep(0.5)
    
    inflow_df = pd.concat(inflow_dfs).drop_duplicates(subset=['timestamp']) if inflow_dfs else pd.DataFrame()
    
    # Fetch CryptoQuant Outflow
    outflow_dfs = []
    current_time = start_time
    while current_time < end_time:
        cryptoquant_params['start_time'] = str(current_time)
        outflow_url = cryptoquant_outflow_url.format(coin=coin_short)
        outflow_data = fetch_data(outflow_url, cryptoquant_params, headers)
        if not outflow_data or 'data' not in outflow_data:
            print(f"⚠️ Stopping CryptoQuant outflow for {coin_short} at {current_time}")
            break
        outflow_df = pd.DataFrame(outflow_data['data'])
        outflow_df['timestamp'] = pd.to_datetime(outflow_df['start_time'], unit='ms')
        outflow_df = outflow_df[['timestamp', 'outflow_total']].rename(columns={'outflow_total': 'outflow'})
        outflow_dfs.append(outflow_df)
        current_time = int(outflow_df['timestamp'].iloc[-1].timestamp() * 1000) + 3600 * 1000
        time.sleep(0.5)
    
    outflow_df = pd.concat(outflow_dfs).drop_duplicates(subset=['timestamp']) if outflow_dfs else pd.DataFrame()
    
    # Fetch Glassnode (BTC only)
    glassnode_df = pd.DataFrame()
    if coin_short == 'btc':
        glassnode_dfs = []
    current_time = start_time
    glassnode_params = params['glassnode'].copy()
    glassnode_params['a'] = coin_short.upper()
    while current_time < end_time:
        glassnode_params['start_time'] = current_time
        glassnode_data = fetch_data(glassnode_url, glassnode_params, headers)
        if not glassnode_data or 'data' not in glassnode_data:
            print(f"⚠️ Stopping Glassnode data for {coin_short} at {current_time}")
            break
        glassnode_df = pd.DataFrame(glassnode_data['data'])
        glassnode_df['timestamp'] = pd.to_datetime(glassnode_df['start_time'], unit='ms')
        glassnode_df = glassnode_df[['timestamp', 'v']].rename(columns={'v': 'utxo_created_value_median'})
        glassnode_dfs.append(glassnode_df)
        current_time = int(glassnode_df['timestamp'].iloc[-1].timestamp() * 1000) + 3600 * 1000
        time.sleep(0.5)
    glassnode_df = pd.concat(glassnode_dfs).drop_duplicates(subset=['timestamp']) if glassnode_dfs else pd.DataFrame()
    # Fetch Coinglass (skip due to consistent failure)
    
    coinglass_df = pd.DataFrame()
    # Uncomment and fix if Coinglass is needed
    # current_time = start_time
    # coinglass_params = params['coinglass'].copy()
    # coinglass_params['symbol'] = symbol
    # while current_time < end_time:
    #     coinglass_params['start_time'] = str(current_time)
    #     coinglass_data = fetch_data(coinglass_url, coinglass_params, headers)
    #     if not coinglass_data or 'data' not in coinglass_data:
    #         print(f"⚠️ Stopping Coinglass data for {symbol} at {current_time}")
    #         break
    #     coinglass_df = pd.DataFrame(coinglass_data['data'])
    #     coinglass_df['timestamp'] = pd.to_datetime(coinglass_df['start_time'], unit='ms')
    #     coinglass_df = coinglass_df[['timestamp', 'open', 'high', 'low', 'close']].rename(
    #         columns={'open': 'oi_open', 'high': 'oi_high', 'low': 'oi_low', 'close': 'oi_close'}
    #     )
    #     coinglass_dfs.append(coinglass_df)
    #     current_time = int(coinglass_df['timestamp'].iloc[-1].timestamp() * 1000) + 3600 * 1000
    #     time.sleep(0.5)
    # coinglass_df = pd.concat(coinglass_dfs).drop_duplicates(subset=['timestamp']) if coinglass_dfs else pd.DataFrame()
    
    # Merge all data
    print("\n🔗 Merging datasets...")
    combined_df = ohlc_df
    for df in [inflow_df, outflow_df, glassnode_df, coinglass_df]:
        if not df.empty:
            combined_df = combined_df.merge(df, on='timestamp', how='left')
    
    combined_df['datetime'] = combined_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    print(f"✅ Merged data for {symbol}: {len(combined_df)} rows")
    return combined_df

# ============== 🧠 Train HMM =================
def train_hmm(df):
    print("\n🧠 Training the Hidden Markov Model (HMM)...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    features = df[numeric_columns].dropna()  # Drop rows with any NaN
    if features.empty:
        print("❌ No valid data for HMM after dropping NaN.")
        return df
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
    model.fit(features_scaled)
    df.loc[features.index, 'hidden_states'] = model.predict(features_scaled)
    
    log_likelihood = model.score(features_scaled)
    print(f"✅ Log-Likelihood: {log_likelihood}")
    aic = -2 * log_likelihood + 2 * len(features_scaled)  # Corrected AIC formula
    print(f"✅ AIC: {aic}")
    bic = -2 * log_likelihood + np.log(len(features_scaled)) * len(features_scaled[0])  # Corrected BIC
    print(f"✅ BIC: {bic}")
    
    print("✅ HMM trained.")
    return df

# ============== 📈 Plot Optional =================
def plot_data(df, symbol, feature_name='close'):
    print(f"\n📈 Plotting {feature_name} with Hidden States for {symbol}...")
    plt.figure(figsize=(15, 6))
    plt.plot(df['timestamp'], df[feature_name], label=feature_name, alpha=0.6)
    plt.scatter(df['timestamp'], df[feature_name], c=df['hidden_states'], cmap='viridis', label='Hidden States', marker='o')
    plt.title(f'{feature_name} with Hidden States Over Time ({symbol})')
    plt.xlabel('Timestamp')
    plt.ylabel(feature_name)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{symbol}_hmm_plot.png")
    print(f"✅ Plot saved as {symbol}_hmm_plot.png")

# ============== 🚀 Main Execution =================
def main():
    for symbol in symbols:
        try:
            combined_df = collect_all_data(symbol)
            if combined_df.empty:
                print(f"❌ No data collected for {symbol}. Skipping...")
                continue
            
            enhanced_df = clean_and_enhance_data(combined_df)
            validate_data(enhanced_df, symbol)
            enhanced_df = train_hmm(enhanced_df)
            plot_data(enhanced_df, symbol)
            
            coin = 'btc' if 'BTC' in symbol else 'eth'
            output_file = f"{coin}_ml_ready.csv"
            print(f"\n💾 Saving to {output_file}...")
            enhanced_df.to_csv(output_file, index=False)
            print(f"✅ Saved {output_file}")
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")

if __name__ == "__main__":
    main()