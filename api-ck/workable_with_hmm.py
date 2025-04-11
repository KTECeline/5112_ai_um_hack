import pandas as pd
import numpy as np
import requests
import json
import os
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

# ============== 🔐 API KEY =================
API_KEY = os.getenv('CRYPTOQUANT_API_KEY')  # Replace with your key if needed
headers = {'X-API-Key': API_KEY}

# ============== 📅 Parameters =================
start_time = str(int(pd.Timestamp.now().timestamp() * 1000) - 90 * 24 * 60 * 60 * 1000)  # 90 days

# ============== 🌐 API URLs & Params =================
cryptoquant_url = "https://api.datasource.cybotrade.rs/cryptoquant/btc/exchange-flows/inflow"
glassnode_url = "https://api.datasource.cybotrade.rs/glassnode/blockchain/utxo_created_value_median"
coinglass_url = "https://api.datasource.cybotrade.rs/coinglass/futures/openInterest/ohlc-history"
binance_ohlc_url = "https://api.binance.com/api/v3/klines"

params = {
    'cryptoquant': {"exchange": "okx", "window": "hour", "start_time": start_time, "limit": "1000"},
    'glassnode': {"a": "BTC", "c": "usd", "i": "1h", "start_time": int(start_time), "limit": 1000, "flatten": False},
    'coinglass': {"exchange": "Binance", "symbol": "BTCUSDT", "interval": "1h", "start_time": start_time, "limit": "1000"},
    'binance': {"symbol": "BTCUSDT", "interval": "1h", "startTime": start_time, "limit": 1000}
}

# ============== 🔁 Helper: Fetch with Retry =================
def fetch_data(url, params=None, headers=None, retries=5, timeout=20):
    session = requests.Session()
    retry_strategy = Retry(total=retries, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    for attempt in range(retries):
        try:
            print(f"⏳ Fetching data from {url}... (Attempt {attempt + 1}/{retries})")
            response = session.get(url, headers=headers, params=params, timeout=timeout)
            print(f"Request URL: {response.url}")
            print(f"Status Code: {response.status_code}")
            if response.status_code == 429:
                print("⚠️ Rate limit hit. Retrying after delay...")
                time.sleep(2 ** attempt)
                continue
            if response.status_code != 200:
                print("Response Body:", response.text)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            continue
    return None

# ============== 🔁 Helper: Fetch CoinGecko Data =================
def fetch_coingecko_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "90"}  # Hourly data for 90 days
    print("\n⏳ Fetching fallback price data from CoinGecko (hourly)...")
    data = fetch_data(url, params)
    if data and 'prices' in data:
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'close_price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"✅ CoinGecko DataFrame columns: {list(df.columns)}")
        return df
    print("⚠️ Warning: Failed to fetch CoinGecko data.")
    return None

# ============== 🔁 Helper: Fetch Kraken Data =================
def fetch_kraken_data():
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": "XXBTZUSD", "interval": 60, "since": int(pd.Timestamp.now().timestamp() - 90 * 24 * 60 * 60)}
    print("\n⏳ Fetching fallback price data from Kraken...")
    data = fetch_data(url, params)
    if data and 'result' in data and 'XXBTZUSD' in data['result']:
        df = pd.DataFrame(data['result']['XXBTZUSD'], columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['close_price'] = df['close'].astype(float)
        df = df[['timestamp', 'close_price']]
        print(f"✅ Kraken DataFrame columns: {list(df.columns)}")
        return df
    print("⚠️ Warning: Failed to fetch Kraken data.")
    return None

# ============== 🔍 Test Binance Connectivity =================
def test_binance_connectivity():
    try:
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
        return response.status_code == 200
    except:
        return False

# ============== 🧲 Get All Data =================
print("\n🔄 Fetching data from all sources...")
cryptoquant_data = fetch_data(cryptoquant_url, params['cryptoquant'], headers)
glassnode_data = fetch_data(glassnode_url, params['glassnode'], headers)
coinglass_data = fetch_data(coinglass_url, params['coinglass'], headers)

# Debug API responses
def debug_response(data, name):
    if data and 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
        print(f"✅ {name} data sample: {list(data['data'][0].keys())}")
        if name == "Coinglass":
            print(f"✅ Coinglass sample values: {data['data'][0]}")
    else:
        print(f"⚠️ {name} data is empty or malformed: {data}")

debug_response(cryptoquant_data, "CryptoQuant")
debug_response(glassnode_data, "Glassnode")
debug_response(coinglass_data, "Coinglass")

# Initialize dfs
dfs = {}

print("\n🔄 Testing Binance connectivity...")
binance_data = None
if not test_binance_connectivity():
    print("⚠️ Warning: Cannot connect to Binance API. Trying CoinGecko as fallback...")
    coingecko_df = fetch_coingecko_data()
    if coingecko_df is not None:
        dfs['binance'] = coingecko_df
        print(f"✅ Using CoinGecko data for prices.")
    else:
        print("⚠️ Warning: CoinGecko failed. Trying Kraken as fallback...")
        kraken_df = fetch_kraken_data()
        if kraken_df is not None:
            dfs['binance'] = kraken_df
            print(f"✅ Using Kraken data for prices.")
        else:
            print("⚠️ Warning: No price data available (Binance, CoinGecko, and Kraken failed).")
else:
    print("✅ Binance API is reachable.")
    binance_data = fetch_data(binance_ohlc_url, params['binance'], None, retries=5, timeout=20)
    debug_response({'data': binance_data}, "Binance")

# Convert to DataFrames
if cryptoquant_data and 'data' in cryptoquant_data:
    cryptoquant_df = pd.DataFrame(cryptoquant_data['data'])
    if 'start_time' in cryptoquant_df.columns and 'inflow_total' in cryptoquant_df.columns:
        cryptoquant_df['timestamp'] = pd.to_datetime(cryptoquant_df['start_time'], unit='ms')
        cryptoquant_df = cryptoquant_df.drop(columns=['start_time'], errors='ignore')
        dfs['cryptoquant'] = cryptoquant_df
        print(f"✅ CryptoQuant DataFrame columns: {list(cryptoquant_df.columns)}")
    else:
        print("⚠️ Warning: CryptoQuant data missing required columns (start_time, inflow_total).")

if glassnode_data and 'data' in glassnode_data:
    glassnode_df = pd.DataFrame(glassnode_data['data'])
    if 'start_time' in glassnode_df.columns and 'v' in glassnode_df.columns:
        glassnode_df['timestamp'] = pd.to_datetime(glassnode_df['start_time'], unit='ms')
        glassnode_df = glassnode_df.drop(columns=['start_time'], errors='ignore')
        glassnode_df = glassnode_df.rename(columns={'v': 'value'})
        dfs['glassnode'] = glassnode_df
        print(f"✅ Glassnode DataFrame columns: {list(glassnode_df.columns)}")
    else:
        print("⚠️ Warning: Glassnode data missing required columns (start_time, v).")

if coinglass_data and 'data' in coinglass_data:
    coinglass_df = pd.DataFrame(coinglass_data['data'])
    if 'start_time' in coinglass_df.columns:
        coinglass_df['timestamp'] = pd.to_datetime(coinglass_df['start_time'], unit='ms')
        coinglass_df = coinglass_df.drop(columns=['start_time'], errors='ignore')
        dfs['coinglass'] = coinglass_df
        print(f"✅ Coinglass DataFrame columns: {list(coinglass_df.columns)}")
        print("⚠️ Note: 'o' in Coinglass data appears to be OHLC open price, not open interest. Skipping 'openInterest' until confirmed.")
    else:
        print("⚠️ Warning: Coinglass data missing required column (start_time).")

if binance_data:
    binance_df = pd.DataFrame(binance_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
    if 'timestamp' in binance_df.columns and 'close' in binance_df.columns:
        binance_df['timestamp'] = pd.to_datetime(binance_df['timestamp'], unit='ms')
        binance_df = binance_df[['timestamp', 'close']].rename(columns={'close': 'close_price'})
        dfs['binance'] = binance_df
        print(f"✅ Binance DataFrame columns: {list(binance_df.columns)}")
    else:
        print("⚠️ Warning: Binance data missing required columns (timestamp, close).")
elif 'binance' in dfs:
    print(f"✅ Binance DataFrame (from CoinGecko/Kraken) columns: {list(dfs['binance'].columns)}")

# Check for missing data
missing_data = []
if not cryptoquant_data:
    missing_data.append("CryptoQuant")
if not glassnode_data:
    missing_data.append("Glassnode")
if not coinglass_data:
    missing_data.append("Coinglass")
if not binance_data and 'binance' not in dfs:
    missing_data.append("Binance/CoinGecko/Kraken")

if missing_data:
    print(f"⚠️ Warning: Failed to fetch data from {', '.join(missing_data)}. Proceeding with available data.")
else:
    print("✅ All data fetched successfully.")

# Merge available DataFrames
combined_df = None
if not dfs:
    raise ValueError("❌ Error: No datasets were successfully fetched.")

for key, df in dfs.items():
    print(f"📊 {key} DataFrame shape: {df.shape}, timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    if combined_df is None:
        combined_df = df
    else:
        combined_df = combined_df.merge(df, on='timestamp', how='outer')
        print(f"📊 After merging {key} (outer), combined_df shape: {combined_df.shape}")
        combined_df = combined_df.sort_values('timestamp').ffill()
        print(f"📊 After filling {key}, combined_df shape: {combined_df.shape}")

combined_df = combined_df.sort_values('timestamp').ffill()
print(f"✅ Final combined_df columns: {list(combined_df.columns)}, shape: {combined_df.shape}")

# ============== 📅 Add Readable Date =================
combined_df['datetime'] = combined_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# ============== 🧽 Feature Engineering =================
print("\n🧽 Feature Engineering...")
combined_df['close_price'] = combined_df['close_price'].ffill()
available_columns = combined_df.columns
features_list = []
if 'inflow_total' in available_columns:
    features_list.append('inflow_total')
if 'value' in available_columns:
    features_list.append('value')
if 'openInterest' in available_columns:
    features_list.append('openInterest')

if 'close_price' in available_columns:
    combined_df['price_change'] = combined_df['close_price'].pct_change()
    combined_df['sma_20'] = combined_df['close_price'].rolling(window=20).mean()
    diff = combined_df['close_price'].diff()
    gain = diff.where(diff > 0, 0).rolling(window=14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    combined_df['rsi_14'] = 100 - (100 / (1 + rs))
    features_list.extend(['close_price', 'price_change', 'sma_20', 'rsi_14'])

if not features_list:
    raise ValueError("❌ Error: No valid features available for HMM training.")

print(f"✅ Selected features: {features_list}")
combined_df.dropna(inplace=True)
print(f"📊 combined_df shape after dropna: {combined_df.shape}")

# Log feature statistics
features_df = combined_df[features_list]
print("📈 Feature statistics:")
print(features_df.describe())

# ============== 🧠 HMM Training =================
features_list = ['inflow_total', 'value']  # Start minimal
print(f"✅ Training HMM with features: {features_list}")
features = combined_df[features_list]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=2000, random_state=42)
try:
    model.fit(features_scaled)
    combined_df['hidden_states'] = model.predict(features_scaled)
    unique_states = len(set(combined_df['hidden_states']))
    print(f"✅ HMM converged with {unique_states} unique states: {sorted(set(combined_df['hidden_states']))}")
except Exception as e:
    print(f"⚠️ Warning: HMM failed to converge: {e}. Assigning default hidden states.")
    combined_df['hidden_states'] = 0

# Plot hidden states
if 'close_price' in combined_df.columns:
    plt.figure(figsize=(15, 6))
    plt.subplot(2, 1, 1)
    plt.plot(combined_df['timestamp'], combined_df['inflow_total'], label='Inflow Total')
    plt.plot(combined_df['timestamp'], combined_df['value'], label='UTXO Value')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.scatter(combined_df['timestamp'], combined_df['hidden_states'], c=combined_df['hidden_states'], cmap='viridis', label='Hidden States')
    plt.title('HMM Hidden States')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============== 📈 Signal Generation =================
if 'close_price' in combined_df.columns and len(combined_df) > 20:
    threshold = 0.01
    combined_df['signal'] = np.where(combined_df['price_change'].shift(-1) > threshold, 'buy',
                                    np.where(combined_df['price_change'].shift(-1) < -threshold, 'sell', 'hold'))
else:
    print("⚠️ Warning: No close_price available or insufficient data. Skipping signal generation and backtesting.")
    combined_df['signal'] = 'hold'
    portfolio_df = pd.DataFrame({'Equity': [10000] * len(combined_df), 'Positions': [0] * len(combined_df), 'Time': combined_df['timestamp']})
    trade_log = pd.DataFrame()
    total_fees = 0

# ============== 📊 Backtesting =================
def backtest(df, initial_cash=10000, fee_rate=0.0006):
    portfolio = {'cash': initial_cash, 'assets': 0, 'value': [], 'positions': [], 'pnl': []}
    trade_log = []
    total_fees = 0
    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        price = row['close_price']
        signal = row['signal']
        if signal == 'buy' and portfolio['cash'] > 0:
            assets_bought = portfolio['cash'] / price
            fee = portfolio['cash'] * fee_rate
            total_fees += fee
            portfolio['cash'] -= (portfolio['cash'] + fee)
            portfolio['assets'] += assets_bought
            trade_log.append({'time': row['timestamp'], 'signal': 'buy', 'price': price, 'fee': fee})
        elif signal == 'sell' and portfolio['assets'] > 0:
            cash_gained = portfolio['assets'] * price
            fee = cash_gained * fee_rate
            total_fees += fee
            prev_price = df['close_price'].iloc[max(0, i-1)] if i > 0 else price
            portfolio['pnl'].append(cash_gained - (portfolio['assets'] * prev_price))
            portfolio['cash'] += cash_gained - fee
            portfolio['assets'] = 0
            trade_log.append({'time': row['timestamp'], 'signal': 'sell', 'price': price, 'fee': fee})
        portfolio['value'].append(portfolio['cash'] + portfolio['assets'] * price)
        portfolio['positions'].append(portfolio['assets'])
    portfolio_df = pd.DataFrame({
        'Equity': portfolio['value'],
        'Positions': portfolio['positions'],
        'Close_price': df['close_price'],
        'Time': df['timestamp']
    }, index=df.index)
    trade_log_df = pd.DataFrame(trade_log)
    return portfolio_df, trade_log_df, total_fees

if 'close_price' in combined_df.columns and len(combined_df) > 20:
    portfolio_df, trade_log, total_fees = backtest(combined_df)
else:
    portfolio_df = pd.DataFrame({'Equity': [10000] * len(combined_df), 'Positions': [0] * len(combined_df), 'Time': combined_df['timestamp']})
    trade_log = pd.DataFrame()
    total_fees = 0

# ============== 📉 Metrics Calculation =================
metrics = {
    'Window': '2024-01-11 to 2024-04-11',
    'Threshold': 0.01,
    'Sharpe': 0,
    'Max Drawdown': 0,
    'Trade/Interval': 0,
    'Fees': total_fees,
    'Drawdown': 0,
    'Equity': portfolio_df['Equity'].iloc[-1],
    'PnL': sum(portfolio_df['pnl']) if 'pnl' in portfolio_df else 0,
    'Trades': len(trade_log),
    'Positions': portfolio_df['Positions'].iloc[-1],
    'Zscore': 0,
    'Price_change': combined_df['price_change'].mean() if 'price_change' in combined_df.columns else 0,
    'Premium_index': combined_df.get('openInterest', pd.Series(0, index=combined_df.index)).mean(),
    'Close_price': portfolio_df['Close_price'].iloc[-1] if 'close_price' in combined_df.columns else 0,
    'Time': combined_df['timestamp'].iloc[-1]
}

if 'close_price' in combined_df.columns:
    daily_returns = portfolio_df['Equity'].pct_change().dropna()
    if len(daily_returns) > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        drawdown = max(1 - portfolio_df['Equity'] / portfolio_df['Equity'].cummax())
        trade_freq = len(trade_log) / len(combined_df)
        zscore = (daily_returns - daily_returns.mean()) / daily_returns.std()
        metrics.update({
            'Sharpe': sharpe,
            'Max Drawdown': -drawdown,
            'Trade/Interval': trade_freq,
            'Drawdown': -drawdown,
            'Zscore': zscore.mean()
        })

# ============== 💾 Export =================
combined_df.to_csv("combined_crypto_data_with_hmm.csv", index=False)
pd.DataFrame([metrics]).to_csv("backtest_metrics.csv", index=False)
print("\n✅ Exported to CSV files.")
print(metrics)

# ============== 📈 Plot =================
if 'close_price' in combined_df.columns:
    plt.figure(figsize=(15, 6))
    plt.plot(portfolio_df['Time'], portfolio_df['Equity'], label='Equity')
    plt.scatter(trade_log['time'], trade_log['price'], c=trade_log['signal'].map({'buy': 'g', 'sell': 'r'}), label='Trades')
    plt.title('Portfolio Equity and Trades')
    plt.legend()
    plt.show()
else:
    print("⚠️ Warning: No close_price available. Skipping plot.")