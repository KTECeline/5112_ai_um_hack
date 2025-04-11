
##got error
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
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta, timezone

# ============== 🔐 API KEYS =================
API_KEY = os.getenv('CRYPTOQUANT_API_KEY')
ETHERSCAN_API_KEY = os.getenv('ETHERSCAN_API_KEY') or "8NK92WM2VMXEVT9P8GMGZWNSNMK31II4XW"
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
if not BEARER_TOKEN:
    print("⚠️ Warning: Twitter Bearer Token not set.")
headers = {'X-API-Key': API_KEY}

# Twitter Client Setup
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True) if BEARER_TOKEN else None
analyzer = SentimentIntensityAnalyzer()

# ============== 📅 Parameters =================
start_time = str(int(pd.Timestamp.now().timestamp() * 1000) - 90 * 24 * 60 * 60 * 1000)  # 90 days

# ============== 🌐 API URLs & Params =================
cryptoquant_url = "https://api.datasource.cybotrade.rs/cryptoquant/btc/exchange-flows/inflow"
glassnode_url = "https://api.datasource.cybotrade.rs/glassnode/blockchain/utxo_created_value_median"
coinglass_url = "https://api.datasource.cybotrade.rs/coinglass/futures/openInterest/ohlc-history"
binance_ohlc_url = "https://api.binance.com/api/v3/klines"
etherscan_url = "https://api.etherscan.io/api"
coingecko_url = "https://api.coingecko.com/api/v3/coins/{coin}/market_chart"

params = {
    'cryptoquant': {"exchange": "okx", "window": "hour", "start_time": start_time, "limit": "1000"},
    'glassnode': {"a": "BTC", "c": "usd", "i": "1h", "start_time": int(start_time), "limit": 1000, "flatten": False},
    'coinglass': {"exchange": "Binance", "symbol": "BTCUSDT", "interval": "1h", "start_time": start_time, "limit": "1000"},
    'binance': {"symbol": "BTCUSDT", "interval": "1h", "startTime": start_time, "limit": 1000},
    'etherscan': {"module": "proxy", "action": "eth_getBlockByNumber", "boolean": "true", "apikey": ETHERSCAN_API_KEY},
    'coingecko': {"vs_currency": "usd", "days": "90"}
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
            response = session.get(url.format(coin=params.get('coin', '')), headers=headers, params=params, timeout=timeout)
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
def fetch_coingecko_data(coin="bitcoin"):
    params['coingecko']['coin'] = coin
    print(f"\n⏳ Fetching {coin.upper()} price and volume data from CoinGecko (hourly)...")
    data = fetch_data(coingecko_url, params['coingecko'])
    if data and 'prices' in data and 'total_volumes' in data:
        if coin == "bitcoin":
            price_col = 'btc_close_price'
            volume_col = 'btc_volume'
        elif coin == "ethereum":
            price_col = 'eth_close_price'
            volume_col = 'eth_volume'
        else:
            price_col = f'{coin[:3]}_close_price'
            volume_col = f'{coin[:3]}_volume'
        
        df = pd.DataFrame(data['prices'], columns=['timestamp', price_col])
        df[volume_col] = [v[1] for v in data['total_volumes']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        print(f"✅ CoinGecko {coin.upper()} DataFrame columns: {list(df.columns)}")
        return df
    print(f"⚠️ Warning: Failed to fetch CoinGecko {coin} data.")
    return None

# ============== 🔁 Helper: Fetch Etherscan ETH Volume =================
def fetch_etherscan_eth_volume(min_value_eth=100, max_blocks=2160):
    print("\n⏳ Fetching large ETH transactions from Etherscan...")
    latest_block = fetch_data(
        etherscan_url,
        {
            "module": "proxy",
            "action": "eth_blockNumber",
            "apikey": ETHERSCAN_API_KEY
        },
        timeout=30
    )
    if not latest_block or 'result' not in latest_block:
        print("⚠️ Warning: Failed to fetch latest block number.")
        return pd.DataFrame(columns=['timestamp', 'tx_hash', 'from', 'to', 'eth_value'])

    try:
        latest_block_num = int(latest_block['result'], 16)
    except (ValueError, TypeError) as e:
        print(f"❌ Error parsing block number: {e}")
        return pd.DataFrame(columns=['timestamp', 'tx_hash', 'from', 'to', 'eth_value'])

    start_block = max(0, latest_block_num - max_blocks)
    tx_data = []
    batch_size = 100
    for block in range(start_block, latest_block_num + 1, batch_size):
        block_hex = hex(block)
        params = {
            "module": "proxy",
            "action": "eth_getBlockByNumber",
            "tag": block_hex,
            "boolean": "true",
            "apikey": ETHERSCAN_API_KEY
        }
        block_data = fetch_data(etherscan_url, params, timeout=30)
        if block_data and 'result' in block_data and block_data['result']:
            try:
                block_result = block_data['result']
                txs = block_result.get('transactions', [])
                block_time = pd.to_datetime(int(block_result['timestamp'], 16), unit='s')
                for tx in txs:
                    if 'value' in tx:
                        eth_value = int(tx['value'], 16) / 1e18
                        if eth_value >= min_value_eth:
                            print(f"🚨 Whale alert: {eth_value:.2f} ETH from {tx['from']} to {tx['to']} at {block_time}")
                            tx_data.append({
                                'timestamp': block_time,
                                'tx_hash': tx['hash'],
                                'from': tx.get('from', ''),
                                'to': tx.get('to', ''),
                                'eth_value': eth_value
                            })
            except (KeyError, ValueError) as e:
                print(f"⚠️ Warning: Error processing block {block}: {e}")
        else:
            print(f"⚠️ Warning: No data for block {block}")
        time.sleep(0.25)

    if not tx_data:
        print("⚠️ Warning: No large transactions found.")
        return pd.DataFrame(columns=['timestamp', 'tx_hash', 'from', 'to', 'eth_value'])

    df = pd.DataFrame(tx_data)
    df.to_csv("whale_transactions.csv", index=False)
    print(f"✅ Etherscan large transactions DataFrame: {len(df)} transactions, columns: {list(df.columns)}")
    return df

# ============== 🔁 Helper: Fetch Twitter Data =================
def fetch_twitter_data():
    if not client:
        print("⚠️ Twitter client not initialized. Skipping Twitter data.")
        return pd.DataFrame({'timestamp': [], 'avg_sentiment': [], 'sentiment_std': [], 'tweet_volume': []})
    
    query = (
        '(crypto OR bitcoin OR ethereum OR "digital assets" OR blockchain OR defi OR nft '
        'OR "crypto news" OR "crypto regulation" OR "crypto ETF" OR "crypto crash" OR "crypto bull" '
        'OR "SEC crypto" OR "market sentiment" OR "whale trade" OR bybit OR binance OR coinbase OR "Elon Musk" '
        'OR "Fed rate" OR "inflation data" OR "interest rates" OR "CPI report" OR "market correction" '
        'OR "bull market" OR "bear market" OR "liquidity" OR "technical analysis" OR "on-chain data") '
        '-is:retweet lang:en'
    )
    end_time = datetime.now(timezone.utc) - timedelta(seconds=10)
    start_time = end_time - timedelta(days=7)
    
    print("\n⏳ Fetching Twitter data for the last 7 days...")
    sentiment_scores = []
    tweet_times = []
    
    try:
        paginator = tweepy.Paginator(
            client.search_recent_tweets,
            query=query,
            max_results=100,
            tweet_fields=["created_at", "text"],
            start_time=start_time,
            end_time=end_time
        )
        for response in paginator:
            tweets = response.data or []
            for tweet in tweets:
                score = analyzer.polarity_scores(tweet.text)['compound']
                tweet_times.append(tweet.created_at)
                sentiment_scores.append(score)
        
        if tweet_times:
            df = pd.DataFrame({'timestamp': tweet_times, 'sentiment': sentiment_scores})
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.resample('h', on='timestamp').agg({
                'sentiment': ['mean', 'std', 'count']
            }).reset_index()
            df.columns = ['timestamp', 'avg_sentiment', 'sentiment_std', 'tweet_volume']
            df['sentiment_std'] = df['sentiment_std'].fillna(0)
            print(f"✅ Twitter DataFrame columns: {list(df.columns)}")
            return df
        return pd.DataFrame({'timestamp': [], 'avg_sentiment': [], 'sentiment_std': [], 'tweet_volume': []})
    
    except Exception as e:
        print(f"❌ Error fetching Twitter data: {e}")
        return pd.DataFrame({'timestamp': [], 'avg_sentiment': [], 'sentiment_std': [], 'tweet_volume': []})

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

dfs = {}
dfs['btc'] = fetch_coingecko_data("bitcoin")
dfs['eth'] = fetch_coingecko_data("ethereum")
dfs['eth_tx'] = fetch_etherscan_eth_volume(min_value_eth=100, max_blocks=2160)
dfs['twitter'] = fetch_twitter_data()

print("\n🔄 Testing Binance connectivity...")
if not test_binance_connectivity():
    print("⚠️ Warning: Cannot connect to Binance API. Using CoinGecko data.")
else:
    binance_data = fetch_data(binance_ohlc_url, params['binance'])
    if binance_data:
        dfs['binance'] = pd.DataFrame(binance_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        dfs['binance']['timestamp'] = pd.to_datetime(dfs['binance']['timestamp'], unit='ms')
        dfs['binance'] = dfs['binance'][['timestamp', 'close']].rename(columns={'close': 'btc_close_price'})

# Convert other data to DataFrames
if cryptoquant_data and 'data' in cryptoquant_data:
    dfs['cryptoquant'] = pd.DataFrame(cryptoquant_data['data'])
    dfs['cryptoquant']['timestamp'] = pd.to_datetime(dfs['cryptoquant']['start_time'], unit='ms')
    dfs['cryptoquant'] = dfs['cryptoquant'].drop(columns=['start_time'], errors='ignore')

if glassnode_data and 'data' in glassnode_data:
    dfs['glassnode'] = pd.DataFrame(glassnode_data['data'])
    dfs['glassnode']['timestamp'] = pd.to_datetime(dfs['glassnode']['start_time'], unit='ms')
    dfs['glassnode'] = dfs['glassnode'].rename(columns={'v': 'value'}).drop(columns=['start_time'], errors='ignore')

if coinglass_data and 'data' in coinglass_data:
    dfs['coinglass'] = pd.DataFrame(coinglass_data['data'])
    dfs['coinglass']['timestamp'] = pd.to_datetime(dfs['coinglass']['start_time'], unit='ms')
    dfs['coinglass'] = dfs['coinglass'].drop(columns=['start_time'], errors='ignore')

# Aggregate eth_tx to hourly volume
if not dfs['eth_tx'].empty:
    dfs['eth_tx'] = dfs['eth_tx'].set_index('timestamp').resample('h').sum(numeric_only=True).reset_index()
    dfs['eth_tx'] = dfs['eth_tx'].rename(columns={'eth_value': 'eth_tx_volume'})
    dfs['eth_tx']['eth_tx_volume'] = dfs['eth_tx']['eth_tx_volume'].fillna(0)
else:
    start_ts = pd.to_datetime("2025-01-11")
    end_ts = pd.to_datetime("2025-04-11")
    timestamps = pd.date_range(start=start_ts, end=end_ts, freq='h')
    dfs['eth_tx'] = pd.DataFrame({'timestamp': timestamps, 'eth_tx_volume': 0})

# Merge DataFrames
combined_df = None
for key, df in dfs.items():
    if df is not None and not df.empty:
        print(f"📊 {key} DataFrame shape: {df.shape}, timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Columns in {key}: {list(df.columns)}")
        df = df.set_index('timestamp').resample('h').mean(numeric_only=True).reset_index()
        if combined_df is None:
            combined_df = df
        else:
            combined_df = combined_df.merge(df, on='timestamp', how='outer')
    else:
        print(f"⚠️ Warning: {key} DataFrame is empty or None.")

fill_columns = ['btc_close_price', 'btc_volume', 'eth_close_price', 'eth_volume', 'inflow_total', 'value']
combined_df[fill_columns] = combined_df[fill_columns].ffill()
combined_df['eth_tx_volume'] = combined_df['eth_tx_volume'].fillna(0)

combined_df['datetime'] = combined_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
print(f"✅ Final combined_df columns: {list(combined_df.columns)}, shape: {combined_df.shape}")

# Feature Engineering
print("\n🧽 Feature Engineering...")
print(f"Columns before feature engineering: {list(combined_df.columns)}")
combined_df['btc_close_price'] = combined_df['btc_close_price'].ffill()

base_features = ['inflow_total', 'value', 'eth_tx_volume', 'btc_volume', 'btc_close_price', 'price_change', 'sma_20', 'rsi_14']
twitter_features = ['avg_sentiment', 'sentiment_std', 'tweet_volume']
features_list = [f for f in base_features if f in combined_df.columns] + [f for f in twitter_features if f in combined_df.columns]

if 'btc_close_price' in combined_df.columns:
    combined_df['price_change'] = combined_df['btc_close_price'].pct_change()
    combined_df['sma_20'] = combined_df['btc_close_price'].rolling(window=20).mean()
    diff = combined_df['btc_close_price'].diff()
    gain = diff.where(diff > 0, 0).rolling(window=14).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    combined_df['rsi_14'] = 100 - (100 / (1 + rs))

combined_df = combined_df.dropna(subset=['btc_close_price', 'btc_volume'])
print(f"✅ Selected features: {features_list}")
print(f"📊 combined_df shape after dropna: {combined_df.shape}")
print("📈 Feature statistics:")
print(combined_df[features_list].describe())

# HMM Training
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
plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(combined_df['timestamp'], combined_df['eth_tx_volume'], label='ETH Tx Volume')
plt.plot(combined_df['timestamp'], combined_df['btc_volume'], label='BTC Volume')
plt.legend()
plt.subplot(2, 1, 2)
plt.scatter(combined_df['timestamp'], combined_df['hidden_states'], c=combined_df['hidden_states'], cmap='viridis', label='Hidden States')
plt.title('HMM Hidden States')
plt.legend()
plt.tight_layout()
plt.show()

# Plot whale transactions
if not dfs['eth_tx'].empty:
    plt.figure(figsize=(15, 6))
    plt.scatter(dfs['eth_tx']['timestamp'], dfs['eth_tx']['eth_tx_volume'], c='blue', label='Large ETH Transfers')
    plt.title('Whale ETH Transfers (>100 ETH)')
    plt.xlabel('Time')
    plt.ylabel('ETH Volume')
    plt.legend()
    plt.show()

# Signal Generation
if 'btc_close_price' in combined_df.columns and len(combined_df) >= 20:
    threshold = 0.005
    combined_df['signal'] = np.where(combined_df['price_change'].shift(-1) > threshold, 'buy',
                                    np.where(combined_df['price_change'].shift(-1) < -threshold, 'sell', 'hold'))
else:
    print("⚠️ Warning: No btc_close_price available or insufficient data. Skipping signal generation and backtesting.")
    combined_df['signal'] = 'hold'
    portfolio_df = pd.DataFrame({'Equity': [10000] * len(combined_df), 'Positions': [0] * len(combined_df), 'Time': combined_df['timestamp']})
    trade_log = pd.DataFrame()
    total_fees = 0

# Backtesting
def backtest(df, initial_cash=10000, fee_rate=0.0006):
    portfolio = {'cash': initial_cash, 'assets': 0, 'value': [], 'positions': [], 'pnl': []}
    trade_log = []
    total_fees = 0
    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        price = row['btc_close_price']
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
            prev_price = df['btc_close_price'].iloc[max(0, i-1)] if i > 0 else price
            portfolio['pnl'].append(cash_gained - (portfolio['assets'] * prev_price))
            portfolio['cash'] += cash_gained - fee
            portfolio['assets'] = 0
            trade_log.append({'time': row['timestamp'], 'signal': 'sell', 'price': price, 'fee': fee})
        portfolio['value'].append(portfolio['cash'] + portfolio['assets'] * price)
        portfolio['positions'].append(portfolio['assets'])
    portfolio_df = pd.DataFrame({
        'Equity': portfolio['value'],
        'Positions': portfolio['positions'],
        'Close_price': df['btc_close_price'],
        'Time': df['timestamp']
    }, index=df.index)
    trade_log_df = pd.DataFrame(trade_log)
    return portfolio_df, trade_log_df, total_fees

if 'btc_close_price' in combined_df.columns and len(combined_df) >= 20:
    portfolio_df, trade_log, total_fees = backtest(combined_df)
else:
    portfolio_df = pd.DataFrame({
        'Equity': [10000] * len(combined_df),
        'Positions': [0] * len(combined_df),
        'Time': combined_df['timestamp']
    })
    trade_log = pd.DataFrame()
    total_fees = 0

# Metrics Calculation
metrics = {
    'Window': '2025-01-11 to 2025-04-11',
    'Threshold': 0.005,
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
    'Close_price': portfolio_df['Close_price'].iloc[-1] if 'Close_price' in portfolio_df.columns else (combined_df['btc_close_price'].iloc[-1] if 'btc_close_price' in combined_df.columns else 0),
    'Time': combined_df['timestamp'].iloc[-1]
}

if 'btc_close_price' in combined_df.columns:
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

# Export
combined_df.to_csv("combined_crypto_data_with_hmm.csv", index=False)
pd.DataFrame([metrics]).to_csv("backtest_metrics.csv", index=False)
print("\n✅ Exported to CSV files.")
print(metrics)

# Plot
if 'btc_close_price' in combined_df.columns:
    plt.figure(figsize=(15, 6))
    plt.plot(portfolio_df['Time'], portfolio_df['Equity'], label='Equity')
    if not trade_log.empty:
        plt.scatter(trade_log['time'], trade_log['price'], c=trade_log['signal'].map({'buy': 'g', 'sell': 'r'}), label='Trades')
    else:
        print("⚠️ No trades occurred. Skipping trade scatter plot.")
    plt.title('Portfolio Equity and Trades (BTC)')
    plt.legend()
    plt.show()
else:
    print("⚠️ Warning: No btc_close_price available. Skipping plot.")