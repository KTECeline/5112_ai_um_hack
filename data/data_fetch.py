import pandas as pd
import requests
import os
import time
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class DataCollector:
    def __init__(self):
        self.api_key = os.getenv('CRYPTOQUANT_API_KEY')
        self.etherscan_api_key = os.getenv('ETHERSCAN_API_KEY') or "YOUR_ETHERSCAN_KEY"
        self.headers = {'X-API-Key': self.api_key} if self.api_key else {}
        self.start_time = str(int(pd.Timestamp.now().timestamp() * 1000) - 90 * 24 * 60 * 60 * 1000)  # 90 days
        CRYPTOQUANT_BASE_URL = "https://api.datasource.cybotrade.rs/cryptoquant"
        self.urls = {
            'cryptoquant_inflow': f"{CRYPTOQUANT_BASE_URL}/btc/exchange-flows/inflow",
            'cryptoquant_outflow': f"{CRYPTOQUANT_BASE_URL}/btc/exchange-flows/outflow",
            'cryptoquant_netflow': f"{CRYPTOQUANT_BASE_URL}/btc/exchange-flows/netflow",
            'glassnode': "https://api.datasource.cybotrade.rs/glassnode/blockchain/utxo_created_value_median",
            'etherscan': "https://api.etherscan.io/api",
            'coingecko': "https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
            'binance': "https://api.binance.com/api/v3/klines",
            'coinglass': "https://api.datasource.cybotrade.rs/coinglass/futures/openInterest/ohlc-history"
        }
        self.params = {
            'cryptoquant_inflow': {"exchange": "okx", "window": "hour", "start_time": self.start_time, "limit": "1000"},
            'cryptoquant_outflow': {"exchange": "okx", "window": "hour", "start_time": self.start_time, "limit": "1000"},
            'cryptoquant_netflow': {"exchange": "okx", "window": "hour", "start_time": self.start_time, "limit": "1000"},
            'glassnode': {"a": "BTC", "c": "usd", "i": "1h", "start_time": int(self.start_time), "limit": 1000, "flatten": False},
            'etherscan': {"module": "proxy", "action": "eth_getBlockByNumber", "boolean": "true", "apikey": self.etherscan_api_key},
            'coingecko': {"vs_currency": "usd", "days": "90"},
            'binance': {"symbol": "BTCUSDT", "interval": "1h", "startTime": self.start_time, "limit": 1000},
            'coinglass': {"exchange": "Binance", "symbol": "BTCUSDT", "interval": "1h", "start_time": self.start_time, "limit": "1000"},
        }

    def fetch_with_retry(self, url, params=None, headers=None, retries=5, timeout=20):
        session = requests.Session()
        retry_strategy = Retry(total=retries, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        for attempt in range(retries):
            try:
                print(f"⏳ Fetching data from {url}... (Attempt {attempt + 1}/{retries})")
                response = session.get(url.format(coin=params.get('coin', '') if params else ''), 
                                     headers=headers, params=params, timeout=timeout)
                print(f"Request URL: {response.url}")
                print(f"Status Code: {response.status_code}")
                if response.status_code == 429:
                    print("⚠️ Rate limit hit. Retrying after delay...")
                    time.sleep(2 ** attempt)
                    continue
                if response.status_code != 200:
                    print(f"⚠️ Non-200 response: {response.status_code}")
                    print(f"Response Body: {response.text}")
                    return None
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"❌ Error fetching data: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                continue
        print(f"⚠️ All {retries} attempts failed for {url}")
        return None

    def test_binance_connectivity(self):
        try:
            response = requests.get("https://api.binance.com/api/v3/ping", timeout=10)  # Increased timeout
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"❌ Binance connectivity test failed: {e}")
            return False

    def fetch_binance_data(self):
        print("\n⏳ Fetching BITCOIN OHLC data from Binance...")
        data = self.fetch_with_retry(self.urls['binance'], self.params['binance'])
        if data:
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.set_index('timestamp').resample('h').mean(numeric_only=True).reset_index()
            df.to_csv("data/price_data_bitcoin_binance.csv", index=False)
            print(f"✅ Saved Binance Bitcoin data to data/price_data_bitcoin_binance.csv")
            return df
        print("⚠️ Warning: Failed to fetch Binance data.")
        return pd.DataFrame()

    def fetch_coingecko_data(self, coin="bitcoin"):
        self.params['coingecko']['coin'] = coin
        print(f"\n⏳ Fetching {coin.upper()} data from CoinGecko...")
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
        params = {"vs_currency": "usd", "days": "90"}  # No interval for hourly data
        data = self.fetch_with_retry(url, params)
        if data and 'prices' in data and 'total_volumes' in data:
            # Create DataFrame with OHLC and volume
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
            volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            df = pd.merge(prices, volumes, on='timestamp', how='outer')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            # Approximate OHLC (CoinGecko only provides close)
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df['high'] = df['close']  # Simplified
            df['low'] = df['close']   # Simplified
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.set_index('timestamp').resample('h').last().reset_index()
            df.to_csv(f"data/price_data_{coin}.csv", index=False)
            print(f"✅ Saved {coin} data to data/price_data_{coin}.csv")
            return df
        print(f"⚠️ Warning: Failed to fetch CoinGecko {coin} data.")
        return pd.DataFrame()

    def fetch_btc_data(self):
        print("\n🔄 Testing Binance connectivity...")
        if not self.test_binance_connectivity():
            print("⚠️ Warning: Cannot connect to Binance API. Falling back to CoinGecko.")
            return self.fetch_coingecko_data("bitcoin")
        else:
            binance_df = self.fetch_binance_data()
            if not binance_df.empty:
                return binance_df
            print("⚠️ Binance data empty. Falling back to CoinGecko.")
            return self.fetch_coingecko_data("bitcoin")

    def fetch_etherscan_eth_volume(self, min_value_eth=100, max_blocks=2160):
        print("\n⏳ Fetching large ETH transactions from Etherscan...")
        latest_block = self.fetch_with_retry(
            self.urls['etherscan'],
            {"module": "proxy", "action": "eth_blockNumber", "apikey": self.etherscan_api_key},
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
                "apikey": self.etherscan_api_key
            }
            block_data = self.fetch_with_retry(self.urls['etherscan'], params, timeout=30)
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
            time.sleep(0.25)

        df = pd.DataFrame(tx_data) if tx_data else pd.DataFrame(columns=['timestamp', 'tx_hash', 'from', 'to', 'eth_value'])
        if not df.empty:
            df.to_csv("data/etherscan_data.csv", index=False)
            df_agg = df.set_index('timestamp').resample('h').sum(numeric_only=True).reset_index()
            df_agg['eth_tx_volume'] = df_agg['eth_value'].fillna(0)
            df_agg = df_agg[['timestamp', 'eth_tx_volume']]
            df_agg.to_csv("data/etherscan_data_agg.csv", index=False)
            print(f"✅ Saved etherscan data to data/etherscan_data.csv and data/etherscan_data_agg.csv")
        return df

    def fetch_cryptoquant_data(self):
        print("\n⏳ Fetching exchange flows data from CryptoQuant...")
        
        # Fetch all three metrics
        inflow_data = self.fetch_with_retry(self.urls['cryptoquant_inflow'], self.params['cryptoquant_inflow'], self.headers)
        outflow_data = self.fetch_with_retry(self.urls['cryptoquant_outflow'], self.params['cryptoquant_outflow'], self.headers)
        netflow_data = self.fetch_with_retry(self.urls['cryptoquant_netflow'], self.params['cryptoquant_netflow'], self.headers)
        
        dfs = []
        
        def process_flow_data(data, flow_type):
            if not data or not isinstance(data, dict) or 'data' not in data:
                print(f"⚠️ Warning: Failed to fetch {flow_type} data or data is malformed")
                return None
            
            df = pd.DataFrame(data['data'])
            print(f"📋 {flow_type} data columns: {list(df.columns)}")
            
            if df.empty:
                print(f"⚠️ Warning: No data in {flow_type} response")
                return None
            
            # Handle different flow types with their specific columns
            if flow_type == 'inflow':
                value_col = 'inflow_mean' if 'inflow_mean' in df.columns else None
            elif flow_type == 'outflow':
                value_col = 'outflow_mean' if 'outflow_mean' in df.columns else None
            elif flow_type == 'netflow':
                value_col = 'netflow_total' if 'netflow_total' in df.columns else None
            
            if not value_col:
                print(f"⚠️ Warning: Could not find value column in {flow_type} data")
                return None
            
            if 'start_time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['start_time'], unit='ms')
                df = df.rename(columns={value_col: flow_type})
                return df[['timestamp', flow_type]]
            return None
        
        # Process all three flows
        inflow_df = process_flow_data(inflow_data, 'inflow')
        if inflow_df is not None:
            dfs.append(inflow_df)
        
        outflow_df = process_flow_data(outflow_data, 'outflow')
        if outflow_df is not None:
            dfs.append(outflow_df)
        
        netflow_df = process_flow_data(netflow_data, 'netflow')
        if netflow_df is not None:
            dfs.append(netflow_df)
        
        if not dfs:
            print("⚠️ Warning: Failed to fetch any CryptoQuant data.")
            return pd.DataFrame()
        
        # Merge all dataframes on timestamp
        df = dfs[0]
        for other_df in dfs[1:]:
            df = pd.merge(df, other_df, on='timestamp', how='outer')
        
        # Save raw data to CSV
        df.to_csv("data/cryptoquant_raw_data.csv", index=False)
        print(f"✅ Saved CryptoQuant data to data/cryptoquant_raw_data.csv with columns: {list(df.columns)}")
        return df

    def fetch_glassnode_volume(self):
        print("\n⏳ Fetching transfer volume data from Glassnode...")
        data = self.fetch_with_retry(
            "https://api.datasource.cybotrade.rs/glassnode/transactions/transfers_volume_sum",
            self.params['glassnode'],
            self.headers
        )
        
        if not data:
            print("⚠️ Warning: Failed to fetch Glassnode data.")
            return pd.DataFrame(columns=['timestamp'])
        
        if not (isinstance(data, dict) and 'data' in data and isinstance(data['data'], list)):
            print("⚠️ Warning: Glassnode data is empty or malformed.")
            return pd.DataFrame(columns=['timestamp'])
        
        df = pd.DataFrame(data['data'])
        print(f"📋 Available columns in Glassnode data: {list(df.columns)}")
        
        if df.empty:
            print("⚠️ Warning: No data in Glassnode response.")
            return pd.DataFrame(columns=['timestamp'])
        
        # Rename start_time to timestamp and convert to datetime
        if 'start_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['start_time'], unit='ms')
            df = df.drop(columns=['start_time'], errors='ignore')
        
        # Save raw data to CSV
        df.to_csv("data/glassnode_volume_raw_data.csv", index=False)
        print(f"✅ Saved raw Glassnode data to data/glassnode_volume_raw_data.csv with columns: {list(df.columns)}")
        return df

    def fetch_coinglass_data(self):
        print("\n⏳ Fetching open interest OHLC data from CoinGlass...")
        data = self.fetch_with_retry(self.urls['coinglass'], self.params['coinglass'], self.headers)
        
        if not data:
            print("⚠️ Warning: Failed to fetch CoinGlass data.")
            return pd.DataFrame(columns=['timestamp'])
        
        if not (isinstance(data, dict) and 'data' in data and isinstance(data['data'], list)):
            print("⚠️ Warning: CoinGlass data is empty or malformed.")
            return pd.DataFrame(columns=['timestamp'])
        
        df = pd.DataFrame(data['data'])
        print(f"📋 Available columns in CoinGlass data: {list(df.columns)}")
        
        if df.empty:
            print("⚠️ Warning: No data in CoinGlass response.")
            return pd.DataFrame(columns=['timestamp'])
        
        # Rename start_time to timestamp and convert to datetime
        if 'start_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['start_time'], unit='ms')
            df = df.drop(columns=['start_time'], errors='ignore')
        
        # Save raw data to CSV
        df.to_csv("data/coinglass_raw_data.csv", index=False)
        print(f"✅ Saved raw CoinGlass data to data/coinglass_raw_data.csv with columns: {list(df.columns)}")
        return df
    
    def fetch_all(self):
        dfs = {
            'btc': self.fetch_btc_data(),
            'eth': self.fetch_coingecko_data("ethereum"),
            'eth_tx': self.fetch_etherscan_eth_volume(),
            'cryptoquant': self.fetch_cryptoquant_data(),
            'glassnode': self.fetch_glassnode_volume(),
            'coinglass': self.fetch_coinglass_data()
        }
        return dfs

    def combine_all_data(self):
        # Create master hourly index
        end_time = pd.Timestamp.now(tz='UTC').floor('h')
        start_time = end_time - pd.Timedelta(days=90)
        master_index = pd.date_range(start=start_time, end=end_time, freq='h', tz='UTC')
        merged_df = pd.DataFrame(index=master_index)

        # Process each source with custom aggregation
        for source_name, df in self.fetch_all().items():
            if df.empty:
                print(f"⚠️ Skipping {source_name}: Empty DataFrame")
                continue
                
            # Standardize timestamp index
            if not isinstance(df.index, pd.DatetimeIndex):
                time_col = next((c for c in df.columns if 'time' in c.lower()), None)
                if time_col:
                    df['timestamp'] = pd.to_datetime(df[time_col], utc=True)
                    df = df.set_index('timestamp')
                else:
                    print(f"⚠️ Skipping {source_name}: No timestamp column")
                    continue

            # Custom aggregation per data source
            if source_name == 'btc':
                # Check available columns
                available_cols = set(df.columns)
                expected_cols = {'open', 'high', 'low', 'close', 'volume'}
                if not expected_cols.issubset(available_cols):
                    print(f"⚠️ BTC data missing expected columns. Available: {list(available_cols)}")
                    # Rename if possible (e.g., from fetch_coingecko_data)
                    rename_dict = {
                        f'{source_name}_open': 'open',
                        f'{source_name}_high': 'high',
                        f'{source_name}_low': 'low',
                        f'{source_name}_close': 'close',
                        f'{source_name}_volume': 'volume'
                    }
                    for old_col, new_col in rename_dict.items():
                        if old_col in df.columns:
                            df = df.rename(columns={old_col: new_col})
                    available_cols = set(df.columns)
                
                agg_dict = {}
                if 'open' in available_cols:
                    agg_dict['open'] = 'first'
                if 'high' in available_cols:
                    agg_dict['high'] = 'max'
                if 'low' in available_cols:
                    agg_dict['low'] = 'min'
                if 'close' in available_cols:
                    agg_dict['close'] = 'last'
                if 'volume' in available_cols:
                    agg_dict['volume'] = 'sum'
                
                if not agg_dict:
                    print(f"⚠️ No valid columns for BTC aggregation")
                    continue
                
                try:
                    resampled = df.resample('h').agg(agg_dict)
                    resampled = resampled.add_prefix('btc_')
                except Exception as e:
                    print(f"❌ Error resampling BTC data: {e}")
                    continue
                
            elif source_name == 'coinglass':
                resampled = df.rename(columns={
                    'o': 'open_interest_open',
                    'h': 'open_interest_high',
                    'l': 'open_interest_low',
                    'c': 'open_interest_close'
                }).resample('h').last()
                
            elif source_name == 'cryptoquant':
                resampled = df.resample('h').mean()
                resampled = resampled.add_prefix('cq_')
                
            elif source_name == 'eth':
                # Similar to BTC, handle ETH OHLC
                available_cols = set(df.columns)
                expected_cols = {'open', 'high', 'low', 'close', 'volume'}
                if not expected_cols.issubset(available_cols):
                    rename_dict = {
                        f'{source_name}_open': 'open',
                        f'{source_name}_high': 'high',
                        f'{source_name}_low': 'low',
                        f'{source_name}_close': 'close',
                        f'{source_name}_volume': 'volume'
                    }
                    for old_col, new_col in rename_dict.items():
                        if old_col in df.columns:
                            df = df.rename(columns={old_col: new_col})
                    available_cols = set(df.columns)
                
                agg_dict = {}
                if 'open' in available_cols:
                    agg_dict['open'] = 'first'
                if 'high' in available_cols:
                    agg_dict['high'] = 'max'
                if 'low' in available_cols:
                    agg_dict['low'] = 'min'
                if 'close' in available_cols:
                    agg_dict['close'] = 'last'
                if 'volume' in available_cols:
                    agg_dict['volume'] = 'sum'
                
                if not agg_dict:
                    print(f"⚠️ No valid columns for ETH aggregation")
                    continue
                
                try:
                    resampled = df.resample('h').agg(agg_dict)
                    resampled = resampled.add_prefix('eth_')
                except Exception as e:
                    print(f"❌ Error resampling ETH data: {e}")
                    continue
                
            else:
                resampled = df.resample('h').last()
                resampled = resampled.add_prefix(f'{source_name}_')

            merged_df = merged_df.join(resampled, how='left')

        # Handle missing values
        ohlc_cols = [col for col in merged_df.columns if any(x in col for x in ['open', 'high', 'low', 'close'])]
        for col in merged_df.columns:
            if any(ohlc in col for ohlc in ['open', 'high', 'low', 'close']):
                merged_df[col] = merged_df[col].ffill()
            elif 'volume' in col:
                merged_df[col] = merged_df[col].fillna(0)
            else:
                merged_df[col] = merged_df[col].interpolate()

        return merged_df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    collector = DataCollector()
    combined_df = collector.combine_all_data()
    print(f"📊 Combined dataset shape: {combined_df.shape}")
    print("🧩 Columns:", list(combined_df.columns))