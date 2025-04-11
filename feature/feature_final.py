import pandas as pd
import numpy as np
from pathlib import Path
import sys

# =============================================
# 1. Configuration
# =============================================
DATA_DIR = Path("../data/data")  # Adjust if needed
FEATURE_DIR = Path("feature")
FEATURE_DIR.mkdir(exist_ok=True)

# =============================================
# 2. Load Individual Datasets
# =============================================
def load_datasets():
    """Load all raw datasets with proper preprocessing"""
    try:
        # BTC Price Data
        btc_price_path = DATA_DIR / "price_data_bitcoin.csv"
        if not btc_price_path.exists():
            raise FileNotFoundError(f"BTC price data not found at {btc_price_path}")
            
        btc_price_df = pd.read_csv(
            btc_price_path,
            parse_dates=['timestamp'],
            index_col='timestamp'
        ).rename(columns={
            'open': 'btc_open',
            'high': 'btc_high',
            'low': 'btc_low',
            'close': 'btc_close',
            'volume': 'btc_volume'
        })
        # Remove timezone information
        btc_price_df.index = btc_price_df.index.tz_localize(None)
        
        # ETH Price Data
        eth_price_path = DATA_DIR / "price_data_ethereum.csv"
        eth_price_df = None
        if eth_price_path.exists():
            eth_price_df = pd.read_csv(
                eth_price_path,
                parse_dates=['timestamp'],
                index_col='timestamp'
            ).rename(columns={
                'open': 'eth_open',
                'high': 'eth_high',
                'low': 'eth_low',
                'close': 'eth_close',
                'volume': 'eth_volume'
            })
            # Remove timezone information
            eth_price_df.index = eth_price_df.index.tz_localize(None)
        else:
            print("⚠️ ETH price data not found, proceeding without it")
        
        # ETH Whale Transactions
        whale_path = DATA_DIR / "etherscan_data.csv"
        whale_df = None
        if whale_path.exists():
            whale_df = pd.read_csv(
                whale_path,  
                parse_dates=['timestamp'],
                index_col='timestamp'
            )
            # Remove timezone information
            whale_df.index = whale_df.index.tz_localize(None)
        else:
            print("⚠️ Whale data not found, proceeding without it")
        
        # Exchange Flows (BTC-specific)
        flow_path = DATA_DIR / "cryptoquant_raw_data.csv"
        if not flow_path.exists():
            raise FileNotFoundError(f"Flow data not found at {flow_path}")
            
        flow_df = pd.read_csv(
            flow_path,
            parse_dates=['timestamp'],
            index_col='timestamp'
        ).rename(columns={
            'inflow': 'cq_inflow',
            'outflow': 'cq_outflow',
            'netflow': 'cq_netflow'
        })
        # Remove timezone information
        flow_df.index = flow_df.index.tz_localize(None)
        
        return btc_price_df, eth_price_df, whale_df, flow_df
    
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return None, None, None, None
    
# =============================================
# 3. Feature Engineering
# =============================================
def create_whale_features(whale_df):
    """Create features from ETH whale transaction data"""
    if whale_df is None:
        return None
        
    try:
        # Resample whale transactions by hour
        whale_hourly = whale_df.resample('1h').agg({
            'eth_value': ['sum', 'count'],
            'from': pd.Series.nunique,
            'to': pd.Series.nunique
        })
        
        # Flatten multi-index columns
        whale_hourly.columns = [
            'eth_whale_volume',
            'eth_whale_tx_count',
            'eth_unique_senders',
            'eth_unique_receivers'
        ]
        
        # Add moving averages
        whale_hourly['eth_whale_volume_ma_6h'] = whale_hourly['eth_whale_volume'].rolling(6).mean()
        whale_hourly['eth_whale_tx_ma_6h'] = whale_hourly['eth_whale_tx_count'].rolling(6).mean()
        
        return whale_hourly.fillna(0)
    except Exception as e:
        print(f"❌ Error creating ETH whale features: {e}")
        return None

def create_price_features(df, prefix='btc'):
    if df is None:
        return None
    try:
        # Lag features
        for lag in [1, 6, 24]:
            df[f'{prefix}_close_lag_{lag}h'] = df[f'{prefix}_close'].shift(lag)
        # Moving averages
        for window in [6, 24]:
            df[f'{prefix}_ma_{window}h'] = df[f'{prefix}_close'].rolling(window).mean()
        # Returns and volatility
        df[f'{prefix}_return_1h'] = df[f'{prefix}_close'].pct_change(fill_method=None)
        df[f'{prefix}_volatility_6h'] = df[f'{prefix}_return_1h'].rolling(6).std()
        # RSI
        def compute_rsi(series, window=14):
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window).mean()
            avg_loss = loss.rolling(window).mean()
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        df[f'{prefix}_rsi_14h'] = compute_rsi(df[f'{prefix}_close'])
        return df
    except Exception as e:
        print(f"❌ Error creating {prefix} price features: {e}")
        return None

def create_volume_features(df, prefix='btc'):
    if df is None:
        return None
    try:
        volume_col = f'{prefix}_volume'
        if volume_col in df.columns:
            df[f'{prefix}_volume_change_1h'] = df[volume_col].pct_change(fill_method=None)
            df[f'{prefix}_volume_ma_6h'] = df[volume_col].rolling(6).mean()
            df[f'{prefix}_volume_ma_ratio'] = df[volume_col] / df[f'{prefix}_volume_ma_6h']
        else:
            print(f"⚠️ No {volume_col} column found, volume features will be skipped for {prefix.upper()}")
        return df
    except Exception as e:
        print(f"❌ Error creating {prefix} volume features: {e}")
        return None
    
def create_volume_features(df, prefix='btc'):
    """Create volume features for BTC or ETH"""
    if df is None:
        return None
    
    try:
        volume_col = f'{prefix}_volume'
        if volume_col in df.columns:
            df[f'{prefix}_volume_change_1h'] = df[volume_col].pct_change()
            df[f'{prefix}_volume_ma_6h'] = df[volume_col].rolling(6).mean()
            df[f'{prefix}_volume_ma_ratio'] = df[volume_col] / df[f'{prefix}_volume_ma_6h']
        else:
            print(f"⚠️ No {volume_col} column found, volume features will be skipped for {prefix.upper()}")
        return df
    except Exception as e:
        print(f"❌ Error creating {prefix} volume features: {e}")
        return None

def create_onchain_features(flow_df, whale_df=None):
    """Create on-chain features for BTC (flows) and ETH (whale activity)"""
    if flow_df is None and whale_df is None:
        return None, None
        
    try:
        # BTC Flow metrics
        flow_features = flow_df.copy() if flow_df is not None else None
        if flow_features is not None and all(col in flow_features.columns for col in ['cq_inflow', 'cq_outflow']):
            flow_features['cq_flow_ratio'] = flow_features['cq_inflow'] / (flow_features['cq_outflow'] + 1e-6)
        
        # ETH Whale activity with unique column names
        eth_features = None
        if whale_df is not None and 'eth_value' in whale_df.columns:
            eth_features = whale_df.resample('1h').agg({
                'eth_value': 'sum',
                'tx_hash': 'count'
            }).rename(columns={
                'eth_value': 'eth_whale_volume_onchain',  # Changed to avoid overlap
                'tx_hash': 'eth_whale_tx_count_onchain'   # Changed to avoid overlap
            })
            eth_features['eth_whale_tx_count_large'] = (whale_df['eth_value'] > 100).resample('1h').sum()
            eth_features['eth_whale_volume_ma_6h_onchain'] = eth_features['eth_whale_volume_onchain'].rolling(6).mean()  # Changed to avoid overlap
        
        return flow_features, eth_features
    except Exception as e:
        print(f"❌ Error creating on-chain features: {e}")
        return None, None

# =============================================
# 4. Data Integration (Fixed column overlap)
# =============================================
def integrate_datasets(btc_price_df, eth_price_df, whale_features, flow_df, eth_onchain_df):
    """Integrate BTC and ETH datasets"""
    if btc_price_df is None and eth_price_df is None:
        return None
        
    try:
        # Initialize with BTC price data (if available) or ETH as fallback
        if btc_price_df is not None:
            combined = btc_price_df.resample('1h').last()
        elif eth_price_df is not None:
            combined = eth_price_df.resample('1h').last()
        else:
            return None
        
        # Join ETH price data
        if eth_price_df is not None:
            eth_price_df.index = eth_price_df.index.tz_localize(None)  # Ensure no timezone
            combined = combined.join(eth_price_df.resample('1h').last(), how='left')
        
        # Join flow data (BTC-specific)
        if flow_df is not None:
            flow_df.index = flow_df.index.tz_localize(None)  # Ensure no timezone
            combined = combined.join(flow_df.resample('1h').last().add_prefix('flow_'), how='left')
        
        # Join whale features (ETH-specific)
        if whale_features is not None:
            whale_features.index = whale_features.index.tz_localize(None)  # Ensure no timezone
            combined = combined.join(whale_features, how='left')
        
        # Join ETH on-chain features
        if eth_onchain_df is not None:
            eth_onchain_df.index = eth_onchain_df.index.tz_localize(None)  # Ensure no timezone
            combined = combined.join(eth_onchain_df, how='left')
        
        # Handle missing data
        for col in ['btc_close', 'eth_close']:
            if col in combined.columns:
                combined[col] = combined[col].ffill()
        
        # Fill numerical columns with 0
        num_cols = combined.select_dtypes(include=[np.number]).columns
        combined[num_cols] = combined[num_cols].fillna(0)
        
        return combined
    except Exception as e:
        print(f"❌ Error integrating datasets: {e}")
        return None

# =============================================
# 5. Main Execution
# =============================================
if __name__ == "__main__":
    print("🚀 Starting feature engineering pipeline...")
    
    try:
        # Step 1: Load data
        btc_price_df, eth_price_df, whale_df, flow_df = load_datasets()
        if btc_price_df is None and eth_price_df is None:
            print("❌ No price datasets loaded - exiting")
            sys.exit(1)
        print("✅ Data loaded successfully")
        
        # Step 2: Create features
        btc_price_df = create_price_features(btc_price_df, prefix='btc')
        btc_price_df = create_volume_features(btc_price_df, prefix='btc')
        eth_price_df = create_price_features(eth_price_df, prefix='eth')
        eth_price_df = create_volume_features(eth_price_df, prefix='eth')
        whale_features = create_whale_features(whale_df)
        flow_df, eth_onchain_df = create_onchain_features(flow_df, whale_df)
        
        if btc_price_df is None and eth_price_df is None:
            print("❌ Feature creation failed - exiting")
            sys.exit(1)
        print("✅ Features created")
        
        # Step 3: Combine datasets
        final_df = integrate_datasets(btc_price_df, eth_price_df, whale_features, flow_df, eth_onchain_df)
        if final_df is None:
            print("❌ Dataset integration failed - exiting")
            sys.exit(1)
            
        print(f"✅ Combined dataset shape: {final_df.shape}")
        
        # Step 4: Save results
        output_path = FEATURE_DIR / "engineered_features_final.csv"
        final_df.to_csv(output_path)
        print(f"💾 Saved final features to {output_path}")
        
        # Show sample of important features
        print("\n🔍 Sample features created:")
        important_cols = [
            'btc_close', 'btc_rsi_14h', 'btc_volatility_6h',
            'eth_close', 'eth_rsi_14h', 'eth_volatility_6h',
            'flow_cq_flow_ratio', 'eth_whale_volume', 'eth_whale_tx_count',
            'eth_whale_volume_onchain', 'eth_whale_tx_count_onchain'
        ]
        important_cols = [col for col in important_cols if col in final_df.columns]
        print(final_df[important_cols].tail())
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)