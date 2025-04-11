import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sys

# =============================================
# 1. Configuration
# =============================================
FEATURE_DIR = Path("feature")
OUTPUT_DIR = Path("ml_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Data Parameters
PREDICTION_HORIZON = 1  # Predict next hour
PRICE_THRESHOLD = 0.001  # 0.1% price change for target
HMM_FEATURES = ['btc_return_1h', 'btc_volatility_6h', 'flow_cq_flow_ratio']  # For regime detection
LSTM_FEATURES = None  # Use all features except target and excluded columns (set dynamically)

# =============================================
# 2. Load Engineered Features
# =============================================
def load_features():
    """Load engineered features and perform initial cleaning"""
    try:
        feature_path = FEATURE_DIR / "engineered_features_final.csv"
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found at {feature_path}")
        
        df = pd.read_csv(
            feature_path,
            parse_dates=['timestamp'],
            index_col='timestamp'
        )
        
        # Ensure btc_close exists for target creation
        if 'btc_close' not in df.columns:
            raise ValueError("❌ btc_close column not found in features")
        
        # Drop rows with missing btc_close
        df = df.dropna(subset=['btc_close'])
        
        print(f"✅ Loaded features with shape: {df.shape}")
        return df
    
    except Exception as e:
        print(f"❌ Error loading features: {e}")
        return None

# =============================================
# 3. Define Target Variable
# =============================================
def create_target(df, horizon=1, threshold=0.001):
    """Create binary target: 1 if price increases by threshold, 0 otherwise"""
    try:
        # Calculate future returns
        df['future_return'] = df['btc_close'].pct_change(horizon).shift(-horizon)
        
        # Binary target: 1 if return > threshold, 0 otherwise
        df['target'] = (df['future_return'] > threshold).astype(int)
        
        # Drop rows where target is NaN
        df = df.dropna(subset=['target'])
        
        print(f"✅ Created target with {df['target'].sum()} positive cases ({df['target'].mean():.2%} of total)")
        return df
    
    except Exception as e:
        print(f"❌ Error creating target: {e}")
        return None

# =============================================
# 4. Preprocess Features
# =============================================
def preprocess_features(df, hmm_features, lstm_features=None):
    """Preprocess features for HMM and LSTM"""
    try:
        # Define HMM features (ensure they exist)
        hmm_cols = [col for col in hmm_features if col in df.columns]
        if not hmm_cols:
            raise ValueError("❌ No valid HMM features found")
        
        # Define LSTM features (all except target and excluded columns)
        exclude_cols = ['btc_close', 'eth_close', 'future_return', 'target']
        if lstm_features is None:
            lstm_cols = [col for col in df.columns if col not in exclude_cols]
        else:
            lstm_cols = [col for col in lstm_features if col in df.columns]
        
        # Check for highly correlated features
        all_features = list(set(hmm_cols + lstm_cols))
        corr_matrix = df[all_features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
        print(f"⚠️ Dropping {len(to_drop)} highly correlated features: {to_drop}")
        
        # Update feature lists
        hmm_cols = [col for col in hmm_cols if col not in to_drop]
        lstm_cols = [col for col in lstm_cols if col not in to_drop]
        
        # Create feature matrices
        X_hmm = df[hmm_cols].copy()
        X_lstm = df[lstm_cols].copy()
        y = df['target'].copy()
        
        # Fill missing values
        X_hmm = X_hmm.fillna(0)
        X_lstm = X_lstm.fillna(0)
        
        # Scale features
        hmm_scaler = StandardScaler()
        lstm_scaler = StandardScaler()
        X_hmm_scaled = pd.DataFrame(hmm_scaler.fit_transform(X_hmm), index=X_hmm.index, columns=X_hmm.columns)
        X_lstm_scaled = pd.DataFrame(lstm_scaler.fit_transform(X_lstm), index=X_lstm.index, columns=X_lstm.columns)
        
        print(f"✅ Preprocessed {len(hmm_cols)} HMM features and {len(lstm_cols)} LSTM features")
        return X_hmm_scaled, X_lstm_scaled, y, hmm_cols, lstm_cols
    
    except Exception as e:
        print(f"❌ Error preprocessing features: {e}")
        return None, None, None, None, None

# =============================================
# 5. Save ML-Ready Data
# =============================================
def save_ml_data(X_hmm, X_lstm, y, hmm_cols, lstm_cols, output_path):
    """Save ML-ready data for HMM and LSTM training"""
    try:
        # Combine HMM and LSTM features with target
        ml_data = pd.concat([X_hmm, X_lstm, y], axis=1)
        ml_data.to_csv(output_path)
        print(f"💾 Saved ML-ready data to {output_path}")
        
        # Save feature lists
        with open(output_path.parent / "hmm_features.txt", 'w') as f:
            f.write("\n".join(hmm_cols))
        with open(output_path.parent / "lstm_features.txt", 'w') as f:
            f.write("\n".join(lstm_cols))
        print(f"💾 Saved HMM feature list to {output_path.parent / 'hmm_features.txt'}")
        print(f"💾 Saved LSTM feature list to {output_path.parent / 'lstm_features.txt'}")
        
    except Exception as e:
        print(f"❌ Error saving ML data: {e}")

# =============================================
# 6. Main Execution
# =============================================
if __name__ == "__main__":
    print("🚀 Starting ML data processing pipeline...")
    
    try:
        # Step 1: Load features
        df = load_features()
        if df is None:
            print("❌ No features loaded - exiting")
            sys.exit(1)
        
        # Step 2: Create target
        df = create_target(df, horizon=PREDICTION_HORIZON, threshold=PRICE_THRESHOLD)
        if df is None:
            print("❌ Target creation failed - exiting")
            sys.exit(1)
        
        # Step 3: Preprocess features
        X_hmm, X_lstm, y, hmm_cols, lstm_cols = preprocess_features(df, hmm_features=HMM_FEATURES)
        if X_hmm is None:
            print("❌ Feature preprocessing failed - exiting")
            sys.exit(1)
        
        # Step 4: Save ML-ready data
        output_path = OUTPUT_DIR / "ml_ready_data.csv"
        save_ml_data(X_hmm, X_lstm, y, hmm_cols, lstm_cols, output_path)
        
        # Step 5: Print summary
        print("\n🔍 Data summary:")
        print(f"Total samples: {len(df)}")
        print(f"HMM features: {len(hmm_cols)} ({hmm_cols})")
        print(f"LSTM features: {len(lstm_cols)}")
        print(f"Positive target ratio: {y.mean():.2%}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)