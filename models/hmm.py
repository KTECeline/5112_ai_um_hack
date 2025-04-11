import pandas as pd
import numpy as np
from pathlib import Path
import sys
from hmmlearn.hmm import GaussianHMM
import pickle
import matplotlib.pyplot as plt

# =============================================
# 1. Configuration
# =============================================
ML_DATA_DIR = Path("../feature/ml_data")
FEATURE_DIR = Path("../feature/feature")
OUTPUT_DIR = Path("hmm_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# HMM Parameters
N_REGIMES = 3  # Try 3 regimes
RANDOM_STATE = 42
N_ITER = 2000
TOL = 1e-6
COVARIANCE_TYPE = "full"

# HMM Features (expanded)
HMM_FEATURES = ['btc_return_1h', 'btc_volatility_6h', 'flow_cq_flow_ratio', 'eth_whale_tx_count', 'btc_rsi_14h']

# =============================================
# 2. Load ML-Ready Data
# =============================================
def load_ml_data():
    try:
        data_path = ML_DATA_DIR / "ml_ready_data.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"ML-ready data not found at {data_path}")
        
        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        df = df.loc[:, ~df.columns.duplicated()]
        
        hmm_features_path = ML_DATA_DIR / "hmm_features.txt"
        if hmm_features_path.exists():
            with open(hmm_features_path, 'r') as f:
                hmm_features = [line.strip() for line in f.readlines()]
        else:
            hmm_features = HMM_FEATURES
            print("⚠️ HMM features file not found, using default features")
        
        hmm_features = list(set(hmm_features))
        missing_features = [col for col in hmm_features if col not in df.columns]
        if missing_features:
            print(f"⚠️ Missing HMM features: {missing_features}. Using available features")
            hmm_features = [col for col in hmm_features if col in df.columns]
        
        if not hmm_features:
            raise ValueError("❌ No valid HMM features available")
        
        X_hmm = df[hmm_features].copy()
        
        if 'flow_cq_flow_ratio' in X_hmm.columns:
            X_hmm['flow_cq_flow_ratio'] = X_hmm['flow_cq_flow_ratio'].clip(lower=-5, upper=5)
        
        variances = X_hmm.var()
        low_variance = variances[variances < 1e-5].index.tolist()
        if low_variance:
            print(f"⚠️ Low variance features: {low_variance}. Removing them")
            X_hmm = X_hmm.drop(columns=low_variance)
            hmm_features = [f for f in hmm_features if f not in low_variance]
        
        btc_close = None
        feature_path = FEATURE_DIR / "engineered_features_final.csv"
        if feature_path.exists():
            feature_df = pd.read_csv(feature_path, parse_dates=['timestamp'], index_col='timestamp')
            if 'btc_close' in feature_df.columns:
                btc_close = feature_df['btc_close'].reindex(df.index, method='ffill')
        
        if X_hmm.isna().any().any():
            X_hmm = X_hmm.fillna(0)
            print("⚠️ Filled missing values with 0")
        
        if len(X_hmm) < 10000:
            print(f"⚠️ Dataset has {len(X_hmm)} samples (~{len(X_hmm)//24} days). Multi-year backtest requires ~35000 samples (4 years hourly).")
        
        print(f"✅ Loaded ML-ready data with shape: {df.shape}")
        print(f"✅ HMM features: {hmm_features}")
        return X_hmm, btc_close, hmm_features
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

# =============================================
# 3. Train HMM
# =============================================
def train_hmm(X_hmm, n_regimes=3):
    try:
        hmm_model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=COVARIANCE_TYPE,
            n_iter=N_ITER,
            tol=TOL,
            random_state=RANDOM_STATE
        )
        hmm_model.fit(X_hmm)
        
        if not hmm_model.monitor_.converged:
            print("⚠️ HMM did not converge. Try increasing N_ITER or adjusting features.")
        
        regimes = hmm_model.predict(X_hmm)
        regime_df = pd.DataFrame(regimes, index=X_hmm.index, columns=['regime'])
        
        regime_counts = regime_df['regime'].value_counts().sort_index()
        regime_summary = X_hmm.groupby(regime_df['regime']).mean()
        
        min_regime_proportion = regime_counts.min() / regime_counts.sum()
        if min_regime_proportion < 0.05:
            print(f"⚠️ Imbalanced regimes: Smallest regime has {min_regime_proportion:.2%} of samples.")
        
        print(f"✅ Trained HMM with {n_regimes} regimes")
        print(f"Regime distribution:\n{regime_counts}")
        print(f"Regime characteristics:\n{regime_summary}")
        return hmm_model, regime_df
    
    except Exception as e:
        print(f"❌ Error training HMM: {e}")
        return None, None

# =============================================
# 4. Save Results
# =============================================
def save_results(hmm_model, regime_df, hmm_features, output_dir):
    try:
        model_path = output_dir / "hmm_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(hmm_model, f)
        print(f"💾 Saved HMM model to {model_path}")
        
        regimes_path = output_dir / "hmm_regimes.csv"
        regime_df.to_csv(regimes_path)
        print(f"💾 Saved regimes to {regimes_path}")
        
        features_path = output_dir / "hmm_features_used.txt"
        with open(features_path, 'w') as f:
            f.write("\n".join(hmm_features))
        print(f"💾 Saved HMM features to {features_path}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")

# =============================================
# 5. Visualize Regimes
# =============================================
def visualize_regimes(regime_df, btc_close, X_hmm, output_dir):
    try:
        plt.figure(figsize=(15, 10))
        
        if btc_close is not None:
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(btc_close.index, btc_close, label='BTC Price', color='blue')
            ax1.set_title('BTC Price and Market Regimes')
            ax1.set_ylabel('Price (USD)')
            ax1.legend()
            
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        else:
            ax2 = plt.subplot(2, 1, 1)
        
        ax2.plot(X_hmm.index, X_hmm['btc_return_1h'], label='BTC Return (1h)', color='green')
        ax2.set_ylabel('Return')
        ax2.legend()
        
        ax3 = plt.subplot(3, 1, 3, sharex=ax1) if btc_close is not None else plt.subplot(2, 1, 2, sharex=ax2)
        for regime in sorted(regime_df['regime'].unique()):
            mask = regime_df['regime'] == regime
            ax3.scatter(
                regime_df.index[mask],
                [regime] * mask.sum(),
                label=f'Regime {regime}',
                s=10
            )
        ax3.set_ylabel('Regime')
        ax3.set_xlabel('Time')
        ax3.legend()
        
        plt.tight_layout()
        plot_path = output_dir / "hmm_regimes_plot.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"💾 Saved regime visualization to {plot_path}")
        
    except Exception as e:
        print(f"❌ Error visualizing regimes: {e}")

# =============================================
# 6. Main Execution
# =============================================
if __name__ == "__main__":
    print("🚀 Starting HMM training pipeline...")
    
    try:
        X_hmm, btc_close, hmm_features = load_ml_data()
        if X_hmm is None:
            print("❌ No data loaded - exiting")
            sys.exit(1)
        
        hmm_model, regime_df = train_hmm(X_hmm, n_regimes=N_REGIMES)
        if hmm_model is None:
            print("❌ HMM training failed - exiting")
            sys.exit(1)
        
        save_results(hmm_model, regime_df, hmm_features, OUTPUT_DIR)
        visualize_regimes(regime_df, btc_close, X_hmm, OUTPUT_DIR)
        
        print("\n🔍 HMM training completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)