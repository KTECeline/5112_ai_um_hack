# crypto-trader/hmm_model.py
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

class HMMModel:
    def __init__(self, n_components=3, n_iter=2000, random_state=42):
        self.model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=n_iter, random_state=random_state)
        self.scaler = StandardScaler()
        self.features_list = [
            'inflow_total', 'value', 'eth_tx_volume', 'btc_volume', 'btc_close_price',
            'price_change', 'sma_20', 'rsi_14', 'avg_sentiment', 'sentiment_std', 'tweet_volume'
        ]

    def train(self, df):
        if df.empty:
            print("❌ Error: Empty DataFrame provided.")
            return None

        # Select available features
        features = [f for f in self.features_list if f in df.columns]
        if not features:
            print("❌ Error: No valid features available.")
            return None

        X = df[features]
        X_scaled = self.scaler.fit_transform(X)
        try:
            self.model.fit(X_scaled)
            hidden_states = self.model.predict(X_scaled)
            df['hidden_states'] = hidden_states
            print(f"✅ HMM trained with {len(set(hidden_states))} unique states: {sorted(set(hidden_states))}")
            return df
        except Exception as e:
            print(f"⚠️ Warning: HMM failed to converge: {e}")
            df['hidden_states'] = 0
            return df

    def save(self, df, output_path="data/combined_crypto_data_with_hmm.csv"):
        df.to_csv(output_path, index=False)
        print(f"✅ Saved HMM output to {output_path}")

if __name__ == "__main__":
    df = pd.read_csv("data/combined_features.csv")
    hmm = HMMModel()
    df_hmm = hmm.train(df)
    hmm.save(df_hmm)