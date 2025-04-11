# crypto-trader/train.py
from fetch_data import DataCollector
from features import FeatureEngineer
from hmm_model import HMMModel
from lstm_model import LSTMModel

def main():
    # Fetch data
    print("🔄 Fetching data...")
    collector = DataCollector()
    dfs = collector.fetch_all()

    # Engineer features
    print("\n🧽 Engineering features...")
    engineer = FeatureEngineer()
    combined_df = engineer.engineer_features()

    # Train HMM
    print("\n🧠 Training HMM...")
    hmm = HMMModel()
    df_hmm = hmm.train(combined_df)
    hmm.save(df_hmm)

    # Train LSTM
    print("\n🧠 Training LSTM...")
    lstm = LSTMModel()
    df_lstm = lstm.train(df_hmm)
    lstm.save(df_lstm)

if __name__ == "__main__":
    main()