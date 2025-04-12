# test_main.py
import pandas as pd
from backtester import SimpleBacktester

# Configuration settings (editable by user)
config = {
    "entry_threshold": 0.5,  # Set the entry threshold for buy/sell signals
    "exit_threshold": 1,   # Set the exit threshold
    "fee_rate": 0.0006,    # Fee rate per trade (e.g., 0.06%)
    "csv_file_path": 'api-ck/btc_ml_ready.csv',  # Path to the historical data CSV
    "datetime_column": 'datetime',  # The column name for datetime in the CSV
    "returns_column": 'returns',   # Set the name of the returns column (e.g., "returns")
    "fill_missing_values": True,   # Whether to fill missing values in the data
    "rolling_window": 40,        # Set the rolling window for z-score calculation
}

# Load the historical data
df = pd.read_csv(config["csv_file_path"])
df['Time'] = pd.to_datetime(df[config["datetime_column"]])

# Fill missing values if needed
if config["fill_missing_values"]:
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

# STRATEGY INSERT (User can insert strategies based on the data input)
# For eth_ml_ready & btc
def custom_generate_signals(bt):
    # Strategy using z-scores (user can modify this logic)
    df = bt.df
    df['volatility_zscore'] = (df['volatility'] - df['volatility'].rolling(window=bt.rolling_window).mean()) / df['volatility'].rolling(window=bt.rolling_window).std()
    df['volume_zscore'] = (df['volume'] - df['volume_ma']) / df['volume'].rolling(window=bt.rolling_window).std()

    # Composite signal based on volatility and volume z-scores
    df['signal'] = 0
    df.loc[(df['volatility_zscore'] > bt.entry_threshold) & (df['volume_zscore'] > bt.entry_threshold), 'signal'] = -1  # sell signal
    df.loc[(df['volatility_zscore'] < -bt.entry_threshold) & (df['volume_zscore'] < -bt.entry_threshold), 'signal'] = 1  # buy signal
    df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)
    bt.df = df
    
# For btc_ml_ready
# def custom_generate_signals(self):
#     # Use the rolling_window for the zscore calculation
#     self.df['zscore'] = (self.df['inflow_inflow_mean'] - self.df['inflow_inflow_mean_ma7']) / self.df['inflow_inflow_mean'].rolling(window=self.rolling_window).std()
#     self.df['signal'] = 0
#     self.df.loc[self.df['zscore'] > self.entry_threshold, 'signal'] = -1  # sell
#     self.df.loc[self.df['zscore'] < -self.entry_threshold, 'signal'] = 1  # buy
#     self.df['position'] = self.df['signal'].replace(to_replace=0, method='ffill').fillna(0)

# Create the backtester object with customizable strategy

bt = SimpleBacktester(
    df,
    generate_signals_func=custom_generate_signals,  # Pass the user-defined strategy function
    returns_column=config["returns_column"],        # Pass the returns column name from config
    entry_threshold=config["entry_threshold"],      # Pass entry threshold from config
    exit_threshold=config["exit_threshold"],        # Pass exit threshold from config
    fee_rate=config["fee_rate"],                    # Pass fee rate from config
    rolling_window=config["rolling_window"],         # Pass rolling window from config
)

# Run the backtest
bt.run_backtest()

# Print performance metrics
metrics = bt.calculate_metrics()
print("Performance Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Plot the results
bt.plot_strategy_performance()
bt.plot_drawdowns()

# use only if the strategy got buy
# bt.plot_signal_distribution()

