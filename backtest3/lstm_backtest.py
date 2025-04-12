import pandas as pd
from backtester import SimpleBacktester
import matplotlib.pyplot as plt

# Configuration settings (editable by user)
config = {
    "entry_threshold": 0.5,  # Use a threshold on the LSTM prediction score for signal generation
    "exit_threshold": 0.0,   # Exit threshold (could be dynamic or zero if you want to close positions)
    "fee_rate": 0.0006,      # Fee rate per trade (e.g., 0.06%)
    "csv_file_path": 'models/lstm_results/merged_data_with_lstm_predictions.csv',  # Path to the historical data CSV
    "datetime_column": 'timestamp',  # The column name for datetime in the CSV
    "returns_column": 'btc_return_1h',   # Set the name of the returns column (e.g., "btc_return_1h")
    "fill_missing_values": True,   # Whether to fill missing values in the data
    "rolling_window": 30          # Set the rolling window for z-score calculation (if used)
}

# Load the historical data
df = pd.read_csv(config["csv_file_path"])
df['Time'] = pd.to_datetime(df[config["datetime_column"]])

# Fill missing values if needed
if config["fill_missing_values"]:
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

# Custom strategy using LSTM predictions
def custom_generate_signals(bt):
    """
    Strategy that uses the LSTM predictions for buy/sell signals
    """
    df = bt.df

    # Buy when LSTM prediction is positive (e.g., 1)
    df['signal'] = 0
    df.loc[df['lstm_predictions'] > bt.entry_threshold, 'signal'] = 1  # Buy signal
    df.loc[df['lstm_predictions'] < -bt.entry_threshold, 'signal'] = -1  # Sell signal

    # Fill the position column to carry the signal forward
    df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)
    
    # Pass the dataframe back to the backtester object
    bt.df = df

# Create the backtester object with customizable strategy
bt = SimpleBacktester(
    df,
    generate_signals_func=custom_generate_signals,  # Pass the user-defined strategy function
    returns_column=config["returns_column"],        # Pass the returns column name from config
    entry_threshold=config["entry_threshold"],      # Pass entry threshold from config
    exit_threshold=config["exit_threshold"],        # Pass exit threshold from config
    fee_rate=config["fee_rate"],                    # Pass fee rate from config
    rolling_window=config["rolling_window"]         # Pass rolling window from config
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

# You can also plot the signal distribution (if necessary)
# bt.plot_signal_distribution()
