# test_main.py
import pandas as pd
from backtester import SimpleBacktester

# Configuration settings (editable by user)
config = {
    "entry_threshold": 1.5,  # Set the entry threshold for buy/sell signals
    "exit_threshold": 1,   # Set the exit threshold
    "fee_rate": 0.0006,      # Fee rate per trade (e.g., 0.06%)
    "csv_file_path": 'api-ck/btc_ml_ready.csv',  # Path to the historical data CSV
    "datetime_column": 'datetime',  # The column name for datetime in the CSV
    "fill_missing_values": True,  # Whether to fill missing values in the data
    "rolling_window": 20        # Set the rolling window for z-score calculation
}

# Load the historical data
df = pd.read_csv(config["csv_file_path"])
df['Time'] = pd.to_datetime(df[config["datetime_column"]])

# Fill missing values if needed
if config["fill_missing_values"]:
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

# STRATEGY INSERT (User can insert strategies based on the data input)
# For eth_ml_ready
# def custom_generate_signals(bt):
#     # Strategy using z-scores (user can modify this logic)
#     df = bt.df
#     df['volatility_zscore'] = (df['volatility'] - df['volatility'].rolling(window=bt.rolling_window).mean()) / df['volatility'].rolling(window=bt.rolling_window).std()
#     df['volume_zscore'] = (df['volume'] - df['volume_ma']) / df['volume'].rolling(window=bt.rolling_window).std()

#     # Composite signal based on volatility and volume z-scores
#     df['signal'] = 0
#     df.loc[(df['volatility_zscore'] > bt.entry_threshold) & (df['volume_zscore'] > bt.entry_threshold), 'signal'] = -1  # sell signal
#     df.loc[(df['volatility_zscore'] < -bt.entry_threshold) & (df['volume_zscore'] < -bt.entry_threshold), 'signal'] = 1  # buy signal
#     df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)
#     bt.df = df
    
# For btc_ml_ready
def custom_generate_signals(self):
    # Use the rolling_window for the zscore calculation
    self.df['zscore'] = (self.df['inflow_inflow_mean'] - self.df['inflow_inflow_mean_ma7']) / self.df['inflow_inflow_mean'].rolling(window=self.rolling_window).std()
    self.df['signal'] = 0
    self.df.loc[self.df['zscore'] > self.entry_threshold, 'signal'] = -1  # sell
    self.df.loc[self.df['zscore'] < -self.entry_threshold, 'signal'] = 1  # buy
    self.df['position'] = self.df['signal'].replace(to_replace=0, method='ffill').fillna(0)

# Create the backtester object with customizable strategy
bt = SimpleBacktester(
    df,
    generate_signals_func=custom_generate_signals,  # Pass the user-defined strategy function
    entry_threshold=config["entry_threshold"],
    exit_threshold=config["exit_threshold"],
    fee_rate=config["fee_rate"],
    rolling_window=config["rolling_window"]  # Pass the rolling_window from config
)

# Run the backtest
bt.run_backtest()

# Print performance metrics
metrics = bt.calculate_metrics()
print("Performance Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Plot the results
bt.plot()


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap
# from backtester import SimpleBacktester

# # Configuration settings (editable by user)
# config = {
#     "entry_thresholds": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  # Expanded range for entry thresholds
#     "exit_threshold": 1,   # Set the exit threshold
#     "fee_rate": 0.0006,     # Fee rate per trade (e.g., 0.06%)
#     "csv_file_path": 'api-ck/btc_ml_ready.csv',  # Path to the historical data CSV
#     "datetime_column": 'datetime',  # The column name for datetime in the CSV
#     "fill_missing_values": True,   # Whether to fill missing values in the data
#     "rolling_windows": [5, 10, 15, 20, 30, 40, 50],  # Expanded range for rolling windows
# }

# # Load the historical data
# df = pd.read_csv(config["csv_file_path"])
# df['Time'] = pd.to_datetime(df[config["datetime_column"]])

# # Fill missing values if needed
# if config["fill_missing_values"]:
#     df.fillna(method='ffill', inplace=True)
#     df.fillna(method='bfill', inplace=True)

# # Initialize a list to store results for the heatmap
# heatmap_data = []

# # Loop through different entry thresholds and rolling windows
# for entry_threshold in config["entry_thresholds"]:
#     for rolling_window in config["rolling_windows"]:
#         # Create the backtester object
#         bt = SimpleBacktester(
#             df,
#             entry_threshold=entry_threshold,
#             exit_threshold=config["exit_threshold"],
#             fee_rate=config["fee_rate"],
#             rolling_window=rolling_window
#         )

#         # Run the backtest
#         bt.run_backtest()

#         # Calculate the metrics and extract Sharpe Ratio
#         metrics = bt.calculate_metrics()
#         sharpe_ratio = metrics.get("Sharpe Ratio", None)

#         # Store the result in the heatmap_data list
#         heatmap_data.append([rolling_window, entry_threshold, sharpe_ratio])

# # Create a DataFrame from the results
# results_df = pd.DataFrame(heatmap_data, columns=['Rolling Window', 'Entry Threshold', 'Sharpe Ratio'])

# # Pivot the DataFrame for heatmap generation
# heatmap_data_pivot = results_df.pivot(index='Rolling Window', columns='Entry Threshold', values='Sharpe Ratio')

# # Create a custom red to green colormap
# cmap = LinearSegmentedColormap.from_list("red_green", ["red", "green"])

# # Generate the heatmap with the custom colormap
# plt.figure(figsize=(12, 8))  # Increased figure size
# sns.heatmap(heatmap_data_pivot, annot=True, cmap=cmap, fmt='.2f', cbar_kws={'label': 'Sharpe Ratio'})
# plt.title('Sharpe Ratio Heatmap - Entry Threshold vs. Rolling Window')
# plt.xlabel('Entry Threshold')
# plt.ylabel('Rolling Window')
# plt.show()
