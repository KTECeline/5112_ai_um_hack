import pandas as pd
from backtester import SimpleBacktester

# Load the data (ensure the CSV file path and column names are correct)
df = pd.read_csv('models/lstm_results/merged_data_with_lstm_predictions.csv')
df['Time'] = pd.to_datetime(df['timestamp'])

# Define the configuration
config = {
    "returns_column": 'btc_return_1h',  # The returns column name in your CSV
    "fee_rate": 0.0006,
}

# Define the strategy signal generation function
def custom_generate_signals(bt):
    df = bt.df
    # Use the LSTM predictions for generating signals
    df['signal'] = 0
    df.loc[df['lstm_predictions'] > bt.entry_threshold, 'signal'] = -1  # Sell signal
    df.loc[df['lstm_predictions'] < -bt.entry_threshold, 'signal'] = 1  # Buy signal
    df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)
    bt.df = df

# Set the parameter ranges for entry_threshold and rolling_window
entry_thresholds = [0.5, 1, 1.5, 2, 2.5]  # Adjust this range as necessary
rolling_windows = [10, 20, 30, 40, 50]  # Adjust this range as necessary

# Generate the heatmap using the plot_sharpe_heatmap function
def plot_sharpe_heatmap(df, strategy_func, entry_thresholds, rolling_windows, fee_rate=0.0006):
    heatmap_data = []

    for entry in entry_thresholds:
        for window in rolling_windows:
            bt = SimpleBacktester(
                df=df,
                generate_signals_func=strategy_func,
                entry_threshold=entry,
                fee_rate=fee_rate,
                rolling_window=window,
                returns_column=config["returns_column"]
            )
            try:
                bt.run_backtest()
                metrics = bt.calculate_metrics()
                sharpe = metrics.get("Sharpe Ratio", 0)
                heatmap_data.append([window, entry, sharpe])
            except Exception as e:
                print(f"Error with entry {entry}, window {window}: {e}")
                heatmap_data.append([window, entry, None])

    heatmap_df = pd.DataFrame(heatmap_data, columns=["Rolling Window", "Entry Threshold", "Sharpe Ratio"])
    heatmap_pivot = heatmap_df.pivot(index="Rolling Window", columns="Entry Threshold", values="Sharpe Ratio")

    # Plot the heatmap
    import seaborn as sns
    import matplotlib.pyplot as plt

    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_pivot, annot=True, cmap=cmap, fmt='.2f', cbar_kws={'label': 'Sharpe Ratio'})
    plt.title("Sharpe Ratio Heatmap - Entry Threshold vs Rolling Window")
    plt.xlabel("Entry Threshold")
    plt.ylabel("Rolling Window")
    plt.tight_layout()
    plt.show()

# Call the function to plot the heatmap
plot_sharpe_heatmap(df, custom_generate_signals, entry_thresholds, rolling_windows)
