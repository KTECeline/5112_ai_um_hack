import pandas as pd
from backtester import SimpleBacktester
import matplotlib.pyplot as plt

# Configuration settings (editable by user)
config = {
    "asset_to_test": "btc",  # Options: 'btc', 'eth', 'both'
    "entry_threshold": 0.5,  # Threshold for LSTM prediction score
    "exit_threshold": 0.0,   # Exit threshold (set to 0 to hold positions until opposite signal)
    "fee_rate": 0.0006,      # Fee rate per trade (e.g., 0.06%)
    "rolling_window": 30,    # Rolling window (optional, kept for compatibility)
    "csv_file_path": 'models/lstm_results/merged_data_with_lstm_predictions.csv',  # Path to CSV
    "datetime_column": 'timestamp',  # Datetime column name
    "returns_column_btc": 'btc_return_1h',  # Returns column for BTC
    "returns_column_eth": 'eth_return_1h',  # Returns column for ETH (adjust if different)
    "fill_missing_values": True,  # Whether to fill missing values
    "forwardtest_period": ('2025-01-11', '2025-04-11')  # Adjusted to match sample data
}

def custom_generate_signals(bt):
    """
    Strategy using LSTM predictions for buy/sell signals.
    """
    df = bt.df
    df['signal'] = 0
    df.loc[df['lstm_predictions'] > bt.entry_threshold, 'signal'] = 1  # Buy signal
    df.loc[df['lstm_predictions'] < -bt.entry_threshold, 'signal'] = -1  # Sell signal
    df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)
    bt.df = df

def run_forward_test(asset):
    """
    Run forward test for the specified asset (BTC or ETH).
    """
    if asset == 'btc':
        returns_column = config['returns_column_btc']
        strategy_func = custom_generate_signals
    elif asset == 'eth':
        returns_column = config['returns_column_eth']
        strategy_func = custom_generate_signals
    else:
        raise ValueError(f"Unknown asset: {asset}")

    # Load and filter data
    df = pd.read_csv(config['csv_file_path'])
    df['Time'] = pd.to_datetime(df[config['datetime_column']])
    start_date, end_date = config['forwardtest_period']
    df = df[(df['Time'] >= start_date) & (df['Time'] <= end_date)]
    
    print(f"Data range for {asset}: {df['Time'].min()} to {df['Time'].max()}")
    if df.empty:
        raise ValueError(f"No data available for {asset} in period {start_date} to {end_date}")

    # Fill missing values
    if config['fill_missing_values']:
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

    # Check if returns column exists
    if returns_column not in df.columns:
        raise ValueError(f"Returns column '{returns_column}' not found in CSV for {asset}")

    # Initialize backtester
    bt = SimpleBacktester(
        df,
        generate_signals_func=strategy_func,
        returns_column=returns_column,
        entry_threshold=config['entry_threshold'],
        exit_threshold=config['exit_threshold'],
        fee_rate=config['fee_rate'],
        rolling_window=config['rolling_window']
    )

    # Run backtest
    print(f"Running forward test for {asset}...")
    bt.run_backtest()
    
    # Print metrics
    metrics = bt.calculate_metrics()
    print(f"Forward Test Metrics for {asset}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Calculate signal frequency
    signal_freq = (bt.results['signal'] != 0).sum() / len(bt.results) * 100
    print(f"Signal Frequency for {asset}: {signal_freq:.2f}%")
    
    # Generate plots
    bt.plot_strategy_performance()
    bt.plot_drawdowns()
    
    # # Save results
    # bt.results.to_csv(f'{asset}_lstm_forwardtest_results.csv')
    # print(f"Results saved to {asset}_lstm_forwardtest_results.csv")

if __name__ == "__main__":
    asset_to_test = config['asset_to_test']
    if asset_to_test == 'btc':
        try:
            run_forward_test('btc')
        except Exception as e:
            print(f"Error running forward test for BTC: {e}")
    elif asset_to_test == 'eth':
        try:
            run_forward_test('eth')
        except Exception as e:
            print(f"Error running forward test for ETH: {e}")
    elif asset_to_test == 'both':
        try:
            run_forward_test('btc')
        except Exception as e:
            print(f"Error running forward test for BTC: {e}")
        try:
            run_forward_test('eth')
        except Exception as e:
            print(f"Error running forward test for ETH: {e}")
    else:
        print("Invalid asset_to_test in config. Use 'btc', 'eth', or 'both'.")