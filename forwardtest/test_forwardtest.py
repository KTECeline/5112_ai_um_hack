# test_forwardtest.py
import pandas as pd
from backtester import SimpleBacktester

# Configuration settings (editable by user)
config = {
    "entry_threshold": 1.5,      # Fixed entry threshold from backtest optimization
    "exit_threshold": 1.0,       # Fixed exit threshold
    "fee_rate": 0.0006,          # Fee rate per trade (e.g., 0.06%)
    "rolling_window": 20,        # Fixed rolling window from backtest
    "csv_file_path_btc": 'api-ck/btc_ml_ready.csv',  # BTC data path
    "csv_file_path_eth": 'api-ck/eth_ml_ready.csv',  # ETH data path
    "datetime_column": 'datetime',  # Column name for datetime
    "fill_missing_values": True,    # Whether to fill missing values
    "forwardtest_period": ('2025-01-01', '2025-04-10')  # Forward test date range
}

# STRATEGY INSERT (User can insert strategies based on the data input)
# For BTC
def custom_generate_signals_btc(bt):
    """
    Signal generation for BTC based on z-score of inflow_inflow_mean.
    Matches test_main.py logic.
    """
    df = bt.df
    df['zscore'] = (df['inflow_inflow_mean'] - df['inflow_inflow_mean_ma7']) / df['inflow_inflow_mean'].rolling(window=bt.rolling_window).std()
    df['signal'] = 0
    df.loc[df['zscore'] > bt.entry_threshold, 'signal'] = -1  # Sell
    df.loc[df['zscore'] < -bt.entry_threshold, 'signal'] = 1   # Buy
    df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)
    bt.df = df

#For ETH (placeholder, with your commented example from test_main.py)
def custom_generate_signals_eth(bt):
    """
    Signal generation for ETH based on volatility and volume z-scores.
    User can modify this logic.
    """
    df = bt.df
    df['volatility_zscore'] = (df['volatility'] - df['volatility'].rolling(window=bt.rolling_window).mean()) / df['volatility'].rolling(window=bt.rolling_window).std()
    df['volume_zscore'] = (df['volume'] - df['volume_ma']) / df['volume'].rolling(window=bt.rolling_window).std()
    df['signal'] = 0
    df.loc[(df['volatility_zscore'] > bt.entry_threshold) & (df['volume_zscore'] > bt.entry_threshold), 'signal'] = -1  # Sell
    df.loc[(df['volatility_zscore'] < -bt.entry_threshold) & (df['volume_zscore'] < -bt.entry_threshold), 'signal'] = 1  # Buy
    df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)
    bt.df = df

def run_forward_test(asset):
    # Select CSV path and strategy based on asset
    if asset == 'btc':
        csv_path = config['csv_file_path_btc']
        strategy_func = custom_generate_signals_btc
    elif asset == 'eth':
        csv_path = config['csv_file_path_eth']
        strategy_func = custom_generate_signals_eth
    else:
        raise ValueError(f"Unknown asset: {asset}")

    # Load the historical data
    df = pd.read_csv(csv_path)
    df['Time'] = pd.to_datetime(df[config['datetime_column']])

    # Filter for forward test period
    start_date, end_date = config['forwardtest_period']
    df = df[(df['Time'] >= start_date) & (df['Time'] <= end_date)]
    if df.empty:
        raise ValueError(f"No data available for {asset} in period {start_date} to {end_date}")

    # Fill missing values if needed
    if config['fill_missing_values']:
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

    # Create the backtester object with fixed strategy
    bt = SimpleBacktester(
        df,
        generate_signals_func=strategy_func,
        entry_threshold=config['entry_threshold'],
        exit_threshold=config['exit_threshold'],
        fee_rate=config['fee_rate'],
        rolling_window=config['rolling_window']
    )

    # Run the forward test
    bt.run_backtest()

    # Print performance metrics
    metrics = bt.calculate_metrics()
    print(f"Forward Test Metrics for {asset}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Plot the results
    bt.plot()

    # Save results to CSV
    # bt.results.to_csv(f'{asset}_forwardtest_results.csv')
    # print(f"Results saved to {asset}_forwardtest_results.csv")

# Run forward tests
try:
    run_forward_test('btc')
except Exception as e:
    print(f"Error running forward test for BTC: {e}")

try:
    run_forward_test('eth')
except Exception as e:
    print(f"Error running forward test for ETH: {e}")