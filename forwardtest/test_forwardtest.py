# test_forwardtest.py
import pandas as pd
from backtester import SimpleBacktester

# Configuration settings (editable by user)
config = {
    "asset_to_test": "eth",  # Options: 'btc', 'eth', 'both'
    "entry_threshold": 1.5,
    "exit_threshold": 1.0,
    "fee_rate": 0.0006,
    "rolling_window": 20,
    "csv_file_path_btc": 'api-ck/btc_ml_ready.csv',  # Update if changed
    "csv_file_path_eth": 'api-ck/eth_ml_ready.csv',  # Update if changed
    "datetime_column": 'datetime',
    "fill_missing_values": True,
    "forwardtest_period": ('2025-01-01', '2025-04-10')
}

def custom_generate_signals_btc(bt):
    df = bt.df
    df['zscore'] = (df['inflow_inflow_mean'] - df['inflow_inflow_mean_ma7']) / df['inflow_inflow_mean'].rolling(window=bt.rolling_window).std()
    df['signal'] = 0
    df.loc[df['zscore'] > bt.entry_threshold, 'signal'] = -1
    df.loc[df['zscore'] < -bt.entry_threshold, 'signal'] = 1
    df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)
    bt.df = df

def custom_generate_signals_eth(bt):
    df = bt.df
    df['volatility_zscore'] = (df['volatility'] - df['volatility'].rolling(window=bt.rolling_window).mean()) / df['volatility'].rolling(window=bt.rolling_window).std()
    df['volume_zscore'] = (df['volume'] - df['volume_ma']) / df['volume'].rolling(window=bt.rolling_window).std()
    df['signal'] = 0
    df.loc[(df['volatility_zscore'] > bt.entry_threshold) & (df['volume_zscore'] > bt.entry_threshold), 'signal'] = -1
    df.loc[(df['volatility_zscore'] < -bt.entry_threshold) & (df['volume_zscore'] < -bt.entry_threshold), 'signal'] = 1
    df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)
    bt.df = df

def run_forward_test(asset):
    if asset == 'btc':
        csv_path = config['csv_file_path_btc']
        strategy_func = custom_generate_signals_btc
    elif asset == 'eth':
        csv_path = config['csv_file_path_eth']
        strategy_func = custom_generate_signals_eth
    else:
        raise ValueError(f"Unknown asset: {asset}")

    df = pd.read_csv(csv_path)
    df['Time'] = pd.to_datetime(df[config['datetime_column']])
    start_date, end_date = config['forwardtest_period']
    df = df[(df['Time'] >= start_date) & (df['Time'] <= end_date)]
    print(f"Data range for {asset}: {df['Time'].min()} to {df['Time'].max()}")
    if df.empty:
        raise ValueError(f"No data available for {asset} in period {start_date} to {end_date}")

    if config['fill_missing_values']:
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

    bt = SimpleBacktester(
        df,
        generate_signals_func=strategy_func,
        entry_threshold=config['entry_threshold'],
        exit_threshold=config['exit_threshold'],
        fee_rate=config['fee_rate'],
        rolling_window=config['rolling_window']
    )

    print(f"Running forward test for {asset}...")
    bt.run_backtest()
    metrics = bt.calculate_metrics()
    print(f"Forward Test Metrics for {asset}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    signal_freq = (bt.results['signal'] != 0).sum() / len(bt.results) * 100
    print(f"Signal Frequency for {asset}: {signal_freq:.2f}%")
    bt.plot()
    bt.results.to_csv(f'{asset}_forwardtest_results.csv')
    print(f"Results saved to {asset}_forwardtest_results.csv")

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