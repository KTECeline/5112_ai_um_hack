# forwardtest.py
import sys
import os
print("Python path:", sys.path)
print("Current directory:", os.getcwd())
from backtester import SimpleBacktester
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data
from config import ASSETS, FEE_RATE, ENTRY_THRESHOLD, EXIT_THRESHOLD, ROLLING_WINDOW


def generate_signals_btc(bt):
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

def generate_signals_eth(bt):
    """
    Signal generation for ETH based on z-score of volatility.
    Placeholder; adjust based on your ETH strategy.
    """
    df = bt.df
    df['zscore'] = (df['volatility'] - df['volatility'].rolling(window=bt.rolling_window).mean()) / df['volatility'].rolling(window=bt.rolling_window).std()
    df['signal'] = 0
    df.loc[df['zscore'] > bt.entry_threshold, 'signal'] = -1  # Sell
    df.loc[df['zscore'] < -bt.entry_threshold, 'signal'] = 1   # Buy
    df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)
    bt.df = df

def run_forward_test(asset):
    """
    Run forward test for the specified asset.
    
    Args:
        asset (str): 'btc' or 'eth'.
    
    Returns:
        dict: Forward test results.
    """
    print(f"Running forward test for {asset}...")
    
    # Load forward test data
    df = load_data(asset, period='forward')
    if df.empty:
        raise ValueError(f"No data available for {asset}")
    
    # Select signal generation function
    # Comment/uncomment to switch between BTC and ETH
    generate_signals_func = generate_signals_btc  # For BTC
    # generate_signals_func = generate_signals_eth  # For ETH
    
    # Initialize backtester
    bt = SimpleBacktester(
        df=df,
        generate_signals_func=generate_signals_func,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        fee_rate=FEE_RATE,
        rolling_window=ROLLING_WINDOW
    )
    
    # Run backtest (same logic as forward test)
    bt.run_backtest()
    
    # Calculate metrics
    metrics = bt.calculate_metrics()
    
    # Print metrics
    print(f"Forward Test Metrics for {asset}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(df['Time'], bt.results['market'], label='Market Returns', linestyle='--', alpha=0.6)
    plt.plot(df['Time'], bt.results['net_equity'], label='Strategy Equity (net of fees)', linewidth=2)
    
    buy_signals = bt.results[bt.results['signal'] == 1]
    sell_signals = bt.results[bt.results['signal'] == -1]
    
    plt.scatter(buy_signals['Time'], bt.results.loc[buy_signals.index, 'net_equity'], label='Buy', marker='^', color='green')
    plt.scatter(sell_signals['Time'], bt.results.loc[sell_signals.index, 'net_equity'], label='Sell', marker='v', color='red')
    
    plt.title(f'{asset.upper()} Forward Test: Strategy Equity vs Market')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{asset}_forwardtest_equity.png')
    plt.show()
    
    return {
        'results': bt.results,
        'metrics': metrics
    }

def main():
    # Run forward test for BTC
    try:
        btc_results = run_forward_test('btc')
        btc_results['results'].to_csv('btc_forwardtest_results.csv')
    except Exception as e:
        print(f"Error running forward test for BTC: {e}")
    
    # Run forward test for ETH (uncomment to enable)
    # try:
    #     eth_results = run_forward_test('eth')
    #     eth_results['results'].to_csv('eth_forwardtest_results.csv')
    # except Exception as e:
    #     print(f"Error running forward test for ETH: {e}")

if __name__ == "__main__":
    main()