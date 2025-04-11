import pandas as pd
from backtester import SimpleBacktester, plot_sharpe_heatmap

# Example signal generator strategy
def example_signal_strategy(bt):
    df = bt.df
    df['returns'] = df['Close'].pct_change().fillna(0)
    df['rolling_mean'] = df['Close'].rolling(bt.rolling_window).mean()
    df['zscore'] = (df['Close'] - df['rolling_mean']) / df['Close'].rolling(bt.rolling_window).std()

    df['signal'] = 0
    df.loc[df['zscore'] > bt.entry_threshold, 'signal'] = -1
    df.loc[df['zscore'] < -bt.entry_threshold, 'signal'] = 1
    df['position'] = df['signal'].ffill().fillna(0)

# Load and prepare historical data
df = pd.read_csv("api-ck/btc_ml_ready.csv")
df['Time'] = pd.to_datetime(df['datetime'])
df['Close'] = df['close']  # make sure there's a 'close' column

# Define parameter ranges
entry_thresholds = [0.5, 1.0, 1.5, 2.0]
rolling_windows = [10, 20, 30, 40]

# Plot the heatmap
plot_sharpe_heatmap(
    df=df,
    strategy_func=example_signal_strategy,
    entry_thresholds=entry_thresholds,
    rolling_windows=rolling_windows,
    exit_threshold=1.0,
    fee_rate=0.0006
)
