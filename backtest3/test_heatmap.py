import pandas as pd
from backtester import SimpleBacktester, plot_sharpe_heatmap

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

# btc strategy
# def custom_generate_signals(self):
#     # Use the rolling_window for the zscore calculation
#     
#     self.df['zscore'] = (self.df['inflow_inflow_mean'] - self.df['inflow_inflow_mean_ma7']) / self.df['inflow_inflow_mean'].rolling(window=self.rolling_window).std()
#     self.df['signal'] = 0
#     self.df.loc[self.df['zscore'] > self.entry_threshold, 'signal'] = -1  # sell
#     self.df.loc[self.df['zscore'] < -self.entry_threshold, 'signal'] = 1  # buy
#     self.df['position'] = self.df['signal'].replace(to_replace=0, method='ffill').fillna(0)

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
    strategy_func=custom_generate_signals,
    entry_thresholds=entry_thresholds,
    rolling_windows=rolling_windows,
    exit_threshold=1.0,
    fee_rate=0.0006
)
