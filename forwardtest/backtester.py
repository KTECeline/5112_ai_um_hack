# backtester.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

class SimpleBacktester:
    def __init__(self, df, generate_signals_func=None, entry_threshold=1.0, exit_threshold=0.0, fee_rate=0.0006, rolling_window=20):
        self.df = df.copy()
        self.generate_signals_func = generate_signals_func  # Accept strategy as a parameter
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.fee_rate = fee_rate
        self.rolling_window = rolling_window  # Add rolling window parameter
        self.results = None

    def run_backtest(self):
        if self.generate_signals_func:
            self.generate_signals_func(self)  # Use the strategy function provided
        else:
            raise ValueError("No signal generation function provided.")
        
        self.df['strategy_returns'] = self.df['position'].shift(1) * self.df['returns']
        self.df['equity'] = (1 + self.df['strategy_returns']).cumprod()
        self.df['market'] = (1 + self.df['returns']).cumprod()
        self.df['trades'] = self.df['position'].diff().abs()
        self.df['fees'] = self.df['trades'] * self.fee_rate
        self.df['net_equity'] = self.df['equity'] * (1 - self.df['fees'].cumsum())
        self.results = self.df

    def calculate_metrics(self):
        if self.results is None:
            raise ValueError("Run backtest first.")
        
        # Calculate strategy returns and drawdown
        returns = self.results['strategy_returns'].dropna()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
        drawdown = (self.results['net_equity'] / self.results['net_equity'].cummax() - 1)
        max_dd = drawdown.min()

        # Win rate
        win_rate = (self.results[self.results['strategy_returns'] > 0].shape[0] / self.results.shape[0]) * 100

        # Profit factor
        gross_profit = self.results[self.results['strategy_returns'] > 0]['strategy_returns'].sum()
        gross_loss = self.results[self.results['strategy_returns'] < 0]['strategy_returns'].sum()
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')

        # ROI
        roi = (self.results['net_equity'].iloc[-1] - 1) * 100

        # Return the calculated metrics
        return {
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd,
            "Trades per Interval": self.results['trades'].sum() / len(self.results),
            "Total Fees": self.results['fees'].sum(),
            "Win Rate (%)": win_rate,
            "Profit Factor": profit_factor,
            "ROI (%)": roi,
        }

    def plot(self):
        if self.results is None:
            raise ValueError("Run backtest first.")
        
        df = self.results
        plt.figure(figsize=(14, 7))
        plt.plot(df['Time'], df['market'], label='Market Returns', linestyle='--', alpha=0.6)
        plt.plot(df['Time'], df['net_equity'], label='Strategy Equity (net of fees)', linewidth=2)

        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]

        plt.scatter(buy_signals['Time'], df.loc[buy_signals.index, 'net_equity'], label='Buy', marker='^', color='green')
        plt.scatter(sell_signals['Time'], df.loc[sell_signals.index, 'net_equity'], label='Sell', marker='v', color='red')

        plt.title('Strategy Equity vs Market')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_sharpe_heatmap(df, strategy_func, entry_thresholds, rolling_windows, exit_threshold=0.0, fee_rate=0.0006):
    heatmap_data = []

    for entry in entry_thresholds:
        for window in rolling_windows:
            bt = SimpleBacktester(
                df=df,
                generate_signals_func=strategy_func,
                entry_threshold=entry,
                exit_threshold=exit_threshold,
                fee_rate=fee_rate,
                rolling_window=window
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

    cmap = LinearSegmentedColormap.from_list("red_green", ["red", "green"])
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_pivot, annot=True, cmap=cmap, fmt='.2f', cbar_kws={'label': 'Sharpe Ratio'})
    plt.title("Sharpe Ratio Heatmap - Entry Threshold vs Rolling Window")
    plt.xlabel("Entry Threshold")
    plt.ylabel("Rolling Window")
    plt.tight_layout()
    plt.show()