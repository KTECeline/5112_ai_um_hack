import numpy as np
import matplotlib.pyplot as plt

class BacktestResult:
    def __init__(self, returns, signals, timestamps, raw_data):
        self.returns = returns
        self.signals = signals
        self.timestamps = timestamps
        self.raw_data = raw_data

    def sharpe_ratio(self):
        return np.mean(self.returns) / np.std(self.returns) * np.sqrt(252)

    def max_drawdown(self):
        cum_returns = (1 + self.returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        return drawdown.min()

    def total_return(self):
        return (1 + self.returns).prod() - 1

    def plot_equity_curve(self):
        cum_returns = (1 + self.returns).cumprod()
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, cum_returns, label="Strategy Equity Curve")
        plt.title("Equity Curve")
        plt.legend()
        plt.grid()
        plt.show()
