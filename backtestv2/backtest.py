# backtest.py (Modified single-file version of Backtesting.py)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Backtest:
    def __init__(self, data, strategy, cash=10000, commission=0.0):
        self.data = data.copy()
        self.strategy = strategy
        self.initial_cash = cash
        self.commission = commission
        self.equity_curve = []
        self.trades = []

    def run(self):
        self.data['Position'] = self.strategy(self.data)
        self.data['Market Return'] = self.data['Close'].pct_change().fillna(0)
        self.data['Strategy Return'] = self.data['Position'].shift(1) * self.data['Market Return']
        self.data['Strategy Return'] -= self.data['Position'].diff().abs() * self.commission
        self.data['Equity Curve'] = (1 + self.data['Strategy Return']).cumprod() * self.initial_cash
        return self.data

    def stats(self):
        total_return = self.data['Equity Curve'].iloc[-1] / self.initial_cash - 1
        sharpe = np.mean(self.data['Strategy Return']) / np.std(self.data['Strategy Return']) * np.sqrt(252)
        equity = self.data['Equity Curve']
        dd = (equity / equity.cummax() - 1).min()
        return {
            'Total Return': total_return,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': dd
        }

    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Equity Curve'], label="Strategy")
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.grid()
        plt.show()
