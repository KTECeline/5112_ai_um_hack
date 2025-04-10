# jinyi file
# from backtest.loader import load_data
# from backtest.strategy import RegimeBasedStrategy
# from backtest.engine import run_backtest
# import pandas as pd

# # Load the data from CSV
# df = load_data("combined_crypto_data_with_hmm.csv")

# # Print the column names to the user to inspect them
# print("Columns in the loaded data:")
# print(df.columns)

# # Set up the strategy (e.g., based on market regimes or hidden states)
# strategy = RegimeBasedStrategy(df)

# # Generate signals based on the strategy
# df_with_signals = strategy.generate_signals()

# # Ask the user to specify which price column to use for backtesting
# print("\nPlease select the column for price data for backtesting:")
# print("Available columns:")
# print(df_with_signals.columns)

# # Example user input simulation:
# user_price_column = input("\nEnter the column name for the price data (e.g., 'close', 'price', etc.): ")

# # Run the backtest with the user-specified price column
# try:
#     result = run_backtest(df_with_signals, price_column=user_price_column)
    
#     # After the backtest, show the results to the user
#     print("Sharpe Ratio:", result.sharpe_ratio())
#     print("Max Drawdown:", result.max_drawdown())
#     print("Total Return:", result.total_return())

#     # Optionally, plot the results
#     result.plot_equity_curve()

# except KeyError as e:
#     print(f"Error: The specified price column '{user_price_column}' does not exist in the data.")
#     print(f"Available columns: {df_with_signals.columns}")


# celine btc_ml_ready
from backtest.loader import load_data
from backtest.strategy import RegimeBasedStrategy
from backtest.engine import run_backtest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the new data from CSV
df = load_data("api-ck/btc_ml_ready.csv")  # Ensure this path is correct

# Print the column names to the user to inspect them
print("Columns in the loaded data:")
print(df.columns)

# Custom strategy to handle user input
class CustomRegimeBasedStrategy(RegimeBasedStrategy):
    def __init__(self, data, window_size, threshold):
        super().__init__(data)
        self.window_size = window_size
        self.threshold = threshold

    def generate_signals(self):
        signals = []
        for i in range(len(self.data)):
            # Use the 'returns' or 'close' column instead of 'hidden_states'
            returns = self.data.iloc[i]['returns']  # Assuming 'returns' is available for signal generation
            volatility = self.data.iloc[i]['volatility']  # Assuming 'volatility' is available for volatility checks

            # Example signal generation based on returns and volatility:
            if returns > self.threshold and volatility < self.data['volatility'].median():
                signals.append(1)  # Buy (long)
            elif returns < -self.threshold and volatility > self.data['volatility'].median():
                signals.append(-1)  # Sell (short)
            else:
                signals.append(0)  # Hold (flat)

        self.data['signal'] = signals
        return self.data

    def win_rate(self):
        # Calculate win rate: percentage of profitable trades
        profitable_trades = self.data[self.data['signal'] == 1]['returns'] > 0
        return profitable_trades.mean() * 100  # Returns the win rate in percentage

    def trade_frequency(self):
        # Count the number of trades (number of signals other than 0)
        return (self.data['signal'] != 0).sum()

# Get user inputs
window_size = int(input("Enter the window size for the rolling statistics (e.g., 7 for 7-day moving average): "))
test_size = float(input("Enter the test size (as a proportion, e.g., 0.8 for 80% training data): "))
threshold = float(input("Enter the threshold for signal generation (e.g., 0.01): "))

# Split the data for training and testing based on test_size
train_size = int(len(df) * test_size)
train_data = df[:train_size]
test_data = df[train_size:]

# Initialize the strategy with the new data
strategy = CustomRegimeBasedStrategy(train_data, window_size, threshold)

# Generate signals based on the new strategy
df_with_signals = strategy.generate_signals()

# Ask the user to specify which price column to use for backtesting
print("\nPlease select the column for price data for backtesting:")
print("Available columns:")
print(df_with_signals.columns)

# Example user input simulation:
user_price_column = input("\nEnter the column name for the price data (e.g., close, inflow_inflow_mean): ")

# Run the backtest with the user-specified price column
try:
    result = run_backtest(df_with_signals, price_column=user_price_column)
    
    # After the backtest, show the results to the user
    print("Sharpe Ratio:", result.sharpe_ratio())
    print("Max Drawdown:", result.max_drawdown())
    print("Total Return:", result.total_return())
    print("Win Rate:", strategy.win_rate(), "%")
    print("Trade Frequency:", strategy.trade_frequency())

    # Optionally, plot the results
    result.plot_equity_curve()

    # Plot equity curve as additional visualization
    plt.figure(figsize=(10, 6))
    plt.plot(result.equity_curve['Date'], result.equity_curve['Equity'], label="Equity Curve")
    plt.title("Equity Curve of Strategy")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.show()

except KeyError as e:
    print(f"Error: The specified price column '{user_price_column}' does not exist in the data.")
    print(f"Available columns: {df_with_signals.columns}")
