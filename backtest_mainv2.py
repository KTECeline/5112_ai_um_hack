import matplotlib.pyplot as plt
from backtestv2.data_handler import load_data
from backtestv2.strategy_engine import generate_signals
from backtestv2.portfolio_simulator import simulate_portfolio
from backtestv2.risk_manager import apply_risk_management
from backtestv2.metrics_calculator import calculate_metrics

def run_backtest(file_path, fees=0.06/100, slippage=0.1/100, stop_loss_pct=0.05):
    # Load data
    df = load_data(file_path)

    # Generate trading signals
    df['position'] = generate_signals(df)

    # Simulate the portfolio and calculate equity curve
    df = simulate_portfolio(df, fees, slippage)

    # Apply risk management techniques (stop-loss, max drawdown)
    df = apply_risk_management(df, stop_loss_pct)

    # Calculate key metrics
    metrics = calculate_metrics(df)

    # Print metrics
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']}")
    print(f"Trades per Interval: {metrics['trades_per_interval']}")

    # Plot equity curve and close price
    plt.figure(figsize=(12,6))

    plt.subplot(2, 1, 1)
    plt.plot(df['datetime'], df['equity_curve'], label="Equity Curve", color='green')
    plt.title('Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Equity')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df['datetime'], df['close'], label="Close Price", color='orange')
    plt.title('Close Price')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define the path to your data file
    file_path = 'api-ck/btc_ml_ready.csv'  # Replace with your file path

    # You can change these parameters as needed, or leave them to use defaults
    fees = 0.06 / 100  # Example fee
    slippage = 0.1 / 100  # Example slippage
    stop_loss_pct = 0.05  # Example stop-loss percentage

    # Run the backtest with the specified parameters
    run_backtest(file_path, fees, slippage, stop_loss_pct)
