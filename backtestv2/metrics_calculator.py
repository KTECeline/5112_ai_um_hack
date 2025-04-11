import numpy as np




def calculate_metrics(df):
    """
    Calculate key performance metrics.
    Args:
        df (pd.DataFrame): The dataset with portfolio data.
    Returns:
        dict: Calculated metrics (Sharpe ratio, max drawdown, etc.)
    """
    # Sharpe Ratio (based on daily returns)
    df['daily_return'] = df['pnl'].pct_change()
    sharpe_ratio = df['daily_return'].mean() / df['daily_return'].std() * np.sqrt(252)  # Adjust for annualization


    # Maximum Drawdown
    max_drawdown = df['drawdown'].min()


    # Trades per interval
    trades_per_interval = df['position'].abs().sum()


    return {
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trades_per_interval': trades_per_interval,
    }


