import numpy as np

def simulate_portfolio(df, fees=0.06 / 100, slippage=0.1 / 100):
    """
    Simulate portfolio: calculate PnL and equity curve.
    Args:
        df (pd.DataFrame): The dataset with positions and price changes.
        fees (float): The transaction fee per trade.
        slippage (float): The slippage per trade.
    Returns:
        pd.DataFrame: Updated dataframe with PnL and equity curve.
    """
    df['pnl'] = df['position'] * df['price_change'] - fees * np.abs(df['position']) - slippage * np.abs(df['position'])
    df['equity_curve'] = df['pnl'].cumsum()
    return df