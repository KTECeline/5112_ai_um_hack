import numpy as np

def generate_signals(df):
    """
    Generate trading signals based on a simple strategy (price change).
    Args:
        df (pd.DataFrame): The input dataset.
    Returns:
        pd.Series: Buy (1) / Sell (-1) signals.
    """
    df['price_change'] = df['close'].pct_change()
    df['position'] = np.where(df['price_change'] > 0, 1, -1)  # 1 for Buy, -1 for Sell
    return df['position']
