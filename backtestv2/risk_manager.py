import numpy as np

def apply_risk_management(df, stop_loss_pct=0.05):
    """
    Apply risk management techniques like stop-loss and max drawdown.
    Args:
        df (pd.DataFrame): The dataset with positions and equity.
        stop_loss_pct (float): Percentage threshold for stop-loss.
    Returns:
        pd.DataFrame: Updated dataframe with stop-loss applied.
    """
    df['drawdown'] = df['equity_curve'] - df['equity_curve'].cummax()
    df['stop_loss'] = df['equity_curve'] < df['equity_curve'].cummax() * (1 - stop_loss_pct)
   
    # Apply stop-loss (close positions if drawdown exceeds threshold)
    df['position'] = np.where(df['stop_loss'], 0, df['position'])
   
    # Ensure that drawdown is recalculated with stop-loss applied
    df['drawdown'] = df['equity_curve'] - df['equity_curve'].cummax()
    return df
