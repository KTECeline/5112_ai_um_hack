# metrics.py
import numpy as np

def calculate_metrics(equity_curve, returns, trades, initial_capital, risk_free_rate=0.02):
    """
    Calculate backtest/forward test metrics.
    
    Args:
        equity_curve (pd.Series): Portfolio value over time.
        returns (pd.Series): Trade returns.
        trades (list): List of trade dicts with 'profit', 'fee', etc.
        initial_capital (float): Starting capital.
        risk_free_rate (float): Annual risk-free rate for Sharpe.
    
    Returns:
        dict: Metrics including Sharpe, drawdown, etc.
    """
    metrics = {}
    
    # Annualized Sharpe Ratio
    if len(returns) > 0 and returns.std() != 0:
        annualized_returns = returns.mean() * 365 * 24  # Assuming hourly data
        annualized_vol = returns.std() * np.sqrt(365 * 24)
        metrics['sharpe_ratio'] = (annualized_returns - risk_free_rate) / annualized_vol
    else:
        metrics['sharpe_ratio'] = 0.0
    
    # Max Drawdown
    roll_max = equity_curve.cummax()
    drawdowns = (roll_max - equity_curve) / roll_max
    metrics['max_drawdown'] = drawdowns.max() if len(drawdowns) > 0 else 0.0
    
    # Trades per Interval (assuming hourly data, annualized)
    metrics['trades_per_interval'] = len(trades) / (len(equity_curve) / (365 * 24))
    
    # Total Fees
    metrics['total_fees'] = sum(trade.get('fee', 0) for trade in trades)
    
    # Win Rate
    wins = sum(1 for trade in trades if trade.get('profit', 0) > 0)
    metrics['win_rate'] = wins / len(trades) if trades else 0.0
    
    # Profit Factor
    gross_profits = sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0)
    gross_losses = sum(-trade.get('profit', 0) for trade in trades if trade.get('profit', 0) < 0)
    metrics['profit_factor'] = gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    # ROI (%)
    final_equity = equity_curve.iloc[-1] if len(equity_curve) > 0 else initial_capital
    metrics['roi'] = (final_equity - initial_capital) / initial_capital * 100
    
    return metrics