import numpy as np
from backtest.result import BacktestResult

def run_backtest(data, price_column='4', signal_column='signal', fee=0.001):
    data = data.copy()
    data['returns'] = data[price_column].pct_change()
    data['strategy_returns'] = data[signal_column].shift(1) * data['returns']
    data['strategy_returns'] -= fee * np.abs(data[signal_column].diff())

    data.dropna(inplace=True)
    
    result = BacktestResult(
        returns=data['strategy_returns'],
        signals=data[signal_column],
        timestamps=data.index,
        raw_data=data
    )
    return result
