# data_loader.py
import pandas as pd
from config import ASSETS, FORWARDTEST_PERIOD, BACKTEST_PERIOD, FILL_MISSING_VALUES

def load_data(asset, period='forward'):
    """
    Load and preprocess CSV data for the specified asset and period.
    
    Args:
        asset (str): 'btc' or 'eth'.
        period (str): 'backtest' or 'forward'.
    
    Returns:
        pd.DataFrame: Filtered data.
    """
    config = ASSETS.get(asset)
    if not config:
        raise ValueError(f"Unknown asset: {asset}")
    
    df = pd.read_csv(config['csv_file_path'])
    df['Time'] = pd.to_datetime(df[config['datetime_column']])
    df.set_index('Time', inplace=True)
    
    # Filter by date range
    start_date, end_date = FORWARDTEST_PERIOD if period == 'forward' else BACKTEST_PERIOD
    df = df.loc[start_date:end_date]
    
    # Fill missing values
    if FILL_MISSING_VALUES:
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
    
    # Ensure required columns
    required_cols = ['returns', 'inflow_inflow_mean', 'inflow_inflow_mean_ma7'] if asset == 'btc' else ['returns', 'volatility']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns for {asset}: {required_cols}")
    
    return df