# config.py

# Common settings
FORWARDTEST_PERIOD = ('2025-01-01', '2025-04-10')  # Forward test data range
BACKTEST_PERIOD = ('2022-04-11', '2024-12-31')    # Backtest data range (for reference)
FEE_RATE = 0.0006                                 # 0.06% fee per trade
ENTRY_THRESHOLD = 1.5                             # Z-score threshold for entry
EXIT_THRESHOLD = 1.0                              # Z-score threshold for exit
ROLLING_WINDOW = 20                               # Rolling window for z-score
FILL_MISSING_VALUES = True                        # Fill missing values in data

# Asset-specific settings
ASSETS = {
    'btc': {
        'csv_file_path': 'api-ck/btc_ml_ready.csv',
        'datetime_column': 'datetime',
        'signal_column': 'inflow_inflow_mean',  # For z-score calculation
        'signal_ma_column': 'inflow_inflow_mean_ma7'
    },
    'eth': {
        'csv_file_path': 'api-ck/eth_ml_ready.csv',
        'datetime_column': 'datetime',
        'signal_column': 'volatility',  # Example for ETH; adjust if different
        'signal_ma_column': None        # ETH may not use MA; adjust as needed
    }
}