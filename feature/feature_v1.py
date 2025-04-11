import pandas as pd
import numpy as np

def calculate_rsi(df, window=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def feature_engineering(df):
    # Convert 'timestamp' to datetime if needed
    if 'datetime' not in df.columns and 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    # Ensure 'datetime' is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Drop NaT and duplicates based on datetime
    df = df.dropna(subset=['datetime'])
    df = df.drop_duplicates(subset='datetime')

    # Set datetime as index
    df = df.set_index('datetime')

    # 1. Price-based Features
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=7).std()

    # 2. Moving Averages
    df['sma_7'] = df['close'].rolling(window=7).mean()
    df['sma_30'] = df['close'].rolling(window=30).mean()

    # 3. Technical Indicators (RSI)
    df['rsi'] = calculate_rsi(df)

    # 4. Netflow Ratio
    if 'inflow' in df.columns and 'outflow' in df.columns:
        df['netflow_ratio'] = df['inflow'] / (df['outflow'] + 1e-5)

    # 5. Lag Features (previous closes)
    df['close_lag_1'] = df['close'].shift(1)
    df['close_lag_2'] = df['close'].shift(2)

    # 6. Time-based Features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek

    # 7. Target Feature (Future Close Price)
    df['future_close'] = df['close'].shift(-1)

    # 8. Add additional columns as needed
    df['volume_ma'] = df['volume'].rolling(window=7).mean()  # Moving average of volume
    df['volatility'] = df['returns'].rolling(window=7).std()  # Volatility based on returns

    # 9. Add 'utxo_created_value_median' if it's in your dataset
    if 'utxo_created_value_median' in df.columns:
        df['utxo_created_value_median'] = df['utxo_created_value_median']

    # 10. Add 'hidden_states' column (this will be added after HMM modeling, placeholder here)
    df['hidden_states'] = np.nan  # Placeholder, will be populated later

    # Drop any rows with NaN values (result of shifting)
    df = df.dropna()

    # Create target and features
    target_columns = ['future_close']
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'inflow', 'outflow', 
                       'netflow_ratio', 'sma_7', 'sma_30', 'rsi', 'returns', 'volatility', 
                       'volume_ma', 'close_lag_1', 'close_lag_2', 'utxo_created_value_median', 'hour', 'day_of_week']

    # Ensure there are no missing values before returning
    df = df[feature_columns + target_columns].dropna()

    return df

def preprocess_data(df):
    # Feature Scaling (MinMaxScaler)
    from sklearn.preprocessing import MinMaxScaler
    
    # Selecting the features for scaling
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume', 'inflow', 'outflow', 
                                               'netflow_ratio', 'sma_7', 'sma_30', 'rsi', 'returns', 
                                               'volatility', 'volume_ma', 'close_lag_1', 'close_lag_2', 
                                               'utxo_created_value_median', 'hour', 'day_of_week']])
    
    df_scaled = pd.DataFrame(scaled_features, columns=['open', 'high', 'low', 'close', 'volume', 'inflow', 'outflow', 
                                                       'netflow_ratio', 'sma_7', 'sma_30', 'rsi', 'returns', 
                                                       'volatility', 'volume_ma', 'close_lag_1', 'close_lag_2', 
                                                       'utxo_created_value_median', 'hour', 'day_of_week'])
    df[df_scaled.columns] = df_scaled

    return df

def create_sequences(df, lookback=30, target_column='future_close'):
    """Prepare data for LSTM: Convert the data into sequences of (X, y) pairs."""
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

    # Create the features and target variable
    X = df[['open', 'high', 'low', 'close', 'volume', 'inflow', 'outflow', 'netflow_ratio', 
            'sma_7', 'sma_30', 'rsi', 'returns', 'volatility', 'volume_ma', 'close_lag_1', 'close_lag_2', 
            'utxo_created_value_median', 'hour', 'day_of_week']].values
    y = df[target_column].values

    # Use TimeseriesGenerator for LSTM data formatting
    generator = TimeseriesGenerator(X, y, length=lookback, batch_size=32)
    
    return generator

def feature_pipeline(df):
    # Check the columns of the DataFrame
    print("Columns in the dataset:", df.columns)

    # Ensure the datetime column exists and is in the correct format
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')  # Convert to datetime format
        df['hour'] = df['datetime'].dt.hour  # Extract the hour from the datetime
        df['day_of_week'] = df['datetime'].dt.dayofweek  # Extract the day of the week (0=Monday, 6=Sunday)
    else:
        print("Warning: 'datetime' column not found in the dataset.")
    
    # Calculate the 'returns' column if it doesn't exist
    if 'returns' not in df.columns:
        df['returns'] = df['close'].pct_change()  # Calculate returns (percentage change)
    
    # Calculate volatility (if needed, based on returns or other columns)
    if 'volatility' not in df.columns:
        df['volatility'] = df['returns'].rolling(window=20).std()  # Rolling standard deviation as volatility
    
    # Calculate volume moving average (use inflow_total instead of volume)
    if 'volume_ma' not in df.columns:
        df['volume_ma'] = df['inflow_total'].rolling(window=20).mean()  # Use 'inflow_total' as volume proxy
    
    # Calculate netflow ratio (example calculation, adjust as necessary)
    if 'netflow_ratio' not in df.columns:
        df['netflow_ratio'] = df['inflow_total'] / (df['outflow_total'] + 1)  # Example formula for netflow ratio
    
  
    return df

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv(r'..\data\data\combined_dataset.csv')
    
    # Apply the feature pipeline
    df = feature_pipeline(df)
    
    # Save the processed data
    df.to_csv('processed_feature_data.csv', index=False)
    
    print("Feature pipeline completed. Data saved to 'processed_feature_data.csv'.")