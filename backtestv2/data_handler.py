import pandas as pd


def load_data(file_path):
    """
    Load the dataset and preprocess it.
    Args:
        file_path (str): Path to the dataset (CSV).
    Returns:
        pd.DataFrame: Preprocessed data.
    """
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df