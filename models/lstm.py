import pandas as pd
import numpy as np
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# =============================================
# 1. Configuration
# =============================================
ML_DATA_DIR = Path("../feature/ml_data")
OUTPUT_DIR = Path("hmm_results")
LSTM_DIR = Path("lstm_results")
LSTM_DIR.mkdir(exist_ok=True)

# LSTM Parameters
SEQUENCE_LENGTH = 24  # 1 day of hourly data
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# =============================================
# 2. Load and Merge Data
# =============================================
def load_and_merge_data():
    try:
        data_path = ML_DATA_DIR / "ml_ready_data.csv"
        regimes_path = OUTPUT_DIR / "hmm_regimes.csv"
        
        if not data_path.exists():
            raise FileNotFoundError(f"ML-ready data not found at {data_path}")
        if not regimes_path.exists():
            raise FileNotFoundError(f"HMM regimes not found at {regimes_path}")
        
        # Load the main data and the HMM regimes
        df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
        regimes_df = pd.read_csv(regimes_path, parse_dates=['timestamp'], index_col='timestamp')
        
        # Merge the dataframes on the timestamp
        df = df.join(regimes_df[['regime']], how='inner')
        
        if 'regime' not in df.columns:
            raise ValueError("❌ 'regime' column not found after merge")
        
        # Fill any missing values
        if df.isna().any().any():
            df.fillna(0, inplace=True)
            print("⚠️ Filled missing values with 0")
        
        print(f"✅ Loaded and merged data with shape: {df.shape}")
        output_file = OUTPUT_DIR / "merged_data_with_regime.csv"
        df.to_csv(output_file)
        print(f"✅ Merged data saved to {output_file}")
        return df
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


# =============================================
# 3. Prepare Sequences
# =============================================
def prepare_sequences(df, sequence_length=SEQUENCE_LENGTH):
    try:
        # Use all columns except the 'target' for features
        feature_cols = [col for col in df.columns if col not in ['target']]
        X = df[feature_cols].values
        y = df['target'].values

        # Scale the features and target
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)

        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

        # Prepare sequences for LSTM
        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - sequence_length):
            X_seq.append(X_scaled[i:i + sequence_length])
            y_seq.append(y_scaled[i + sequence_length])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        # Split into train and test datasets
        train_size = int(len(X_seq) * (1 - VALIDATION_SPLIT))
        X_train = X_seq[:train_size]
        y_train = y_seq[:train_size]
        X_test = X_seq[train_size:]
        y_test = y_seq[train_size:]

        print(f"✅ Prepared sequences - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
        return X_train, y_train, X_test, y_test, scaler_X, scaler_y

    except Exception as e:
        print(f"❌ Error preparing sequences: {e}")
        return None, None, None, None, None, None

# =============================================
# 4. Define PyTorch LSTM Model
# =============================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# =============================================
# 5. Train the LSTM Model
# =============================================
def train_lstm(X_train, y_train, X_test, y_test):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = X_train.shape[2]

        model = LSTMModel(input_size).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                     torch.tensor(y_test, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        train_losses, val_losses = [], []

        for epoch in range(EPOCHS):
            model.train()
            total_train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    total_val_loss += loss.item()

            train_loss = total_train_loss / len(train_loader)
            val_loss = total_val_loss / len(test_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"📘 Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        return model, train_losses, val_losses

    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None, None, None

# =============================================
# 6. Save and Visualize Results
# =============================================
def save_and_visualize(model, train_losses, val_losses, X_test, y_test, scaler_y, output_dir):
    try:
        # Save model
        model_path = output_dir / "lstm_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"💾 Saved PyTorch model to {model_path}")

        # Predict on test data
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            predictions = model(X_test_tensor).cpu().numpy()

        y_test_inv = scaler_y.inverse_transform(y_test)
        y_pred_inv = scaler_y.inverse_transform(predictions)

        mse = np.mean((y_test_inv - y_pred_inv) ** 2)
        print(f"Test MSE: {mse:.6f}")

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('LSTM Training History (PyTorch)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.savefig(output_dir / "lstm_training_history.png")
        plt.close()

        # Plot predictions
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_inv, label='Actual Target')
        plt.plot(y_pred_inv, label='Predicted Target')
        plt.title('LSTM Predictions vs Actual (PyTorch)')
        plt.xlabel('Time Step')
        plt.ylabel('Target Value')
        plt.legend()
        plt.savefig(output_dir / "lstm_predictions.png")
        plt.close()

        print(f"💾 Saved plots to {output_dir}")

    except Exception as e:
        print(f"❌ Error during visualization: {e}")

# 1. Train the LSTM Model and Make Predictions
def train_and_predict_lstm(X_train, y_train, X_test, y_test, model_output_dir):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = X_train.shape[2]

        # Initialize the model, loss function, and optimizer
        model = LSTMModel(input_size).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Create DataLoader for training and testing
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                     torch.tensor(y_test, dtype=torch.float32))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

        # Make predictions on the test set
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            predictions = model(X_test_tensor).cpu().numpy()

        # Return predictions
        return predictions

    except Exception as e:
        print(f"❌ Error during LSTM training and prediction: {e}")
        return None


# 2. Save the Merged Data with LSTM Predictions
def save_with_lstm_predictions(df, predictions, scaler_y, output_dir):
    try:
        # Inverse transform the LSTM predictions to original scale
        predictions_inv = scaler_y.inverse_transform(predictions)

        # Add the predictions as a new column to the DataFrame
        df['lstm_predictions'] = np.nan  # Initialize the new column with NaN
        df.iloc[len(df) - len(predictions):, df.columns.get_loc('lstm_predictions')] = predictions_inv.flatten()

        # Save the DataFrame with predictions to a new CSV file
        output_file = output_dir / "merged_data_with_lstm_predictions.csv"
        df.to_csv(output_file)
        print(f"✅ Merged data with LSTM predictions saved to {output_file}")

    except Exception as e:
        print(f"❌ Error saving merged data with LSTM predictions: {e}")


# =============================================
# 7. Main Execution
# =============================================
if __name__ == "__main__":
    print("🚀 Starting PyTorch LSTM training pipeline...")

    # Load and merge data
    df = load_and_merge_data()
    if df is None:
        sys.exit(1)

    # Prepare data sequences for LSTM
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = prepare_sequences(df)
    if X_train is None:
        sys.exit(1)

    # Train LSTM and get predictions
    predictions = train_and_predict_lstm(X_train, y_train, X_test, y_test, LSTM_DIR)
    if predictions is None:
        sys.exit(1)

    # Save merged data with LSTM predictions
    save_with_lstm_predictions(df, predictions, scaler_y, LSTM_DIR)

    print("\n✅ LSTM pipeline with PyTorch completed successfully!")

