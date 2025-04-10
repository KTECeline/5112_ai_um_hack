from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def create_sequences(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def preprocess_data(df):
    # Normalize data
    scaler = MinMaxScaler()
    df['value_scaled'] = scaler.fit_transform(df[['inflow_total']])  # Ensure column name matches

    # Create sequences
    X, y = create_sequences(df['value_scaled'].values)
    
    # Reshape X to add an extra dimension for features
    X = X.reshape(X.shape[0], X.shape[1], 1)  # (samples, time steps, features)
    
    # Reshape y to (samples, 1)
    y = y.reshape(-1, 1)  # (samples, 1)
    
    return X, y

# Example usage:
if __name__ == "__main__":
    df = pd.read_csv("btc_inflow_okx_cleaned.csv")  # or whichever file you're using
    X, y = preprocess_data(df)

    print(f"X shape: {X.shape}, y shape: {y.shape}")

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last time step's output
        return out

# Prepare the data (assuming X and y are numpy arrays)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Initialize the model
model = LSTMModel()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)


    # Backward pass
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/20], Loss: {loss.item():.4f}")
