import requests
import pandas as pd
import os
import json

# ✅ Step 1: Load API key from environment variable
API_KEY = os.getenv('CRYPTOQUANT_API_KEY')
if not API_KEY:
    raise ValueError("API key not found. Set the CRYPTOQUANT_API_KEY environment variable.")

# ✅ Step 2: Set the target endpoint
url = "https://api.datasource.cybotrade.rs/cryptoquant/btc/exchange-flows/inflow"

# ✅ Step 3: Define the query parameters
params = {
    "exchange": "okx",
    "window": "hour",
    "start_time": "1740990814113",
    "limit": "1000"
}

# ✅ Step 4: Set headers with your API key
headers = {
    'X-API-Key': API_KEY
}

# ✅ Step 5: Make the GET request
response = requests.get(url, headers=headers, params=params)

# ✅ Step 6: Handle response
if response.status_code == 200:
    try:
        data = response.json()

        # Print the structure of the JSON to debug
        print("Raw JSON Response:\n", json.dumps(data, indent=2))

        # ✅ Extract and convert only the 'data' part if available
        if "data" in data and isinstance(data["data"], list):
            df = pd.DataFrame(data["data"])

            # ✅ Convert timestamp column if it exists
            if not df.empty and 'timestamp' in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            print("\nFormatted DataFrame Preview:")
            print(df.head())

            # ✅ Optional: Save to CSV
            df.to_csv("btc_inflow_okx.csv", index=False)
        else:
            print("Unexpected JSON format or missing 'data' field.")
    except Exception as e:
        print("Failed to parse JSON or process data:", str(e))
else:
    print("Error fetching data:", response.status_code, response.text)

    # ... [Previous code for fetching data remains the same until df is created] ...
# ... [Previous code for fetching data remains the same until df is created] ...

# ✅ Step 7: Data Cleaning
if not df.empty:
    # 1. Drop duplicate rows (if any)
    df = df.drop_duplicates()

    # 2. Handle missing values
    # Check for missing values in critical columns
    print("\nMissing Values Before Cleaning:")
    print(df.isnull().sum())

    # Example: Drop rows with missing 'inflow_total' (assuming this is your key metric)
    df = df.dropna(subset=['inflow_total'])  # Replace 'value' with 'inflow_total'

    # 3. Ensure numeric columns are properly typed
    for col in ['inflow_mean', 'inflow_mean_ma7', 'inflow_top10', 'inflow_total']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Filter out implausible values (e.g., negative inflows)
    if 'inflow_total' in df.columns:
        df = df[df['inflow_total'] >= 0]  # Replace 'value' with 'inflow_total'

    # 5. Sort by timestamp (if needed)
    if 'datetime' in df.columns:  # Use 'datetime' instead of 'timestamp'
        df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure it's in datetime format
        df = df.sort_values(by='datetime', ascending=True)

    # 6. Reset index after cleaning
    df = df.reset_index(drop=True)

    # 7. Add derived columns (e.g., hour, day)
    if 'datetime' in df.columns:
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour

    # 8. Verify cleaned data
    print("\nMissing Values After Cleaning:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe())

    # Save cleaned data
    df.to_csv("btc_inflow_okx_cleaned.csv", index=False)
    print("\nCleaned data saved to btc_inflow_okx_cleaned.csv")
else:
    print("No data to clean.")