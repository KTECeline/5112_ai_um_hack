import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import sqlite3
import schedule
import time
import os

def fetch_and_analyze():
    # Load and clean data
    df = pd.read_csv('btc_inflow_okx_cleaned.csv')
    df.columns = df.columns.str.strip()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['value'] = df['inflow_total']  # Standardize 'value' as main column
    df['hour'] = df['datetime'].dt.hour  # Extract hour for hourly analysis

    # Print execution time
    print("Pipeline executed at:", time.ctime())

    # Summary statistics
    print("\n[Key Statistics]")
    print(df['value'].describe())

    # Feature engineering
    df['value_ma_3h'] = df['value'].rolling(window=3).mean()  # 3-hour moving average
    df['value_pct_change'] = df['value'].pct_change() * 100  # Percent change

    # Display engineered features
    print("\n[Engineered Features Sample]")
    print(df[['datetime', 'value', 'value_ma_3h', 'value_pct_change']].head())

    # Detect anomalies (2 std devs above mean)
    threshold = df['value'].mean() + 2 * df['value'].std()
    anomalies = df[df['value'] > threshold]
    print("\n[Anomalies Detected (Above 2 Std Devs)]")
    print(anomalies[['datetime', 'value']])

    # Correlation analysis
    correlation = df[['value', 'value_ma_3h']].corr()
    print("\n[Correlation Matrix]")
    print(correlation)

    # Visualizations
    # 1. Hourly BTC inflow line plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='datetime', y='value')
    plt.title("Hourly BTC Inflow on OKX")
    plt.xlabel("Time")
    plt.ylabel("Inflow (BTC)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Distribution of BTC inflow values
    plt.figure(figsize=(10, 6))
    sns.histplot(df['value'], kde=True)
    plt.title("Distribution of BTC Inflow Values")
    plt.xlabel("BTC Inflow")
    plt.tight_layout()
    plt.show()

    # 3. Hourly average inflow bar plot
    plt.figure(figsize=(10, 4))
    df.groupby('hour')['value'].mean().plot(kind='bar', color='skyblue')
    plt.title("Average BTC Inflow by Hour")
    plt.xlabel("Hour (0-23)")
    plt.ylabel("Average Inflow (BTC)")
    plt.tight_layout()
    plt.savefig("avg_inflow_by_hour.png")
    plt.close()

    # 4. Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # Generate PDF report
    with PdfPages("btc_inflow_report.pdf") as pdf:
        # Line plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='datetime', y='value')
        plt.title("BTC Inflow Over Time")
        plt.xlabel("Datetime")
        plt.ylabel("BTC Inflow")
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Anomaly scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(anomalies['datetime'], anomalies['value'], color='red')
        plt.title("Anomalies Detected (Above 2 Std Devs)")
        plt.xlabel("Datetime")
        plt.ylabel("BTC Inflow")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print("✅ Report saved to btc_inflow_report.pdf")

    # Save to SQLite database with schema check
    conn = sqlite3.connect('crypto_data.db')
    cursor = conn.cursor()

    # Check if table exists and its columns
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='btc_inflow'")
    table_exists = cursor.fetchone()

    if table_exists:
        # Get existing columns
        cursor.execute("PRAGMA table_info(btc_inflow)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        
        # Add missing columns if they don't exist
        if 'value_ma_3h' not in existing_columns:
            cursor.execute("ALTER TABLE btc_inflow ADD COLUMN value_ma_3h REAL")
        if 'value_pct_change' not in existing_columns:
            cursor.execute("ALTER TABLE btc_inflow ADD COLUMN value_pct_change REAL")
        if 'hour' not in existing_columns:
            cursor.execute("ALTER TABLE btc_inflow ADD COLUMN hour INTEGER")

    # Save DataFrame to SQLite
    df.to_sql('btc_inflow', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()
    print("✅ Data inserted into crypto_data.db")

# Run once immediately
fetch_and_analyze()

# Schedule hourly runs
schedule.every().hour.do(fetch_and_analyze)

# Keep script running
while True:
    schedule.run_pending()
    time.sleep(1)