# Highlight values 2 standard deviations above mean
threshold = df['value'].mean() + 2 * df['value'].std()
anomalies = df[df['value'] > threshold]
print("Anomalies:\n", anomalies)

df.groupby('hour')['value'].mean().plot(kind='bar')

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages("btc_inflow_report.pdf") as pdf:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='timestamp', y='value')
    plt.title("BTC Inflow Trends")
    pdf.savefig()
    plt.close()

    # Add to fetch_data.py
if __name__ == "__main__":
    fetch_and_clean()  # Your existing function
    os.system("python analyze_data.py")  # Trigger analysis

    import sqlite3
conn = sqlite3.connect('crypto_data.db')
df.to_sql('btc_inflow', conn, if_exists='append')

import streamlit as st
st.title("Live BTC Inflow")
st.line_chart(df.set_index('timestamp')['value'])