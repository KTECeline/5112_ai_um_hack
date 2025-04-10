import tweepy
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Setup Twitter API (authentication)
bearer_token = os.getenv("BEARER_TOKEN")
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# Define query with crypto and market-related terms
query = (
    '(crypto OR bitcoin OR ethereum OR "digital assets" OR blockchain OR defi OR nft '
    'OR "crypto news" OR "crypto regulation" OR "crypto ETF" OR "crypto crash" OR "crypto bull" '
    'OR "SEC crypto" OR "market sentiment" OR "whale trade" OR bybit OR binance OR coinbase OR "Elon Musk" '
    'OR "Fed rate" OR "inflation data" OR "interest rates" OR "CPI report" OR "market correction" '
    'OR "bull market" OR "bear market" OR "liquidity" OR "technical analysis" OR "on-chain data") '
    '-is:retweet lang:en'
)

# Time window: Last 24 hours, with end_time 10 seconds before now
end_time = datetime.now(timezone.utc) - timedelta(seconds=10)
start_time = end_time - timedelta(hours=24)

# Fetch tweets
try:
    response = client.search_recent_tweets(
        query=query,
        max_results=50,
        tweet_fields=["created_at", "text"],
        start_time=start_time,
        end_time=end_time
    )
    tweets = response.data or []
except Exception as e:
    print("Error fetching tweets:", e)
    tweets = []

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Analyze sentiment and collect additional features
sentiment_scores = []
tweet_times = []
for tweet in tweets:
    text = tweet.text
    score = analyzer.polarity_scores(text)['compound']
    sentiment_scores.append(score)
    tweet_times.append(tweet.created_at)

# Calculate sentiment statistics
avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
sentiment_std = np.std(sentiment_scores) if sentiment_scores else 0
tweet_volume = len(tweets)

# Fetch market data (e.g., Bitcoin price) for the same period
crypto_symbol = "BTC-USD"
market_data = yf.download(crypto_symbol, start=start_time.date(), end=end_time.date() + timedelta(days=1), interval="1h")

# Extract relevant market features (ensure scalar values)
if not market_data.empty:
    price_change = (market_data['Close'].iloc[-1] - market_data['Open'].iloc[0]) / market_data['Open'].iloc[0]
    volatility = market_data['Close'].pct_change().std()
    volume = market_data['Volume'].mean()
else:
    price_change = volatility = volume = 0.0

# Create DataFrame with features and target
today = end_time.date()
data = pd.DataFrame({
    'date': [today],
    'avg_sentiment': [avg_sentiment],
    'sentiment_std': [sentiment_std],
    'tweet_volume': [tweet_volume],
    'price_change': [price_change],
    'market_volatility': [volatility],
    'trading_volume': [volume]
})

# Load existing data and append new data
csv_path = "crypto_trading_data.csv"
if os.path.exists(csv_path):
    old_data = pd.read_csv(csv_path, parse_dates=["date"])
    # Convert all numeric columns to float, replacing invalid entries with NaN
    numeric_cols = ['avg_sentiment', 'sentiment_std', 'tweet_volume', 'price_change', 'market_volatility', 'trading_volume']
    for col in numeric_cols:
        old_data[col] = pd.to_numeric(old_data[col], errors='coerce')
    df = pd.concat([old_data, data]).drop_duplicates(subset="date", keep='last')
else:
    df = data

# Save to CSV
df.to_csv(csv_path, index=False)

print("\n📊 Daily Crypto Trading Data Summary:")
print(df.tail())

# Preprocess for ML (normalize features)
scaler = MinMaxScaler()
features = ['avg_sentiment', 'sentiment_std', 'tweet_volume', 'market_volatility', 'trading_volume']
# Replace NaN with 0 for scaling (or handle differently if preferred)
df[features] = df[features].fillna(0)
df[features] = scaler.fit_transform(df[features])

# Define target (1 = price increase, 0 = price decrease), handling NaN
df['price_change'] = pd.to_numeric(df['price_change'], errors='coerce').fillna(0)
df['target'] = (df['price_change'] > 0).astype(int)

print("\n📈 Preprocessed Data for ML:")
print(df.tail())