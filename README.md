Pitching slides : https://www.canva.com/design/DAGkDxQOxO0/4ACdBhT_8DOPM8jGur5IMA/edit?utm_content=DAGkDxQOxO0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

Group 5112 (127): Flux Backtester
Overview
This project is a comprehensive pipeline for analyzing cryptocurrency markets, focusing on Bitcoin (BTC) and Ethereum (ETH). It collects data from multiple sources, engineers features, trains machine learning models (HMM and LSTM), and evaluates trading strategies through backtesting and forward testing. The pipeline is designed for researchers, traders, and developers interested in data-driven crypto market insights.

Pipeline Stages
Data Collection: Fetches price, volume, and on-chain data from APIs (Binance, CoinGecko, Etherscan, CryptoQuant, Glassnode, CoinGlass).
Feature Engineering: Creates technical indicators (RSI, volatility), volume metrics, and on-chain features.
ML Data Preparation: Prepares data for machine learning with a binary target (price increase > 0.1%).
HMM Training: Identifies market regimes (bull, bear, neutral) using a Hidden Markov Model.
LSTM Training: Predicts price movements with an LSTM model, incorporating regimes.
Backtesting: Evaluates trading strategies with metrics like Sharpe ratio and drawdown.
Forward Testing: Tests strategies on recent data (Jan–Apr 2025).
Prerequisites: Python: 3.8+
Dependencies: pip install pandas numpy requests torch sklearn hmmlearn matplotlib seaborn plantuml
API Keys:
CryptoQuant, Etherscan (set as environment variables: CRYPTOQUANT_API_KEY, ETHERSCAN_API_KEY)

Installation
1. git clone https://github.com/KTECeline/5112_ai_um_hack.git
2. pip install -r requirements.txt
3. export CRYPTOQUANT_API_KEY="your_cryptoquant_key"
export ETHERSCAN_API_KEY="your_etherscan_key"

Usage
Run the Pipeline: Execute scripts in order (ensure the data, feature, ml_data, hmm_results, and lstm_results directories are writable):

python DataCollector.py
python FeatureEngineering.py
python MLDataPrep.py
python HMMTraining.py
python LSTMTraining.py
python Backtester.py
python TestForwardtest.py

