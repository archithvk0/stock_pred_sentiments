# 🚀 Google Colab Setup Guide for Stock Prediction Project

This guide will walk you through setting up and running the Stock Prediction with News and Social Media Analysis project on Google Colab.

## 📋 Prerequisites

Before starting, make sure you have:

- A Google account
- Access to Google Colab (free)
- API keys for data sources (optional but recommended)

## 🎯 Quick Start (3 Steps)

### Step 1: Upload Your Project

**Option A: GitHub (Recommended)**

```python
# In a Colab cell, run:
!git clone https://github.com/yourusername/your-repo-name.git
%cd your-repo-name/stock_prediction_news_social_media
```

**Option B: Direct Upload**

```python
# Upload your project zip file
from google.colab import files
uploaded = files.upload()
!unzip stock_prediction_news_social_media.zip
%cd stock_prediction_news_social_media
```

### Step 2: Run Auto-Setup

```python
# Run the quick start script
!python colab_quick_start.py
```

### Step 3: Configure API Keys

```python
import os

# Alpha Vantage API key (free at https://www.alphavantage.co/)
os.environ['ALPHA_VANTAGE_API_KEY'] = 'your_key_here'

# Optional: Reddit API
os.environ['REDDIT_CLIENT_ID'] = 'your_reddit_client_id'
os.environ['REDDIT_CLIENT_SECRET'] = 'your_reddit_client_secret'
os.environ['REDDIT_USER_AGENT'] = 'your_user_agent'
```

## 🔧 Detailed Setup Process

### 1. Enable GPU (Recommended)

1. Go to **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Click **Save**

### 2. Install Dependencies

The `colab_quick_start.py` script will automatically install:

- Core data science libraries (pandas, numpy, matplotlib)
- Machine learning libraries (scikit-learn, tensorflow, torch)
- Financial data libraries (yfinance, alpha-vantage)
- Sentiment analysis libraries (vaderSentiment, textblob, nltk)
- Web scraping libraries (requests, beautifulsoup4)

### 3. Download Required Models

The setup script will download:

- NLTK data (punkt, stopwords, etc.)
- spaCy English model
- FinBERT model (for financial sentiment analysis)

## 📊 Example Usage

### Basic Stock Analysis

```python
# Import modules
from data_collector import DataCollector
from sentiment_analyzer import SentimentAnalyzer
from feature_engineer import FeatureEngineer
from ml_models import StockPredictor
from backtester import Backtester

# Initialize components
collector = DataCollector()
analyzer = SentimentAnalyzer()
feature_engineer = FeatureEngineer()
predictor = StockPredictor()
backtester = Backtester()

# Collect stock data
symbol = 'AAPL'
stock_data = collector.get_stock_data(symbol, period='1y')
print(f"Collected {len(stock_data)} days of data")
```

### Sentiment Analysis

```python
# Analyze news sentiment
news_data = collector.get_news_data(symbol, days=30)
analyzed_news = analyzer.analyze_news_articles(news_data.to_dict('records'))

# Calculate daily sentiment summary
daily_sentiment = analyzer.calculate_daily_sentiment_summary(analyzed_news, 'date')
print(f"Daily sentiment calculated for {len(daily_sentiment)} days")
```

### Machine Learning Pipeline

```python
# Create features
features = feature_engineer.create_technical_indicators(stock_data)
features = feature_engineer.add_sentiment_features(features, daily_sentiment)

# Prepare data for training
X, y = feature_engineer.prepare_features_for_training(features)
X_train, X_test, y_train, y_test = predictor.split_data(X, y, test_size=0.2)

# Train model
model = predictor.train_lstm_model(X_train, y_train, X_test, y_test)
```

### Backtesting

```python
# Make predictions and create signals
predictions = predictor.predict(model, X_test)
signals = [1 if pred > 0.02 else (-1 if pred < -0.02 else 0) for pred in predictions]

# Run backtest
test_prices = stock_data['Close'].iloc[-len(predictions):].values
results = backtester.run_backtest(test_prices, signals, initial_capital=10000)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
```

## 🎨 Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Stock price and predictions
axes[0, 0].plot(test_prices, label='Actual Price')
axes[0, 0].scatter(range(len(predictions)), test_prices, c=predictions, cmap='RdYlGn')
axes[0, 0].set_title('Stock Price vs Predictions')

# Portfolio value
axes[0, 1].plot(results['portfolio_values'], label='Portfolio Value')
axes[0, 1].set_title('Portfolio Value Over Time')

# Daily returns distribution
axes[1, 0].hist(results['daily_returns'], bins=30, alpha=0.7)
axes[1, 0].set_title('Distribution of Daily Returns')

# Sentiment over time
if not daily_sentiment.empty:
    axes[1, 1].plot(daily_sentiment['avg_vader_compound'], label='VADER Sentiment')
    axes[1, 1].set_title('Sentiment Over Time')

plt.tight_layout()
plt.show()
```

## 💾 Saving Results

### Save to Google Drive

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create results directory
import os
results_dir = '/content/drive/MyDrive/stock_prediction_results'
os.makedirs(results_dir, exist_ok=True)

# Save results
import pandas as pd
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save stock data
stock_data.to_csv(f'{results_dir}/{symbol}_stock_data_{timestamp}.csv')

# Save sentiment data
daily_sentiment.to_csv(f'{results_dir}/{symbol}_sentiment_{timestamp}.csv')

# Save backtest results
results_df = pd.DataFrame({
    'date': stock_data.index[-len(predictions):],
    'price': test_prices,
    'prediction': predictions,
    'signal': signals,
    'portfolio_value': results['portfolio_values']
})
results_df.to_csv(f'{results_dir}/{symbol}_backtest_results_{timestamp}.csv')

print(f"Results saved to {results_dir}")
```

## 🔑 API Keys Setup

### Alpha Vantage (Free)

1. Go to [Alpha Vantage](https://www.alphavantage.co/)
2. Sign up for a free API key
3. Use the key in your code:

```python
os.environ['ALPHA_VANTAGE_API_KEY'] = 'your_key_here'
```

### Reddit API (Optional)

1. Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Create a new app
3. Use the credentials:

```python
os.environ['REDDIT_CLIENT_ID'] = 'your_client_id'
os.environ['REDDIT_CLIENT_SECRET'] = 'your_client_secret'
os.environ['REDDIT_USER_AGENT'] = 'your_user_agent'
```

## ⚠️ Important Notes

### Memory Management

- Colab has 12GB RAM limit
- For large datasets, use smaller time periods
- Use `del` to free memory:

```python
del large_variable
```

### Runtime Limits

- Colab sessions timeout after 12 hours
- Save your work frequently
- Use Google Drive for persistent storage

### API Limits

- Alpha Vantage: 5 calls/minute, 500/day (free tier)
- Reddit: Rate limited
- Twitter: Requires paid access

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**

```python
# Make sure you're in the correct directory
%cd stock_prediction_news_social_media
!ls  # Should show your project files
```

2. **Memory Errors**

```python
# Use smaller datasets
stock_data = collector.get_stock_data(symbol, period='6mo')  # Instead of '1y'
```

3. **GPU Issues**

```python
# Check GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
```

4. **API Errors**

```python
# Test API connection
collector.test_api_connection()
```

### Performance Tips

1. **Use GPU Runtime** for faster model training
2. **Reduce dataset size** for quick testing
3. **Use smaller models** initially
4. **Save intermediate results** to avoid recomputation

## 📚 Available Modules

### DataCollector

- `get_stock_data()`: Collect stock price data
- `get_news_data()`: Collect news articles
- `get_reddit_posts()`: Collect Reddit posts
- `get_reddit_comments()`: Collect Reddit comments

### SentimentAnalyzer

- `analyze_text_sentiment()`: Analyze single text
- `analyze_news_articles()`: Analyze news articles
- `analyze_reddit_posts()`: Analyze Reddit posts
- `calculate_daily_sentiment_summary()`: Daily sentiment aggregation

### FeatureEngineer

- `create_technical_indicators()`: Technical analysis indicators
- `add_sentiment_features()`: Add sentiment features
- `prepare_features_for_training()`: Prepare ML features

### StockPredictor

- `train_lstm_model()`: Train LSTM model
- `train_random_forest()`: Train Random Forest
- `predict()`: Make predictions
- `evaluate_model()`: Model evaluation

### Backtester

- `run_backtest()`: Run trading strategy backtest
- `calculate_metrics()`: Calculate performance metrics
- `plot_results()`: Visualize results

## 🎯 Next Steps

1. **Customize the analysis** for your specific needs
2. **Experiment with different models** and parameters
3. **Add more data sources** (Twitter, other news APIs)
4. **Implement your own trading strategies**
5. **Deploy to production** if needed

## 📞 Support

If you encounter issues:

1. Check the main README.md for detailed documentation
2. Verify all dependencies are installed correctly
3. Ensure API keys are properly configured
4. Check Colab's runtime logs for error messages

Happy trading! 📈
