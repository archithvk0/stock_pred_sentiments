# Google Colab Setup Guide

This guide will help you set up and use the Stock Prediction with News and Social Media Analysis project in Google Colab.

## Quick Start Options

### Option 1: GitHub (Recommended)

1. **Upload your project to GitHub:**

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git push -u origin main
   ```

2. **In Google Colab, run:**
   ```python
   !git clone https://github.com/yourusername/your-repo-name.git
   %cd your-repo-name/stock_prediction_news_social_media
   !python colab_setup.py
   ```

### Option 2: Direct Upload

1. **Zip your project folder** on your local machine
2. **In Google Colab, run:**
   ```python
   from google.colab import files
   uploaded = files.upload()
   !unzip stock_prediction_news_social_media.zip
   %cd stock_prediction_news_social_media
   !python colab_setup.py
   ```

## Step-by-Step Setup

### 1. Install Dependencies

Run the setup script to install all required packages:

```python
!python colab_setup.py
```

This will install:

- Core data science libraries (pandas, numpy, matplotlib)
- Machine learning libraries (scikit-learn, tensorflow, torch)
- Financial data libraries (yfinance, alpha-vantage)
- Sentiment analysis libraries (vaderSentiment, textblob, nltk)
- Web scraping libraries (requests, beautifulsoup4)

### 2. Configure API Keys

Set up your API keys for data collection:

```python
import os

# Alpha Vantage API key (free tier available at https://www.alphavantage.co/)
os.environ['ALPHA_VANTAGE_API_KEY'] = 'your_alpha_vantage_key_here'

# Reddit API credentials (optional)
os.environ['REDDIT_CLIENT_ID'] = 'your_reddit_client_id'
os.environ['REDDIT_CLIENT_SECRET'] = 'your_reddit_client_secret'
os.environ['REDDIT_USER_AGENT'] = 'your_user_agent'

# Twitter API credentials (optional)
os.environ['TWITTER_BEARER_TOKEN'] = 'your_twitter_bearer_token'
```

### 3. Import and Use Modules

```python
# Import the main modules
from data_collector import DataCollector
from sentiment_analyzer import SentimentAnalyzer
from ml_models import StockPredictor
from feature_engineer import FeatureEngineer
from backtester import Backtester
import config

print("All modules imported successfully!")
```

## Example Usage

### Collect Stock Data

```python
# Initialize data collector
collector = DataCollector()

# Collect stock data
symbol = 'AAPL'
stock_data = collector.get_stock_data(symbol, period='1y')
print(f"Collected {len(stock_data)} days of stock data for {symbol}")
```

### Analyze Sentiment

```python
# Initialize sentiment analyzer
analyzer = SentimentAnalyzer()

# Analyze news sentiment
news_data = collector.get_news_data(symbol, days=30)
analyzed_news = analyzer.analyze_news_articles(news_data.to_dict('records'))

# Calculate daily sentiment summary
daily_sentiment = analyzer.calculate_daily_sentiment_summary(analyzed_news, 'date')
```

### Train ML Model

```python
# Initialize feature engineer and predictor
feature_engineer = FeatureEngineer()
predictor = StockPredictor()

# Create features
features = feature_engineer.create_technical_indicators(stock_data)
features = feature_engineer.add_sentiment_features(features, daily_sentiment)

# Prepare data and train
X, y = feature_engineer.prepare_features_for_training(features)
X_train, X_test, y_train, y_test = predictor.split_data(X, y, test_size=0.2)
model = predictor.train_lstm_model(X_train, y_train, X_test, y_test)
```

### Run Backtest

```python
# Initialize backtester
backtester = Backtester()

# Make predictions and create signals
predictions = predictor.predict(model, X_test)
signals = [1 if pred > 0.02 else (-1 if pred < -0.02 else 0) for pred in predictions]

# Run backtest
test_prices = stock_data['Close'].iloc[-len(predictions):].values
results = backtester.run_backtest(test_prices, signals, initial_capital=10000)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
```

## Important Notes

### GPU Usage

To use GPU acceleration in Colab:

1. Go to Runtime → Change runtime type
2. Select "GPU" as Hardware accelerator
3. This will speed up LSTM model training significantly

### Memory Management

- Colab has limited memory (12GB RAM)
- For large datasets, consider using smaller time periods
- Use `del` to free up memory when done with large variables

### API Limits

- Alpha Vantage free tier: 5 API calls per minute, 500 per day
- Reddit API: Rate limited, use responsibly
- Twitter API: Requires paid access for most endpoints

### Saving Results

To save your results in Colab:

```python
# Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Save results
results_df.to_csv('/content/drive/MyDrive/stock_prediction_results.csv')
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the correct directory

   ```python
   %cd stock_prediction_news_social_media
   ```

2. **Memory errors**: Reduce dataset size or use smaller models

   ```python
   # Use smaller time period
   stock_data = collector.get_stock_data(symbol, period='6mo')
   ```

3. **API errors**: Check your API keys and limits

   ```python
   # Test API connection
   collector.test_api_connection()
   ```

4. **Model training issues**: Use GPU acceleration
   ```python
   # Check if GPU is available
   import torch
   print(f"GPU available: {torch.cuda.is_available()}")
   ```

## Support

If you encounter issues:

1. Check the main README.md for detailed documentation
2. Verify all dependencies are installed correctly
3. Ensure API keys are properly configured
4. Check Colab's runtime logs for error messages

Happy trading! 📈
