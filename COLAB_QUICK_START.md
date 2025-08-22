# 🚀 Quick Start: Running on Google Colab

This guide shows you the fastest ways to run the Stock Prediction project on Google Colab.

## 🎯 Option 1: One-Click Setup (Easiest)

1. **Open Google Colab** and create a new notebook
2. **Copy and paste this code** into a cell:

```python
# Download and run the one-click setup
!wget https://raw.githubusercontent.com/yourusername/your-repo/main/stock_prediction_news_social_media/colab_one_click_setup.py
exec(open('colab_one_click_setup.py').read())
```

3. **Run the cell** - it will automatically:
   - Install all dependencies
   - Download required models
   - Test the system
   - Run a quick demo

## 🎯 Option 2: Manual Setup (More Control)

### Step 1: Upload Your Project

**Option A: GitHub**

```python
!git clone https://github.com/yourusername/your-repo-name.git
%cd your-repo-name/stock_prediction_news_social_media
```

**Option B: Direct Upload**

```python
from google.colab import files
uploaded = files.upload()
!unzip stock_prediction_news_social_media.zip
%cd stock_prediction_news_social_media
```

### Step 2: Run Setup Script

```python
!python colab_quick_start.py
```

### Step 3: Configure API Keys

```python
import os
os.environ['ALPHA_VANTAGE_API_KEY'] = 'your_key_here'
```

## 🎯 Option 3: Use the Notebook Template

1. **Download the notebook template**:

```python
!wget https://raw.githubusercontent.com/yourusername/your-repo/main/stock_prediction_news_social_media/colab_notebook_template.ipynb
```

2. **Open the notebook** and follow the step-by-step instructions

## ⚡ Quick Test

After setup, test that everything works:

```python
# Import modules
from data_collector import DataCollector
from sentiment_analyzer import SentimentAnalyzer

# Test data collection
collector = DataCollector()
stock_data = collector.get_stock_data('AAPL', period='1mo')
print(f"Collected {len(stock_data)} days of data")

# Test sentiment analysis
analyzer = SentimentAnalyzer()
result = analyzer.analyze_text_sentiment("AAPL stock is looking bullish!")
print(f"Sentiment score: {result['vader_scores']['compound']:.3f}")
```

## 🔧 Enable GPU (Recommended)

For faster model training:

1. Go to **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Click **Save**

## 📊 Example Usage

```python
# Complete analysis pipeline
from data_collector import DataCollector
from sentiment_analyzer import SentimentAnalyzer
from feature_engineer import FeatureEngineer
from ml_models import StockPredictor
from backtester import Backtester

# Initialize
collector = DataCollector()
analyzer = SentimentAnalyzer()
feature_engineer = FeatureEngineer()
predictor = StockPredictor()
backtester = Backtester()

# Collect data
symbol = 'AAPL'
stock_data = collector.get_stock_data(symbol, period='1y')
news_data = collector.get_news_data(symbol, days=30)

# Analyze sentiment
analyzed_news = analyzer.analyze_news_articles(news_data.to_dict('records'))
daily_sentiment = analyzer.calculate_daily_sentiment_summary(analyzed_news, 'date')

# Create features
features = feature_engineer.create_technical_indicators(stock_data)
features = feature_engineer.add_sentiment_features(features, daily_sentiment)

# Train model
X, y = feature_engineer.prepare_features_for_training(features)
X_train, X_test, y_train, y_test = predictor.split_data(X, y, test_size=0.2)
model = predictor.train_lstm_model(X_train, y_train, X_test, y_test)

# Make predictions and backtest
predictions = predictor.predict(model, X_test)
signals = [1 if pred > 0.02 else (-1 if pred < -0.02 else 0) for pred in predictions]
test_prices = stock_data['Close'].iloc[-len(predictions):].values
results = backtester.run_backtest(test_prices, signals, initial_capital=10000)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
```

## 💾 Save Results

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Save results
import pandas as pd
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = '/content/drive/MyDrive/stock_prediction_results'

stock_data.to_csv(f'{results_dir}/{symbol}_data_{timestamp}.csv')
daily_sentiment.to_csv(f'{results_dir}/{symbol}_sentiment_{timestamp}.csv')
```

## 🐛 Troubleshooting

### Common Issues:

1. **Import errors**: Make sure you're in the correct directory
2. **Memory errors**: Use smaller datasets (`period='6mo'` instead of `'1y'`)
3. **API errors**: Check your API keys
4. **GPU issues**: Verify GPU runtime is enabled

### Performance Tips:

- Use GPU runtime for faster training
- Start with smaller datasets for testing
- Save intermediate results to avoid recomputation
- Use `del` to free memory when done with large variables

## 📚 Available Files

- `colab_one_click_setup.py` - One-click setup script
- `colab_quick_start.py` - Manual setup script
- `colab_notebook_template.ipynb` - Complete notebook template
- `COLAB_SETUP_GUIDE.md` - Detailed setup guide
- `requirements.txt` - Dependencies list

## 🎉 You're Ready!

Once setup is complete, you can:

- Collect stock data and news
- Analyze sentiment
- Train machine learning models
- Run backtests
- Visualize results

Happy trading! 📈
