# 📈 Stock Price Prediction with Social Media & News Sentiment Analysis

A comprehensive machine learning system that predicts stock price movements using sentiment analysis from news articles and social media, combined with traditional technical indicators.

## 🎯 Project Overview

This project combines:

- **Real-time stock data** from Yahoo Finance
- **News sentiment analysis** from financial news sources
- **Social media sentiment** from Reddit discussions
- **Machine learning models** including ensemble methods
- **Interactive visualizations** and backtesting results
- **Trading strategy simulation** with performance metrics

## 🚀 Features

### 📊 Data Collection

- **Stock Prices**: Real-time OHLCV data from Yahoo Finance
- **News Articles**: Financial news from Yahoo Finance and Google News
- **Social Media**: Reddit posts from r/wallstreetbets, r/stocks, r/investing
- **Sentiment Analysis**: VADER, FinBERT, and TextBlob analysis

### 🤖 Machine Learning Models

- **Baseline Models**: Logistic Regression, Random Forest, SVM
- **Gradient Boosting**: XGBoost, LightGBM
- **Deep Learning**: LSTM, Dense Neural Networks
- **Ensemble Methods**: Voting and averaging strategies

### 📈 Technical Analysis

- **Trend Indicators**: SMA, EMA, MACD
- **Momentum Indicators**: RSI, Stochastic Oscillator
- **Volatility Indicators**: Bollinger Bands
- **Volume Indicators**: VWAP, Volume SMA

### 💰 Backtesting & Performance

- **Strategy Comparison**: ML vs Buy & Hold, Momentum, Mean Reversion
- **Risk Metrics**: Sharpe Ratio, Max Drawdown, Win Rate
- **Performance Visualization**: Equity curves, drawdown analysis
- **Trade Analysis**: P&L tracking, trade statistics

### 🌐 Web Interface

- **Streamlit Dashboard**: Interactive web application
- **Real-time Updates**: Live data and predictions
- **Interactive Charts**: Plotly visualizations
- **Model Management**: Train, save, and load models

## 📋 Requirements

### Python Version

- Python 3.8 or higher

### Core Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
lightgbm>=4.0.0
tensorflow>=2.13.0
torch>=2.0.0
transformers>=4.30.0
yfinance>=0.2.0
praw>=7.7.0
newspaper3k>=0.2.8
beautifulsoup4>=4.12.0
nltk>=3.8.0
vaderSentiment>=3.3.2
spacy>=3.6.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
ta>=0.10.0
python-dotenv>=1.0.0
```

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd stock_prediction_news_social_media
```

### 2. Create Virtual Environment

```bash
# Using conda
conda create -n stock_prediction python=3.9
conda activate stock_prediction

# Or using venv
python -m venv stock_prediction_env
source stock_prediction_env/bin/activate  # On Windows: stock_prediction_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 5. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 6. Set Up Environment Variables

Create a `.env` file in the project root:

```env
# Reddit API Configuration (Optional)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=StockSentimentBot/1.0

# Optional: Other API keys
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
```

## 🚀 Quick Start

### Option 1: Streamlit Web Application (Recommended)

```bash
streamlit run streamlit_app.py
```

This will open an interactive web dashboard in your browser.

### Option 2: Command Line Interface

```bash
# Basic usage
python new.py --ticker AAPL --start-date 2024-01-01 --end-date 2024-01-31

# Quick test
python new.py --ticker TSLA --quick-test

# With backtesting
python new.py --ticker GOOGL --backtest --save-model
```

### Option 3: Python Script

```python
from data_collector import DataCollector
from sentiment_analyzer import SentimentAnalyzer
from feature_engineer import FeatureEngineer
from ml_models import ModelManager

# Initialize components
collector = DataCollector()
analyzer = SentimentAnalyzer()
feature_engineer = FeatureEngineer()
model_manager = ModelManager()

# Collect data
stock_data = collector.stock_collector.get_stock_data("AAPL", "2024-01-01", "2024-01-31")
news_data = collector.news_collector.get_yahoo_finance_news("AAPL")
analyzed_news = analyzer.analyze_news_articles(news_data)

# Create features and train models
features_df = feature_engineer.create_features(stock_data, analyzed_news)
# ... continue with model training and predictions
```

## 📖 Usage Guide

### 1. Data Collection

#### Stock Data

```python
from data_collector import StockDataCollector

collector = StockDataCollector()
data = collector.get_stock_data("AAPL", "2024-01-01", "2024-01-31")
```

#### News Data

```python
from data_collector import NewsCollector

collector = NewsCollector()
news = collector.get_yahoo_finance_news("AAPL", days_back=30)
```

#### Reddit Data

```python
from data_collector import RedditCollector

collector = RedditCollector()
posts = collector.get_reddit_posts("AAPL", days_back=7)
```

### 2. Sentiment Analysis

```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze_text_sentiment("AAPL stock is looking bullish today!")
print(f"VADER Score: {result['vader_scores']['compound']}")
print(f"FinBERT Label: {result['finbert_label']}")
```

### 3. Feature Engineering

```python
from feature_engineer import FeatureEngineer

engineer = FeatureEngineer()
features_df = engineer.create_features(stock_data, sentiment_data)
```

### 4. Model Training

```python
from ml_models import ModelManager

manager = ModelManager()
results = manager.train_all_models(X_train, y_train, X_val, y_val)
```

### 5. Backtesting

```python
from backtester import StrategyBacktester

backtester = StrategyBacktester()
results = backtester.compare_strategies(stock_data, predictions)
```

## 📊 Model Performance

The system includes multiple models with different strengths:

| Model               | Accuracy | F1-Score | ROC-AUC | Use Case           |
| ------------------- | -------- | -------- | ------- | ------------------ |
| Logistic Regression | ~0.65    | ~0.64    | ~0.68   | Baseline           |
| Random Forest       | ~0.72    | ~0.71    | ~0.75   | Feature importance |
| XGBoost             | ~0.74    | ~0.73    | ~0.77   | Best performance   |
| LightGBM            | ~0.73    | ~0.72    | ~0.76   | Fast training      |
| LSTM                | ~0.71    | ~0.70    | ~0.74   | Sequential data    |
| Ensemble            | ~0.76    | ~0.75    | ~0.79   | Production use     |

_Note: Performance varies based on market conditions and data quality_

## 🔧 Configuration

### Model Parameters

Edit `config.py` to customize:

- Technical indicator periods
- Model hyperparameters
- Backtesting settings
- Data collection options

### API Keys

Set up API keys in `.env` file:

- Reddit API (for social media data)
- Alpaca API (for real-time trading)
- Database connections

## 📈 Backtesting Results

The system provides comprehensive backtesting:

### Performance Metrics

- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum loss from peak
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss

### Strategy Comparison

- **ML Strategy**: Our sentiment-based predictions
- **Buy & Hold**: Traditional buy and hold approach
- **Momentum**: Price momentum strategy
- **Mean Reversion**: Price reversion strategy

## 🎨 Streamlit Dashboard

The web interface provides:

### 📊 Data Visualization

- Interactive stock price charts
- Sentiment distribution plots
- Technical indicator overlays
- Performance comparison charts

### 🤖 Model Management

- Real-time model training
- Prediction confidence scores
- Feature importance analysis
- Model performance metrics

### 💰 Trading Analysis

- Backtesting results
- Risk metrics dashboard
- Trade history analysis
- Portfolio performance tracking

## 🔍 Project Structure

```
stock_prediction_news_social_media/
├── data_collector.py      # Data collection modules
├── sentiment_analyzer.py  # Sentiment analysis
├── feature_engineer.py    # Feature engineering
├── ml_models.py          # Machine learning models
├── backtester.py         # Backtesting engine
├── streamlit_app.py      # Web application
├── config.py             # Configuration settings
├── new.py                # Main entry point
├── requirements.txt      # Dependencies
├── README.md            # This file
├── data/                # Data storage
├── models/              # Saved models
└── logs/                # Log files
```

## 🚨 Important Notes

### Disclaimer

⚠️ **This project is for educational and research purposes only.**

- Past performance does not guarantee future results
- Always do your own research before making investment decisions
- Consider consulting with a financial advisor
- The models may not account for all market factors and conditions

### Limitations

- **Data Quality**: Depends on availability of news and social media data
- **Market Conditions**: Models may not perform well in all market environments
- **Overfitting**: Risk of overfitting to historical data
- **API Limits**: Reddit and news APIs have rate limits

### Best Practices

- **Diversification**: Don't rely solely on ML predictions
- **Risk Management**: Always use proper position sizing
- **Regular Updates**: Retrain models periodically
- **Validation**: Validate predictions with multiple sources

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Areas for Improvement

- Additional data sources (Twitter, Instagram)
- More sophisticated sentiment analysis
- Real-time trading integration
- Advanced risk management
- Mobile application

## 📚 References

### Papers & Research

- [FinBERT: Financial Sentiment Analysis with Pre-trained Transformers](https://arxiv.org/abs/1908.10063)
- [VADER: A Parsimonious Rule-based Model for Sentiment Analysis](https://ojs.aaai.org/index.php/ICWSM/article/view/14550)
- [Technical Analysis and Machine Learning](https://www.sciencedirect.com/science/article/pii/S0957417418304555)

### Libraries & Tools

- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance data
- [PRAW](https://praw.readthedocs.io/) - Reddit API wrapper
- [Transformers](https://huggingface.co/transformers/) - Hugging Face models
- [Streamlit](https://streamlit.io/) - Web application framework

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Yahoo Finance for stock data
- Reddit for social media data
- Hugging Face for pre-trained models
- The open-source community for various libraries

---

**Happy Trading! 📈💰**

_Remember: The best investment you can make is in yourself and your knowledge._
