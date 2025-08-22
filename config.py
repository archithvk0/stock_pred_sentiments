"""
Configuration file for Stock Price Prediction System
Contains settings, API keys, and model parameters
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the stock prediction system"""
    
    # API Keys (load from environment variables)
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'StockSentimentBot/1.0')
    
    # Data Collection Settings
    DEFAULT_DAYS_BACK = 30
    MAX_REDDIT_POSTS = 100
    MAX_REDDIT_COMMENTS = 50
    REQUEST_DELAY = 1  # seconds between requests
    
    # Stock Settings
    DEFAULT_TICKERS = ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'META', 'NFLX']
    DEFAULT_START_DATE = '2023-01-01'
    
    # Model Settings
    TRAIN_TEST_SPLIT = 0.8
    RANDOM_STATE = 42
    CONFIDENCE_THRESHOLD = 0.7
    
    # Feature Engineering Settings
    MAX_LAGS = 5
    SEQUENCE_LENGTH = 10  # for LSTM models
    
    # Technical Indicators Settings
    SMA_PERIODS = [5, 10, 20, 50]
    EMA_PERIODS = [12, 26]
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2
    
    # Backtesting Settings
    INITIAL_CAPITAL = 100000
    COMMISSION_RATE = 0.001
    POSITION_SIZE_PCT = 0.95
    
    # Streamlit Settings
    PAGE_TITLE = "Stock Price Prediction with Sentiment Analysis"
    PAGE_ICON = "📈"
    LAYOUT = "wide"
    
    # File Paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    LOGS_DIR = "logs"
    
    # Create directories if they don't exist
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    # Model Parameters
    @classmethod
    def get_model_params(cls):
        """Get model parameters"""
        return {
            'logistic_regression': {
                'max_iter': 1000,
                'random_state': cls.RANDOM_STATE
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': cls.RANDOM_STATE,
                'n_jobs': -1
            },
            'xgboost': {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': cls.RANDOM_STATE
            },
            'lightgbm': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'random_state': cls.RANDOM_STATE
            },
            'lstm': {
                'units': 50,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            'dense_nn': {
                'layers': [128, 64, 32],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            }
        }
    
    # Sentiment Analysis Settings
    @classmethod
    def get_sentiment_settings(cls):
        """Get sentiment analysis settings"""
        return {
            'vader': {
                'use_financial_lexicon': True,
                'compound_threshold': 0.05
            },
            'finbert': {
                'model_name': 'ProsusAI/finbert',
                'max_length': 512
            },
            'textblob': {
                'use_subjectivity': True
            }
        }
    
    # Data Collection Settings
    @classmethod
    def get_data_collection_settings(cls):
        """Get data collection settings"""
        return {
            'stock': {
                'default_interval': '1d',
                'auto_adjust': True,
                'prepost': False
            },
            'news': {
                'sources': ['yahoo_finance', 'google_news'],
                'max_articles_per_source': 50
            },
            'reddit': {
                'subreddits': ['wallstreetbets', 'stocks', 'investing'],
                'sort_by': 'new',
                'time_filter': 'week'
            }
        }
    
    # Feature Engineering Settings
    @classmethod
    def get_feature_settings(cls):
        """Get feature engineering settings"""
        return {
            'technical_indicators': {
                'sma_periods': cls.SMA_PERIODS,
                'ema_periods': cls.EMA_PERIODS,
                'rsi_period': cls.RSI_PERIOD,
                'macd_fast': cls.MACD_FAST,
                'macd_slow': cls.MACD_SLOW,
                'macd_signal': cls.MACD_SIGNAL,
                'bollinger_period': cls.BOLLINGER_PERIOD,
                'bollinger_std': cls.BOLLINGER_STD
            },
            'sentiment_features': {
                'aggregation_methods': ['mean', 'std', 'count'],
                'rolling_windows': [1, 3, 7]
            },
            'price_features': {
                'return_periods': [1, 3, 5, 10],
                'volatility_windows': [5, 10, 20]
            }
        }
    
    # Backtesting Settings
    @classmethod
    def get_backtesting_settings(cls):
        """Get backtesting settings"""
        return {
            'initial_capital': cls.INITIAL_CAPITAL,
            'commission_rate': cls.COMMISSION_RATE,
            'position_size_pct': cls.POSITION_SIZE_PCT,
            'strategies': ['ml_strategy', 'buy_and_hold', 'momentum', 'mean_reversion'],
            'benchmark': 'buy_and_hold'
        }
    
    # Logging Settings
    @classmethod
    def get_logging_settings(cls):
        """Get logging settings"""
        return {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': f'{cls.LOGS_DIR}/stock_prediction.log'
        }

# Create directories on import
Config.create_directories()

# Example environment variables file (.env)
ENV_EXAMPLE = """
# Reddit API Configuration
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=StockSentimentBot/1.0

# Optional: Other API keys
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key

# Database Configuration (optional)
DATABASE_URL=postgresql://user:password@localhost/stock_prediction

# Model Storage
MODEL_PATH=models/stock_prediction_model.pkl
"""

if __name__ == "__main__":
    # Print configuration summary
    print("Stock Prediction System Configuration:")
    print("=" * 50)
    print(f"Default Tickers: {Config.DEFAULT_TICKERS}")
    print(f"Initial Capital: ${Config.INITIAL_CAPITAL:,}")
    print(f"Commission Rate: {Config.COMMISSION_RATE:.3f}")
    print(f"Train/Test Split: {Config.TRAIN_TEST_SPLIT}")
    print(f"Random State: {Config.RANDOM_STATE}")
    print(f"Confidence Threshold: {Config.CONFIDENCE_THRESHOLD}")
    print("\nDirectories:")
    print(f"Data: {Config.DATA_DIR}")
    print(f"Models: {Config.MODELS_DIR}")
    print(f"Logs: {Config.LOGS_DIR}")
    
    print("\nEnvironment Variables Example:")
    print(ENV_EXAMPLE) 