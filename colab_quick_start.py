"""
Quick Start Script for Google Colab
Run this script to automatically set up and test the stock prediction project
"""

import subprocess
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def install_dependencies():
    """Install all required dependencies"""
    print("🔧 Installing dependencies...")
    
    # Core packages
    packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.1.0",
        "tensorflow>=2.10.0",
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "yfinance>=0.1.70",
        "alpha-vantage>=2.3.1",
        "vaderSentiment>=3.3.2",
        "textblob>=0.17.1",
        "nltk>=3.7",
        "spacy>=3.4.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "plotly>=5.10.0",
        "python-dotenv>=0.19.0",
        "tqdm>=4.64.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except Exception as e:
            print(f"⚠️  Warning: Could not install {package}: {e}")
    
    print("✅ Dependencies installation completed!")

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    
    import nltk
    
    nltk_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
    
    for data in nltk_data:
        try:
            nltk.download(data, quiet=True)
            print(f"✅ Downloaded {data}")
        except Exception as e:
            print(f"⚠️  Warning: Could not download {data}: {e}")
    
    print("✅ NLTK data download completed!")

def download_spacy_model():
    """Download spaCy model"""
    print("🤖 Downloading spaCy model...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model downloaded successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not download spaCy model: {e}")

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from data_collector import DataCollector
        from sentiment_analyzer import SentimentAnalyzer
        from ml_models import StockPredictor
        from feature_engineer import FeatureEngineer
        from backtester import Backtester
        import config
        
        print("✅ All modules imported successfully!")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def run_demo():
    """Run a quick demo to test the system"""
    print("🚀 Running demo...")
    
    try:
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
        
        # Test stock data collection
        print("📊 Testing stock data collection...")
        symbol = 'AAPL'
        stock_data = collector.get_stock_data(symbol, period='6mo')
        print(f"✅ Collected {len(stock_data)} days of {symbol} data")
        
        # Test sentiment analysis
        print("📰 Testing sentiment analysis...")
        test_text = "AAPL stock is looking bullish today! The earnings report was amazing."
        sentiment_result = analyzer.analyze_text_sentiment(test_text)
        print(f"✅ Sentiment analysis completed. VADER score: {sentiment_result['vader_scores']['compound']:.3f}")
        
        # Test feature engineering
        print("🔧 Testing feature engineering...")
        features = feature_engineer.create_technical_indicators(stock_data)
        print(f"✅ Created {len(features.columns)} technical indicators")
        
        # Test model training (small sample)
        print("🤖 Testing model training...")
        X, y = feature_engineer.prepare_features_for_training(features)
        if len(X) > 100:  # Use smaller sample for demo
            X = X[-100:]
            y = y[-100:]
        
        X_train, X_test, y_train, y_test = predictor.split_data(X, y, test_size=0.3)
        model = predictor.train_lstm_model(X_train, y_train, X_test, y_test, epochs=5)
        print("✅ Model training completed!")
        
        # Test backtesting
        print("📈 Testing backtesting...")
        predictions = predictor.predict(model, X_test)
        signals = [1 if pred > 0.01 else (-1 if pred < -0.01 else 0) for pred in predictions]
        test_prices = stock_data['Close'].iloc[-len(predictions):].values
        results = backtester.run_backtest(test_prices, signals, initial_capital=10000)
        print(f"✅ Backtest completed! Return: {results['total_return']:.2%}")
        
        print("\n🎉 Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

def setup_environment():
    """Complete environment setup"""
    print("="*60)
    print("🚀 STOCK PREDICTION PROJECT - GOOGLE COLAB SETUP")
    print("="*60)
    
    # Install dependencies
    install_dependencies()
    
    # Download NLTK data
    download_nltk_data()
    
    # Download spaCy model
    download_spacy_model()
    
    # Test imports
    if not test_imports():
        print("❌ Setup failed: Could not import modules")
        return False
    
    # Run demo
    if not run_demo():
        print("❌ Setup failed: Demo did not complete successfully")
        return False
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Your environment is now ready for stock prediction analysis.")
    print("\n📋 Next steps:")
    print("1. Configure your API keys in the next cell")
    print("2. Run the example usage cells")
    print("3. Customize the analysis for your needs")
    print("\n🔧 Available modules:")
    print("- DataCollector: Collect stock data, news, and social media")
    print("- SentimentAnalyzer: Analyze sentiment of text data")
    print("- StockPredictor: Train and use ML models")
    print("- FeatureEngineer: Create technical indicators and features")
    print("- Backtester: Test trading strategies")
    
    return True

if __name__ == "__main__":
    setup_environment() 