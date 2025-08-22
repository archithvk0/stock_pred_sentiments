"""
One-Click Google Colab Setup Script
Run this entire script in a single Colab cell to set up everything automatically
"""

import subprocess
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def main():
    """Main setup function"""
    print("🚀 STOCK PREDICTION PROJECT - ONE-CLICK COLAB SETUP")
    print("="*60)
    
    # Step 1: Install dependencies
    print("\n📦 Step 1: Installing dependencies...")
    install_dependencies()
    
    # Step 2: Download required data/models
    print("\n📚 Step 2: Downloading required data and models...")
    download_models()
    
    # Step 3: Test imports
    print("\n🧪 Step 3: Testing imports...")
    if not test_imports():
        print("❌ Setup failed: Could not import modules")
        return False
    
    # Step 4: Run quick demo
    print("\n🚀 Step 4: Running quick demo...")
    if not run_demo():
        print("❌ Setup failed: Demo did not complete successfully")
        return False
    
    # Step 5: Show next steps
    print("\n" + "="*60)
    print("✅ SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    show_next_steps()
    
    return True

def install_dependencies():
    """Install all required dependencies"""
    packages = [
    "praw>=7.6.0",
    "vaderSentiment>=3.3.2", 
    "textblob>=0.17.1",
    "nltk>=3.7",
    "spacy>=3.4.0",
    "transformers>=4.20.0",
    "torch>=1.12.0",
    "tensorflow>=2.10.0",
    "yfinance>=0.1.70",
    "alpha-vantage>=2.3.1",
    "requests>=2.28.0",
    "beautifulsoup4>=4.11.0",
    "plotly>=5.10.0",
    "python-dotenv>=0.19.0",
    "tqdm>=4.64.0",
    "scikit-learn>=1.1.0",
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0"
]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"✅ {package.split('>=')[0]}")
        except:
            print(f"⚠️  {package.split('>=')[0]}")

def download_models():
    """Download required NLTK data and spaCy model"""
    # Download NLTK data
    import nltk
    nltk_data = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
    
    for data in nltk_data:
        try:
            nltk.download(data, quiet=True)
            print(f"✅ NLTK {data}")
        except:
            print(f"⚠️  NLTK {data}")
    
    # Download spaCy model
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ spaCy en_core_web_sm")
    except:
        print("⚠️  spaCy model")

def test_imports():
    """Test if all modules can be imported"""
    try:
        from data_collector import DataCollector
        from sentiment_analyzer import SentimentAnalyzer
        from ml_models import StockPredictor
        from feature_engineer import FeatureEngineer
        from backtester import Backtester
        import config
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def run_demo():
    """Run a quick demo to test the system"""
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
        symbol = 'AAPL'
        stock_data = collector.get_stock_data(symbol, period='6mo')
        print(f"✅ Collected {len(stock_data)} days of {symbol} data")
        
        # Test sentiment analysis
        test_text = "AAPL stock is looking bullish today! The earnings report was amazing."
        sentiment_result = analyzer.analyze_text_sentiment(test_text)
        print(f"✅ Sentiment analysis: VADER score {sentiment_result['vader_scores']['compound']:.3f}")
        
        # Test feature engineering
        features = feature_engineer.create_technical_indicators(stock_data)
        print(f"✅ Created {len(features.columns)} technical indicators")
        
        # Test model training (small sample)
        X, y = feature_engineer.prepare_features_for_training(features)
        if len(X) > 50:  # Use very small sample for demo
            X = X[-50:]
            y = y[-50:]
        
        X_train, X_test, y_train, y_test = predictor.split_data(X, y, test_size=0.3)
        model = predictor.train_lstm_model(X_train, y_train, X_test, y_test, epochs=3)
        print("✅ Model training completed")
        
        # Test backtesting
        predictions = predictor.predict(model, X_test)
        signals = [1 if pred > 0.01 else (-1 if pred < -0.01 else 0) for pred in predictions]
        test_prices = stock_data['Close'].iloc[-len(predictions):].values
        results = backtester.run_backtest(test_prices, signals, initial_capital=10000)
        print(f"✅ Backtest: {results['total_return']:.2%} return")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

def show_next_steps():
    """Show next steps for the user"""
    print("\n📋 NEXT STEPS:")
    print("1. Configure your API keys:")
    print("   os.environ['ALPHA_VANTAGE_API_KEY'] = 'your_key_here'")
    print("2. Run the example usage:")
    print("   from data_collector import DataCollector")
    print("   collector = DataCollector()")
    print("   stock_data = collector.get_stock_data('AAPL', period='1y')")
    print("\n🔧 Available modules:")
    print("- DataCollector: Collect stock data, news, and social media")
    print("- SentimentAnalyzer: Analyze sentiment of text data")
    print("- StockPredictor: Train and use ML models")
    print("- FeatureEngineer: Create technical indicators and features")
    print("- Backtester: Test trading strategies")
    print("\n💡 Tips:")
    print("- Enable GPU runtime for faster training")
    print("- Use smaller datasets for quick testing")
    print("- Save results to Google Drive")
    print("\n🎉 You're ready to start analyzing stocks!")

if __name__ == "__main__":
    main() 
