"""
Example Script: Stock Price Prediction with Sentiment Analysis
Demonstrates the complete workflow from data collection to predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_collector import DataCollector, StockDataCollector, NewsCollector
from sentiment_analyzer import SentimentAnalyzer
from feature_engineer import FeatureEngineer
from ml_models import ModelManager
from backtester import Backtester, StrategyBacktester
from config import Config

def main():
    """Main example function"""
    print("🚀 Stock Price Prediction Example")
    print("=" * 50)
    
    # Configuration
    ticker = "AAPL"
    start_date = "2024-01-01"
    end_date = "2024-01-31"
    
    print(f"Analyzing {ticker} from {start_date} to {end_date}")
    print()
    
    # Step 1: Data Collection
    print("📊 Step 1: Collecting Data...")
    
    # Initialize collectors
    data_collector = DataCollector()
    sentiment_analyzer = SentimentAnalyzer()
    
    # Collect stock data
    print(f"  - Collecting stock data for {ticker}...")
    stock_data = data_collector.stock_collector.get_stock_data(ticker, start_date, end_date)
    
    if stock_data.empty:
        print(f"❌ No stock data found for {ticker}")
        return
    
    print(f"  ✅ Collected {len(stock_data)} days of stock data")
    
    # Collect news data
    print("  - Collecting news articles...")
    news_data = data_collector.news_collector.get_yahoo_finance_news(ticker)
    analyzed_news = sentiment_analyzer.analyze_news_articles(news_data)
    print(f"  ✅ Collected and analyzed {len(analyzed_news)} news articles")
    
    # Step 2: Feature Engineering
    print("\n🔧 Step 2: Creating Features...")
    
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_features(stock_data, analyzed_news)
    
    if features_df.empty:
        print("❌ No features could be created")
        return
    
    print(f"  ✅ Created {len(features_df)} feature rows with {len(features_df.columns)} features")
    
    # Show feature importance
    feature_importance = feature_engineer.get_feature_importance_ranking(features_df)
    print("\n  📈 Top 5 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance[:5]):
        print(f"    {i+1}. {feature}: {importance:.4f}")
    
    # Step 3: Model Training
    print("\n🤖 Step 3: Training Models...")
    
    # Prepare data
    X = features_df.drop(['target', 'future_return', 'target_multi'], axis=1, errors='ignore')
    y = features_df['target']
    
    # Split data
    split_idx = int(len(X) * Config.TRAIN_TEST_SPLIT)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"  - Training set: {len(X_train)} samples")
    print(f"  - Test set: {len(X_test)} samples")
    
    # Train models
    model_manager = ModelManager()
    results = model_manager.train_all_models(
        X_train.values, y_train.values,
        X_test.values, y_test.values
    )
    
    print("  ✅ Models trained successfully")
    
    # Show model performance
    print("\n  📊 Model Performance (Test Set):")
    for model_name, result in results.items():
        if 'val_metrics' in result:
            metrics = result['val_metrics']
            print(f"    {model_name}: Accuracy={metrics['accuracy']:.3f}, "
                  f"F1={metrics['f1_score']:.3f}, ROC-AUC={metrics.get('roc_auc', 0):.3f}")
    
    # Step 4: Make Predictions
    print("\n🔮 Step 4: Making Predictions...")
    
    # Get latest features
    latest_features = features_df.iloc[-1:].drop(
        ['target', 'future_return', 'target_multi'], axis=1, errors='ignore'
    )
    
    # Make ensemble prediction
    pred, proba = model_manager.ensemble.predict_ensemble(latest_features.values)
    prediction = pred[0]
    confidence = proba[0][1] if proba[0][1] > 0.5 else proba[0][0]
    
    print(f"  📈 Prediction: {'UP' if prediction == 1 else 'DOWN'}")
    print(f"  🎯 Confidence: {confidence:.2%}")
    
    # Step 5: Backtesting
    print("\n📈 Step 5: Running Backtesting...")
    
    # Get predictions for backtesting
    predictions, _ = model_manager.ensemble.predict_ensemble(X_test.values)
    
    # Get stock data for backtesting period
    test_data = stock_data.iloc[-len(X_test):]
    
    # Run backtesting
    strategy_backtester = StrategyBacktester()
    backtest_results = strategy_backtester.compare_strategies(test_data, predictions)
    
    # Display results
    print("\n  📊 Backtesting Results:")
    for strategy_name, result in backtest_results.items():
        print(f"    {strategy_name}:")
        print(f"      - Total Return: {result.total_return:.2%}")
        print(f"      - Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"      - Max Drawdown: {result.max_drawdown:.2%}")
        print(f"      - Win Rate: {result.win_rate:.2%}")
        print(f"      - Total Trades: {result.total_trades}")
    
    # Step 6: Summary
    print("\n📋 Summary")
    print("=" * 50)
    print(f"✅ Successfully analyzed {ticker}")
    print(f"📊 Data: {len(stock_data)} days of stock data, {len(analyzed_news)} news articles")
    print(f"🔧 Features: {len(features_df.columns)} features created")
    print(f"🤖 Models: {len(results)} models trained")
    print(f"📈 Prediction: {'UP' if prediction == 1 else 'DOWN'} with {confidence:.2%} confidence")
    
    # Find best performing strategy
    best_strategy = max(backtest_results.items(), key=lambda x: x[1].total_return)
    print(f"🏆 Best Strategy: {best_strategy[0]} ({best_strategy[1].total_return:.2%} return)")
    
    print("\n🎉 Example completed successfully!")
    print("\n💡 Next Steps:")
    print("  - Try different tickers (TSLA, GOOGL, MSFT)")
    print("  - Adjust date ranges for different market conditions")
    print("  - Run the Streamlit app: streamlit run streamlit_app.py")
    print("  - Experiment with different model parameters")

def quick_demo():
    """Quick demonstration with minimal data"""
    print("⚡ Quick Demo: Stock Sentiment Analysis")
    print("=" * 40)
    
    # Test sentiment analysis
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "AAPL stock is looking bullish today! Great earnings report.",
        "TSLA shares are tanking after the latest recall announcement.",
        "The market is showing mixed signals for tech stocks."
    ]
    
    print("📰 Sentiment Analysis Examples:")
    for i, text in enumerate(test_texts, 1):
        result = analyzer.analyze_text_sentiment(text)
        print(f"\n{i}. Text: {text}")
        print(f"   VADER: {result['vader_label']} ({result['vader_scores']['compound']:.3f})")
        print(f"   FinBERT: {result['finbert_label']}")
        print(f"   TextBlob: {result['textblob_polarity']:.3f}")
    
    # Test stock data collection
    print("\n📈 Stock Data Collection:")
    collector = StockDataCollector()
    data = collector.get_stock_data("AAPL", "2024-01-15", "2024-01-20")
    
    if not data.empty:
        print(f"✅ Collected {len(data)} days of AAPL data")
        print(f"   Latest price: ${data['Close'].iloc[-1]:.2f}")
        print(f"   Price change: {((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100:.2f}%")
    else:
        print("❌ Failed to collect stock data")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_demo()
    else:
        main() 