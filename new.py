"""
Stock Price Prediction with Social Media & News Sentiment Analysis
Main entry point for the application

This system combines:
- Stock price data from Yahoo Finance
- News sentiment analysis from financial sources
- Social media sentiment from Reddit
- Machine learning models for price prediction
- Backtesting and performance evaluation
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from data_collector import DataCollector
from sentiment_analyzer import SentimentAnalyzer
from feature_engineer import FeatureEngineer
from ml_models import ModelManager
from backtester import Backtester, StrategyBacktester, BacktestVisualizer

def main():
    """Main function to run the stock prediction system"""
    parser = argparse.ArgumentParser(
        description="Stock Price Prediction with Sentiment Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python new.py --ticker AAPL --start-date 2024-01-01 --end-date 2024-01-31
  python new.py --ticker TSLA --quick-test
  python new.py --streamlit
        """
    )
    
    parser.add_argument(
        '--ticker', 
        type=str, 
        default='AAPL',
        help='Stock ticker symbol (default: AAPL)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=(datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
        help='Start date for data collection (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date for data collection (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run a quick test with limited data'
    )
    
    parser.add_argument(
        '--streamlit',
        action='store_true',
        help='Launch Streamlit web application'
    )
    
    parser.add_argument(
        '--collect-news',
        action='store_true',
        default=True,
        help='Collect news articles'
    )
    
    parser.add_argument(
        '--collect-reddit',
        action='store_true',
        default=True,
        help='Collect Reddit posts'
    )
    
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save trained models to disk'
    )
    
    parser.add_argument(
        '--load-model',
        type=str,
        help='Load pre-trained model from file'
    )
    
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run backtesting analysis'
    )
    
    args = parser.parse_args()
    
    print("🚀 Stock Price Prediction with Sentiment Analysis")
    print("=" * 60)
    print(f"Ticker: {args.ticker}")
    print(f"Date Range: {args.start_date} to {args.end_date}")
    print(f"Quick Test: {args.quick_test}")
    print(f"Streamlit Mode: {args.streamlit}")
    print("=" * 60)
    
    if args.streamlit:
        launch_streamlit()
    else:
        run_prediction_pipeline(args)

def launch_streamlit():
    """Launch the Streamlit web application"""
    try:
        import subprocess
        import sys
        
        print("🌐 Launching Streamlit web application...")
        print("The app will open in your default web browser.")
        print("Press Ctrl+C to stop the server.")
        
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except ImportError:
        print("❌ Streamlit not installed. Install with: pip install streamlit")
    except KeyboardInterrupt:
        print("\n👋 Streamlit server stopped.")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {str(e)}")

def run_prediction_pipeline(args):
    """Run the complete prediction pipeline"""
    try:
        # Initialize components
        print("📊 Initializing components...")
        data_collector = DataCollector()
        sentiment_analyzer = SentimentAnalyzer()
        feature_engineer = FeatureEngineer()
        model_manager = ModelManager()
        
        # Data Collection
        print(f"\n📈 Collecting data for {args.ticker}...")
        
        # Collect stock data
        stock_data = data_collector.stock_collector.get_stock_data(
            args.ticker, args.start_date, args.end_date
        )
        
        if stock_data.empty:
            print(f"❌ No stock data found for {args.ticker}")
            return
        
        print(f"✅ Collected {len(stock_data)} days of stock data")
        
        # Collect sentiment data
        sentiment_data = []
        
        if args.collect_news:
            print("📰 Collecting news articles...")
            news_data = data_collector.news_collector.get_yahoo_finance_news(args.ticker)
            analyzed_news = sentiment_analyzer.analyze_news_articles(news_data)
            sentiment_data.extend(analyzed_news)
            print(f"✅ Collected {len(analyzed_news)} news articles")
        
        if args.collect_reddit:
            print("💬 Collecting Reddit posts...")
            reddit_posts = data_collector.reddit_collector.get_reddit_posts(args.ticker)
            analyzed_posts = sentiment_analyzer.analyze_reddit_posts(reddit_posts)
            sentiment_data.extend(analyzed_posts)
            print(f"✅ Collected {len(analyzed_posts)} Reddit posts")
        
        # Feature Engineering
        print("\n🔧 Creating features...")
        features_df = feature_engineer.create_features(stock_data, sentiment_data)
        
        if features_df.empty:
            print("❌ No features could be created")
            return
        
        print(f"✅ Created {len(features_df)} feature rows with {len(features_df.columns)} features")
        
        # Model Training
        if args.load_model:
            print(f"\n🤖 Loading pre-trained model from {args.load_model}...")
            model_manager.load_models(args.load_model)
            print("✅ Model loaded successfully")
        else:
            print("\n🤖 Training models...")
            
            # Prepare data
            X = features_df.drop(['target', 'future_return', 'target_multi'], axis=1, errors='ignore')
            y = features_df['target']
            
            # Split data
            split_idx = int(len(X) * Config.TRAIN_TEST_SPLIT)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train models
            results = model_manager.train_all_models(
                X_train.values, y_train.values,
                X_test.values, y_test.values
            )
            
            print("✅ Models trained successfully")
            
            # Show model performance
            print("\n📊 Model Performance:")
            for model_name, result in results.items():
                if 'val_metrics' in result:
                    metrics = result['val_metrics']
                    print(f"  {model_name}: Accuracy={metrics['accuracy']:.3f}, "
                          f"F1={metrics['f1_score']:.3f}, ROC-AUC={metrics.get('roc_auc', 0):.3f}")
            
            # Save model if requested
            if args.save_model:
                model_path = f"{Config.MODELS_DIR}/stock_prediction_model_{args.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                model_manager.save_models(model_path)
                print(f"💾 Model saved to {model_path}")
        
        # Make Predictions
        print("\n🔮 Making predictions...")
        
        # Get latest features
        latest_features = features_df.iloc[-1:].drop(
            ['target', 'future_return', 'target_multi'], axis=1, errors='ignore'
        )
        
        # Make ensemble prediction
        pred, proba = model_manager.ensemble.predict_ensemble(latest_features.values)
        prediction = pred[0]
        confidence = proba[0][1] if proba[0][1] > 0.5 else proba[0][0]
        
        print(f"📈 Prediction: {'UP' if prediction == 1 else 'DOWN'}")
        print(f"🎯 Confidence: {confidence:.2%}")
        
        # Backtesting
        if args.backtest:
            print("\n📈 Running backtesting analysis...")
            
            # Get predictions for backtesting
            X_test = features_df.iloc[split_idx:].drop(
                ['target', 'future_return', 'target_multi'], axis=1, errors='ignore'
            )
            y_test = features_df.iloc[split_idx:]['target']
            
            predictions, _ = model_manager.ensemble.predict_ensemble(X_test.values)
            
            # Get stock data for backtesting period
            test_data = stock_data.iloc[-len(X_test):]
            
            # Run backtesting
            strategy_backtester = StrategyBacktester()
            results = strategy_backtester.compare_strategies(test_data, predictions)
            
            # Display results
            print("\n📊 Backtesting Results:")
            for strategy_name, result in results.items():
                print(f"  {strategy_name}: Return={result.total_return:.2%}, "
                      f"Sharpe={result.sharpe_ratio:.2f}, MaxDD={result.max_drawdown:.2%}")
        
        print("\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

def quick_test():
    """Run a quick test with sample data"""
    print("🧪 Running quick test...")
    
    # Test with a small dataset
    args = argparse.Namespace(
        ticker='AAPL',
        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d'),
        collect_news=True,
        collect_reddit=False,  # Skip Reddit for quick test
        save_model=False,
        load_model=None,
        backtest=True
    )
    
    run_prediction_pipeline(args)

if __name__ == "__main__":
    main()