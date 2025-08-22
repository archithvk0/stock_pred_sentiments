"""
Streamlit Application for Stock Price Prediction
Interactive dashboard for sentiment analysis and stock prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Import our modules
from data_collector import DataCollector, StockDataCollector, NewsCollector, RedditCollector
from sentiment_analyzer import SentimentAnalyzer
from feature_engineer import FeatureEngineer
from ml_models import ModelManager
from backtester import Backtester, StrategyBacktester, BacktestVisualizer

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Stock Price Prediction with Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff7f0e;
    }
</style>
""", unsafe_allow_html=True)

class StockPredictionApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.feature_engineer = FeatureEngineer()
        self.model_manager = ModelManager()
        self.backtester = Backtester()
        self.strategy_backtester = StrategyBacktester()
        self.visualizer = BacktestVisualizer()
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'predictions_made' not in st.session_state:
            st.session_state.predictions_made = False
    
    def run(self):
        """Run the main application"""
        # Header
        st.markdown('<h1 class="main-header">📈 Stock Price Prediction with Sentiment Analysis</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self.sidebar()
        
        # Main content
        if st.session_state.data_loaded:
            self.main_content()
        else:
            self.welcome_page()
    
    def sidebar(self):
        """Sidebar configuration"""
        st.sidebar.title("⚙️ Configuration")
        
        # Stock selection
        st.sidebar.subheader("Stock Selection")
        ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()
        
        # Date range
        st.sidebar.subheader("Date Range")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        start_date_input = st.sidebar.date_input(
            "Start Date",
            value=start_date,
            max_value=end_date
        )
        
        end_date_input = st.sidebar.date_input(
            "End Date",
            value=end_date,
            max_value=end_date
        )
        
        # Data collection options
        st.sidebar.subheader("Data Sources")
        collect_news = st.sidebar.checkbox("Collect News Articles", value=True)
        collect_reddit = st.sidebar.checkbox("Collect Reddit Posts", value=True)
        
        # Model options
        st.sidebar.subheader("Model Options")
        use_ensemble = st.sidebar.checkbox("Use Ensemble Model", value=True)
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.9,
            value=0.7,
            step=0.05
        )
        
        # Store in session state
        st.session_state.ticker = ticker
        st.session_state.start_date = start_date_input
        st.session_state.end_date = end_date_input
        st.session_state.collect_news = collect_news
        st.session_state.collect_reddit = collect_reddit
        st.session_state.use_ensemble = use_ensemble
        st.session_state.confidence_threshold = confidence_threshold
        
        # Action buttons
        st.sidebar.subheader("Actions")
        if st.sidebar.button("🔄 Load Data", type="primary"):
            self.load_data()
        
        if st.sidebar.button("🤖 Train Models"):
            self.train_models()
        
        if st.sidebar.button("🔮 Make Predictions"):
            self.make_predictions()
    
    def welcome_page(self):
        """Welcome page when no data is loaded"""
        st.markdown("""
        ## Welcome to Stock Price Prediction with Sentiment Analysis! 📊
        
        This application combines traditional technical analysis with sentiment analysis from news articles and social media to predict stock price movements.
        
        ### Features:
        - 📈 **Real-time stock data** from Yahoo Finance
        - 📰 **News sentiment analysis** from financial news sources
        - 💬 **Social media sentiment** from Reddit discussions
        - 🤖 **Machine learning models** including ensemble methods
        - 📊 **Interactive visualizations** and backtesting results
        - 💰 **Trading strategy simulation** with performance metrics
        
        ### How to get started:
        1. **Configure** your settings in the sidebar
        2. **Load data** for your chosen stock
        3. **Train models** on historical data
        4. **Make predictions** and view results
        5. **Analyze performance** with backtesting
        
        ### Supported Stocks:
        Any stock available on Yahoo Finance (e.g., AAPL, TSLA, GOOGL, MSFT, etc.)
        
        ---
        
        **Note**: This is for educational purposes only. Always do your own research before making investment decisions.
        """)
        
        # Quick start example
        st.subheader("🚀 Quick Start Example")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Try AAPL"):
                st.session_state.ticker = "AAPL"
                st.session_state.start_date = datetime.now() - timedelta(days=90)
                st.session_state.end_date = datetime.now()
                self.load_data()
        
        with col2:
            if st.button("Try TSLA"):
                st.session_state.ticker = "TSLA"
                st.session_state.start_date = datetime.now() - timedelta(days=90)
                st.session_state.end_date = datetime.now()
                self.load_data()
        
        with col3:
            if st.button("Try GOOGL"):
                st.session_state.ticker = "GOOGL"
                st.session_state.start_date = datetime.now() - timedelta(days=90)
                st.session_state.end_date = datetime.now()
                self.load_data()
    
    def load_data(self):
        """Load data for the selected stock"""
        with st.spinner(f"Loading data for {st.session_state.ticker}..."):
            try:
                # Load stock data
                stock_data = self.data_collector.stock_collector.get_stock_data(
                    st.session_state.ticker,
                    st.session_state.start_date.strftime('%Y-%m-%d'),
                    st.session_state.end_date.strftime('%Y-%m-%d')
                )
                
                if stock_data.empty:
                    st.error(f"No stock data found for {st.session_state.ticker}")
                    return
                
                st.session_state.stock_data = stock_data
                
                # Load sentiment data
                sentiment_data = []
                
                if st.session_state.collect_news:
                    news_data = self.data_collector.news_collector.get_yahoo_finance_news(
                        st.session_state.ticker
                    )
                    analyzed_news = self.sentiment_analyzer.analyze_news_articles(news_data)
                    sentiment_data.extend(analyzed_news)
                
                if st.session_state.collect_reddit:
                    reddit_posts = self.data_collector.reddit_collector.get_reddit_posts(
                        st.session_state.ticker
                    )
                    analyzed_posts = self.sentiment_analyzer.analyze_reddit_posts(reddit_posts)
                    sentiment_data.extend(analyzed_posts)
                
                st.session_state.sentiment_data = sentiment_data
                st.session_state.data_loaded = True
                
                st.success(f"✅ Data loaded successfully!")
                st.info(f"📊 Stock data: {len(stock_data)} days")
                st.info(f"📰 Sentiment data: {len(sentiment_data)} articles/posts")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
    
    def train_models(self):
        """Train machine learning models"""
        if not st.session_state.data_loaded:
            st.error("Please load data first!")
            return
        
        with st.spinner("Training models..."):
            try:
                # Create features
                features_df = self.feature_engineer.create_features(
                    st.session_state.stock_data,
                    st.session_state.sentiment_data
                )
                
                if features_df.empty:
                    st.error("No features could be created. Check your data.")
                    return
                
                st.session_state.features_df = features_df
                
                # Prepare data for training
                X = features_df.drop(['target', 'future_return', 'target_multi'], axis=1, errors='ignore')
                y = features_df['target']
                
                # Split data
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                # Train models
                results = self.model_manager.train_all_models(
                    X_train.values, y_train.values,
                    X_test.values, y_test.values
                )
                
                st.session_state.model_results = results
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.models_trained = True
                
                st.success("✅ Models trained successfully!")
                
                # Show model performance
                self.show_model_performance(results, y_test)
                
            except Exception as e:
                st.error(f"Error training models: {str(e)}")
    
    def show_model_performance(self, results, y_test):
        """Show model performance metrics"""
        st.subheader("📊 Model Performance")
        
        # Create performance comparison
        performance_data = []
        for model_name, result in results.items():
            if 'val_metrics' in result:
                metrics = result['val_metrics']
                performance_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score'],
                    'ROC-AUC': metrics.get('roc_auc', 0)
                })
        
        if performance_data:
            perf_df = pd.DataFrame(performance_data)
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(perf_df, use_container_width=True)
            
            with col2:
                # Plot accuracy comparison
                fig = px.bar(perf_df, x='Model', y='Accuracy', 
                           title='Model Accuracy Comparison',
                           color='Accuracy', color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
    
    def make_predictions(self):
        """Make predictions using trained models"""
        if not st.session_state.models_trained:
            st.error("Please train models first!")
            return
        
        with st.spinner("Making predictions..."):
            try:
                # Get latest features
                latest_features = st.session_state.features_df.iloc[-1:].drop(
                    ['target', 'future_return', 'target_multi'], axis=1, errors='ignore'
                )
                
                # Make ensemble prediction
                if st.session_state.use_ensemble:
                    pred, proba = self.model_manager.ensemble.predict_ensemble(latest_features.values)
                    prediction = pred[0]
                    confidence = proba[0][1] if proba[0][1] > 0.5 else proba[0][0]
                else:
                    # Use best performing model
                    best_model_name = max(st.session_state.model_results.keys(), 
                                        key=lambda x: st.session_state.model_results[x].get('val_metrics', {}).get('accuracy', 0))
                    best_model = st.session_state.model_results[best_model_name]['model']
                    best_scaler = st.session_state.model_results[best_model_name].get('scaler')
                    
                    if best_scaler:
                        latest_features_scaled = best_scaler.transform(latest_features.values)
                    else:
                        latest_features_scaled = latest_features.values
                    
                    proba = best_model.predict_proba(latest_features_scaled)
                    prediction = best_model.predict(latest_features_scaled)[0]
                    confidence = proba[0][1] if proba[0][1] > 0.5 else proba[0][0]
                
                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                st.session_state.predictions_made = True
                
                st.success("✅ Predictions made successfully!")
                
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
    
    def main_content(self):
        """Main content when data is loaded"""
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Stock Data", "📊 Sentiment Analysis", "🤖 Predictions", "📈 Backtesting", "📋 Summary"
        ])
        
        with tab1:
            self.stock_data_tab()
        
        with tab2:
            self.sentiment_analysis_tab()
        
        with tab3:
            self.predictions_tab()
        
        with tab4:
            self.backtesting_tab()
        
        with tab5:
            self.summary_tab()
    
    def stock_data_tab(self):
        """Stock data visualization tab"""
        st.subheader(f"📈 {st.session_state.ticker} Stock Data")
        
        # Stock price chart
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=st.session_state.stock_data.index,
            open=st.session_state.stock_data['Open'],
            high=st.session_state.stock_data['High'],
            low=st.session_state.stock_data['Low'],
            close=st.session_state.stock_data['Close'],
            name='OHLC'
        ))
        
        fig.update_layout(
            title=f'{st.session_state.ticker} Stock Price',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        latest_price = st.session_state.stock_data['Close'].iloc[-1]
        price_change = st.session_state.stock_data['Close'].iloc[-1] - st.session_state.stock_data['Close'].iloc[-2]
        price_change_pct = (price_change / st.session_state.stock_data['Close'].iloc[-2]) * 100
        
        with col1:
            st.metric("Current Price", f"${latest_price:.2f}")
        
        with col2:
            st.metric("Daily Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
        
        with col3:
            st.metric("Volume", f"{st.session_state.stock_data['Volume'].iloc[-1]:,}")
        
        with col4:
            st.metric("Data Points", len(st.session_state.stock_data))
        
        # Data table
        st.subheader("Raw Data")
        st.dataframe(st.session_state.stock_data.tail(10), use_container_width=True)
    
    def sentiment_analysis_tab(self):
        """Sentiment analysis visualization tab"""
        st.subheader("📊 Sentiment Analysis")
        
        if not st.session_state.sentiment_data:
            st.info("No sentiment data available. Enable news/Reddit collection in sidebar.")
            return
        
        # Sentiment distribution
        sentiment_df = pd.DataFrame(st.session_state.sentiment_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # VADER sentiment distribution
            if 'vader_label' in sentiment_df.columns:
                vader_counts = sentiment_df['vader_label'].value_counts()
                fig = px.pie(values=vader_counts.values, names=vader_counts.index, 
                           title='VADER Sentiment Distribution')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # FinBERT sentiment distribution
            if 'finbert_label' in sentiment_df.columns:
                finbert_counts = sentiment_df['finbert_label'].value_counts()
                fig = px.pie(values=finbert_counts.values, names=finbert_counts.index,
                           title='FinBERT Sentiment Distribution')
                st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment over time
        if 'created_utc' in sentiment_df.columns:
            sentiment_df['date'] = pd.to_datetime(sentiment_df['created_utc'], unit='s')
            daily_sentiment = sentiment_df.groupby(sentiment_df['date'].dt.date).agg({
                'vader_scores': lambda x: np.mean([item['compound'] for item in x]),
                'finbert_scores': lambda x: np.mean([item['positive'] for item in x])
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily_sentiment['date'], y=daily_sentiment['vader_scores'],
                                   mode='lines+markers', name='VADER Compound'))
            fig.add_trace(go.Scatter(x=daily_sentiment['date'], y=daily_sentiment['finbert_scores'],
                                   mode='lines+markers', name='FinBERT Positive'))
            
            fig.update_layout(title='Sentiment Over Time', xaxis_title='Date', yaxis_title='Sentiment Score')
            st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment data table
        st.subheader("Sentiment Data")
        display_cols = ['title', 'vader_label', 'finbert_label', 'textblob_polarity']
        display_cols = [col for col in display_cols if col in sentiment_df.columns]
        st.dataframe(sentiment_df[display_cols].head(10), use_container_width=True)
    
    def predictions_tab(self):
        """Predictions tab"""
        st.subheader("🤖 Stock Price Predictions")
        
        if not st.session_state.predictions_made:
            st.info("Make predictions using the sidebar button.")
            return
        
        # Prediction display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            if st.session_state.prediction == 1:
                st.markdown("### 📈 **PREDICTION: UP**")
                st.markdown("The model predicts the stock price will **increase**")
            else:
                st.markdown("### 📉 **PREDICTION: DOWN**")
                st.markdown("The model predicts the stock price will **decrease**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Confidence", f"{st.session_state.confidence:.2%}")
            st.metric("Model Type", "Ensemble" if st.session_state.use_ensemble else "Single Model")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature importance
        if hasattr(st.session_state, 'features_df'):
            feature_importance = self.feature_engineer.get_feature_importance_ranking(
                st.session_state.features_df
            )
            
            if feature_importance:
                st.subheader("🔍 Feature Importance")
                
                importance_df = pd.DataFrame(feature_importance[:10], 
                                           columns=['Feature', 'Importance'])
                
                fig = px.bar(importance_df, x='Importance', y='Feature', 
                           orientation='h', title='Top 10 Most Important Features')
                st.plotly_chart(fig, use_container_width=True)
    
    def backtesting_tab(self):
        """Backtesting tab"""
        st.subheader("📈 Backtesting Results")
        
        if not st.session_state.models_trained:
            st.info("Train models first to see backtesting results.")
            return
        
        # Run backtesting
        if st.button("Run Backtesting Analysis"):
            with st.spinner("Running backtesting..."):
                try:
                    # Get predictions for backtesting
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test
                    
                    if st.session_state.use_ensemble:
                        predictions, _ = self.model_manager.ensemble.predict_ensemble(X_test.values)
                    else:
                        # Use best model
                        best_model_name = max(st.session_state.model_results.keys(), 
                                            key=lambda x: st.session_state.model_results[x].get('val_metrics', {}).get('accuracy', 0))
                        best_model = st.session_state.model_results[best_model_name]['model']
                        best_scaler = st.session_state.model_results[best_model_name].get('scaler')
                        
                        if best_scaler:
                            X_test_scaled = best_scaler.transform(X_test.values)
                        else:
                            X_test_scaled = X_test.values
                        
                        predictions = best_model.predict(X_test_scaled)
                    
                    # Get stock data for backtesting period
                    test_data = st.session_state.stock_data.iloc[-len(X_test):]
                    
                    # Run backtesting
                    results = self.strategy_backtester.compare_strategies(test_data, predictions)
                    
                    # Display results
                    self.display_backtesting_results(results)
                    
                except Exception as e:
                    st.error(f"Error in backtesting: {str(e)}")
    
    def display_backtesting_results(self, results):
        """Display backtesting results"""
        # Performance comparison
        st.subheader("📊 Performance Comparison")
        
        performance_data = []
        for strategy_name, result in results.items():
            performance_data.append({
                'Strategy': strategy_name.replace('_', ' ').title(),
                'Total Return': result.total_return,
                'Sharpe Ratio': result.sharpe_ratio,
                'Max Drawdown': result.max_drawdown,
                'Win Rate': result.win_rate,
                'Total Trades': result.total_trades
            })
        
        perf_df = pd.DataFrame(performance_data)
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(perf_df, use_container_width=True)
        
        with col2:
            # Plot returns comparison
            fig = go.Figure()
            for strategy_name, result in results.items():
                if not result.equity_curve.empty:
                    fig.add_trace(go.Scatter(
                        x=result.equity_curve.index,
                        y=result.equity_curve.values,
                        name=strategy_name.replace('_', ' ').title(),
                        mode='lines'
                    ))
            
            fig.update_layout(
                title='Equity Curves Comparison',
                xaxis_title='Date',
                yaxis_title='Portfolio Value',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def summary_tab(self):
        """Summary tab"""
        st.subheader("📋 Project Summary")
        
        # Project overview
        st.markdown("""
        ## Project Overview
        
        This stock prediction system combines traditional financial analysis with modern sentiment analysis techniques to predict stock price movements.
        
        ### Key Components:
        
        **1. Data Collection** 📊
        - Real-time stock data from Yahoo Finance
        - News articles from financial sources
        - Social media posts from Reddit
        
        **2. Sentiment Analysis** 📰
        - VADER sentiment analysis for social media text
        - FinBERT for finance-specific sentiment
        - TextBlob for additional sentiment insights
        
        **3. Feature Engineering** 🔧
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Sentiment features (daily averages, distributions)
        - Price-based features (returns, volatility)
        
        **4. Machine Learning Models** 🤖
        - Baseline models (Logistic Regression, Random Forest, SVM)
        - Gradient boosting (XGBoost, LightGBM)
        - Deep learning (LSTM, Dense Neural Networks)
        - Ensemble methods for improved performance
        
        **5. Backtesting** 📈
        - Strategy performance evaluation
        - Risk metrics calculation
        - Comparison with benchmark strategies
        
        ### Current Status:
        """)
        
        # Status indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.data_loaded:
                st.success("✅ Data Loaded")
            else:
                st.error("❌ No Data")
        
        with col2:
            if st.session_state.models_trained:
                st.success("✅ Models Trained")
            else:
                st.error("❌ Models Not Trained")
        
        with col3:
            if st.session_state.predictions_made:
                st.success("✅ Predictions Made")
            else:
                st.error("❌ No Predictions")
        
        # Technical details
        if st.session_state.data_loaded:
            st.subheader("📊 Data Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Stock Data:**")
                st.write(f"- Ticker: {st.session_state.ticker}")
                st.write(f"- Date Range: {st.session_state.start_date} to {st.session_state.end_date}")
                st.write(f"- Data Points: {len(st.session_state.stock_data)}")
                st.write(f"- Current Price: ${st.session_state.stock_data['Close'].iloc[-1]:.2f}")
            
            with col2:
                st.write("**Sentiment Data:**")
                st.write(f"- Articles/Posts: {len(st.session_state.sentiment_data)}")
                if st.session_state.sentiment_data:
                    sentiment_df = pd.DataFrame(st.session_state.sentiment_data)
                    if 'vader_label' in sentiment_df.columns:
                        vader_dist = sentiment_df['vader_label'].value_counts()
                        st.write(f"- VADER Positive: {vader_dist.get('positive', 0)}")
                        st.write(f"- VADER Negative: {vader_dist.get('negative', 0)}")
                        st.write(f"- VADER Neutral: {vader_dist.get('neutral', 0)}")
        
        # Disclaimer
        st.markdown("""
        ---
        
        ### ⚠️ Disclaimer
        
        This application is for **educational and research purposes only**. 
        
        - Past performance does not guarantee future results
        - Always do your own research before making investment decisions
        - Consider consulting with a financial advisor
        - The models may not account for all market factors and conditions
        """)

def main():
    """Main function to run the Streamlit app"""
    app = StockPredictionApp()
    app.run()

if __name__ == "__main__":
    main() 