"""
Feature Engineering Module for Stock Price Prediction
Combines sentiment data with technical indicators to create ML features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate technical indicators for stock data"""
    
    def __init__(self):
        pass
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to stock data
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with technical indicators added
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying original
        df_tech = df.copy()
        
        # Trend indicators
        df_tech = self._add_trend_indicators(df_tech)
        
        # Momentum indicators
        df_tech = self._add_momentum_indicators(df_tech)
        
        # Volatility indicators
        df_tech = self._add_volatility_indicators(df_tech)
        
        # Volume indicators
        df_tech = self._add_volume_indicators(df_tech)
        
        # Price-based features
        df_tech = self._add_price_features(df_tech)
        
        return df_tech
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based technical indicators"""
        # Simple Moving Averages
        df['sma_5'] = SMAIndicator(close=df['Close'], window=5).sma_indicator()
        df['sma_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
        df['sma_20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        
        # Exponential Moving Averages
        df['ema_12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()
        
        # MACD
        macd = MACD(close=df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based technical indicators"""
        # RSI
        df['rsi'] = RSIIndicator(close=df['Close']).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based technical indicators"""
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_percent'] = bb.bollinger_pband()
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based technical indicators"""
        # Volume Weighted Average Price
        df['vwap'] = VolumeWeightedAveragePrice(
            high=df['High'], 
            low=df['Low'], 
            close=df['Close'], 
            volume=df['Volume']
        ).volume_weighted_average_price()
        
        # Volume SMA
        df['volume_sma_5'] = df['Volume'].rolling(window=5).mean()
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Returns
        df['daily_return'] = df['Close'].pct_change()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Price changes
        df['price_change'] = df['Close'] - df['Close'].shift(1)
        df['price_change_pct'] = df['price_change'] / df['Close'].shift(1) * 100
        
        # High-Low spread
        df['hl_spread'] = df['High'] - df['Low']
        df['hl_spread_pct'] = df['hl_spread'] / df['Close'] * 100
        
        # Gap
        df['gap'] = df['Open'] - df['Close'].shift(1)
        df['gap_pct'] = df['gap'] / df['Close'].shift(1) * 100
        
        return df

class SentimentFeatures:
    """Extract features from sentiment data"""
    
    def __init__(self):
        pass
    
    def create_sentiment_features(self, sentiment_data: List[Dict], date_column: str = 'date') -> pd.DataFrame:
        """
        Create sentiment features from analyzed data
        
        Args:
            sentiment_data: List of sentiment analysis results
            date_column: Column name containing the date
        
        Returns:
            DataFrame with sentiment features
        """
        if not sentiment_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(sentiment_data)
        
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Group by date and calculate features
        daily_features = df.groupby(date_column).agg({
            # VADER features
            'vader_scores': [
                ('vader_compound_mean', lambda x: np.mean([item['compound'] for item in x])),
                ('vader_compound_std', lambda x: np.std([item['compound'] for item in x])),
                ('vader_positive_mean', lambda x: np.mean([item['pos'] for item in x])),
                ('vader_negative_mean', lambda x: np.mean([item['neg'] for item in x])),
                ('vader_neutral_mean', lambda x: np.mean([item['neu'] for item in x]))
            ],
            
            # FinBERT features
            'finbert_scores': [
                ('finbert_positive_mean', lambda x: np.mean([item['positive'] for item in x])),
                ('finbert_negative_mean', lambda x: np.mean([item['negative'] for item in x])),
                ('finbert_neutral_mean', lambda x: np.mean([item['neutral'] for item in x]))
            ],
            
            # TextBlob features
            'textblob_polarity': ['mean', 'std'],
            'textblob_subjectivity': ['mean', 'std'],
            
            # Count features
            'text': 'count'  # Number of posts/articles per day
        }).reset_index()
        
        # Flatten column names
        daily_features.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                                for col in daily_features.columns]
        
        # Add sentiment label distributions
        daily_features = self._add_sentiment_label_features(df, daily_features, date_column)
        
        # Add engagement features (for Reddit data)
        if 'score' in df.columns:
            daily_features = self._add_engagement_features(df, daily_features, date_column)
        
        return daily_features
    
    def _add_sentiment_label_features(self, df: pd.DataFrame, daily_features: pd.DataFrame, 
                                    date_column: str) -> pd.DataFrame:
        """Add features based on sentiment label distributions"""
        # VADER label distribution
        vader_labels = df.groupby(date_column)['vader_label'].value_counts().unstack(fill_value=0)
        vader_labels.columns = [f'vader_{col}' for col in vader_labels.columns]
        
        # FinBERT label distribution
        finbert_labels = df.groupby(date_column)['finbert_label'].value_counts().unstack(fill_value=0)
        finbert_labels.columns = [f'finbert_{col}' for col in finbert_labels.columns]
        
        # Merge with daily features
        daily_features = daily_features.merge(vader_labels, left_on=date_column, right_index=True, how='left')
        daily_features = daily_features.merge(finbert_labels, left_on=date_column, right_index=True, how='left')
        
        # Fill NaN values with 0
        daily_features = daily_features.fillna(0)
        
        return daily_features
    
    def _add_engagement_features(self, df: pd.DataFrame, daily_features: pd.DataFrame, 
                               date_column: str) -> pd.DataFrame:
        """Add engagement-based features for Reddit data"""
        engagement_features = df.groupby(date_column).agg({
            'score': ['mean', 'std', 'sum'],
            'upvote_ratio': 'mean',
            'num_comments': ['mean', 'sum']
        }).reset_index()
        
        # Flatten column names
        engagement_features.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                                     for col in engagement_features.columns]
        
        # Merge with daily features
        daily_features = daily_features.merge(engagement_features, on=date_column, how='left')
        
        return daily_features

class FeatureEngineer:
    """Main feature engineering orchestrator"""
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
        self.sentiment_features = SentimentFeatures()
    
    def create_features(self, stock_data: pd.DataFrame, sentiment_data: List[Dict], 
                       target_horizon: int = 1) -> pd.DataFrame:
        """
        Create comprehensive feature set for ML model
        
        Args:
            stock_data: DataFrame with OHLCV data
            sentiment_data: List of sentiment analysis results
            target_horizon: Number of days ahead to predict (default: 1)
        
        Returns:
            DataFrame with all features and target variable
        """
        if stock_data.empty:
            return pd.DataFrame()
        
        # Add technical indicators to stock data
        stock_features = self.technical_indicators.add_technical_indicators(stock_data)
        
        # Create sentiment features
        sentiment_features = self.sentiment_features.create_sentiment_features(sentiment_data)
        
        # Merge stock and sentiment features
        features_df = self._merge_features(stock_features, sentiment_features)
        
        # Create target variable
        features_df = self._create_target_variable(features_df, target_horizon)
        
        # Add lagged features
        features_df = self._add_lagged_features(features_df)
        
        # Clean up features
        features_df = self._clean_features(features_df)
        
        return features_df
    
    def _merge_features(self, stock_features: pd.DataFrame, 
                       sentiment_features: pd.DataFrame) -> pd.DataFrame:
        """Merge stock and sentiment features"""
        # Convert date columns to datetime
        stock_features['date'] = pd.to_datetime(stock_features.index.date)
        sentiment_features['date'] = pd.to_datetime(sentiment_features['date'])
        
        # Merge on date
        merged_df = stock_features.merge(sentiment_features, on='date', how='left')
        
        # Forward fill sentiment features (carry forward last known sentiment)
        sentiment_columns = [col for col in merged_df.columns if col not in stock_features.columns]
        merged_df[sentiment_columns] = merged_df[sentiment_columns].fillna(method='ffill')
        
        # Fill remaining NaN with 0
        merged_df = merged_df.fillna(0)
        
        return merged_df
    
    def _create_target_variable(self, df: pd.DataFrame, target_horizon: int) -> pd.DataFrame:
        """Create target variable for prediction"""
        # Create future return
        df['future_return'] = df['Close'].shift(-target_horizon) / df['Close'] - 1
        
        # Create binary target (1 for positive return, 0 for negative)
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # Create multi-class target (strong up, up, down, strong down)
        df['target_multi'] = pd.cut(df['future_return'], 
                                  bins=[-np.inf, -0.02, 0, 0.02, np.inf],
                                  labels=[0, 1, 2, 3])
        
        return df
    
    def _add_lagged_features(self, df: pd.DataFrame, max_lags: int = 5) -> pd.DataFrame:
        """Add lagged versions of key features"""
        # Features to lag
        lag_features = [
            'daily_return', 'volume', 'rsi', 'macd', 'vader_compound_mean',
            'finbert_positive_mean', 'textblob_polarity_mean'
        ]
        
        for feature in lag_features:
            if feature in df.columns:
                for lag in range(1, max_lags + 1):
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for ML"""
        # Remove rows with NaN values (usually at the beginning due to lagged features)
        df = df.dropna()
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Remove date column (not needed for ML)
        if 'date' in df.columns:
            df = df.drop('date', axis=1)
        
        return df
    
    def get_feature_importance_ranking(self, df: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Get feature importance ranking using correlation with target
        
        Args:
            df: DataFrame with features and target
        
        Returns:
            List of (feature_name, correlation) tuples sorted by importance
        """
        if 'target' not in df.columns:
            return []
        
        # Calculate correlation with target
        correlations = df.corr()['target'].abs().sort_values(ascending=False)
        
        # Remove target itself and future_return
        correlations = correlations.drop(['target', 'future_return', 'target_multi'], errors='ignore')
        
        return list(correlations.items())
    
    def select_top_features(self, df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """
        Select top N most important features
        
        Args:
            df: DataFrame with all features
            top_n: Number of top features to select
        
        Returns:
            DataFrame with selected features plus target
        """
        feature_importance = self.get_feature_importance_ranking(df)
        
        if not feature_importance:
            return df
        
        # Get top N feature names
        top_features = [feature for feature, _ in feature_importance[:top_n]]
        
        # Add target columns
        target_columns = [col for col in df.columns if 'target' in col or 'future_return' in col]
        selected_columns = top_features + target_columns
        
        return df[selected_columns]

# Example usage
if __name__ == "__main__":
    # Test the feature engineer
    from data_collector import DataCollector
    from sentiment_analyzer import SentimentAnalyzer
    
    # Collect sample data
    collector = DataCollector()
    analyzer = SentimentAnalyzer()
    
    # Get stock data
    stock_data = collector.stock_collector.get_stock_data("AAPL", "2024-01-01", "2024-01-31")
    
    # Get and analyze sentiment data
    news_data = collector.news_collector.get_yahoo_finance_news("AAPL")
    analyzed_news = analyzer.analyze_news_articles(news_data)
    
    # Create features
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_features(stock_data, analyzed_news)
    
    print(f"Created {len(features_df)} feature rows")
    print(f"Number of features: {len(features_df.columns)}")
    print(f"Target distribution: {features_df['target'].value_counts()}")
    
    # Get feature importance
    importance = feature_engineer.get_feature_importance_ranking(features_df)
    print("\nTop 10 most important features:")
    for feature, corr in importance[:10]:
        print(f"{feature}: {corr:.4f}") 