"""
Sentiment Analysis Module for Stock Price Prediction
Handles sentiment analysis of news articles and social media posts
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
import spacy

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Handles text preprocessing for sentiment analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Add financial-specific stop words
        self.stop_words.update([
            'stock', 'stocks', 'market', 'markets', 'trading', 'trade', 'investor', 'investors',
            'company', 'companies', 'share', 'shares', 'price', 'prices', 'earnings', 'revenue',
            'quarter', 'quarterly', 'annual', 'year', 'years', 'million', 'billion', 'trillion'
        ])
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text for sentiment analysis
        
        Args:
            text: Raw text input
        
        Returns:
            Cleaned text
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stop words from text
        
        Args:
            text: Input text
        
        Returns:
            Text with stop words removed
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers from text (e.g., $AAPL, AAPL)
        
        Args:
            text: Input text
        
        Returns:
            List of found tickers
        """
        # Pattern for $TICKER format
        dollar_pattern = r'\$([A-Z]{1,5})'
        # Pattern for standalone tickers (3-5 capital letters)
        standalone_pattern = r'\b([A-Z]{3,5})\b'
        
        tickers = []
        tickers.extend(re.findall(dollar_pattern, text.upper()))
        tickers.extend(re.findall(standalone_pattern, text.upper()))
        
        return list(set(tickers))

class VADERSentimentAnalyzer:
    """VADER sentiment analysis for social media text"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        # Add financial-specific words to VADER lexicon
        self._add_financial_lexicon()
    
    def _add_financial_lexicon(self):
        """Add financial-specific words to VADER lexicon"""
        financial_words = {
            'bullish': 2.0,
            'bearish': -2.0,
            'rally': 1.5,
            'crash': -2.5,
            'surge': 2.0,
            'plunge': -2.5,
            'soar': 2.0,
            'tank': -2.5,
            'moon': 2.0,
            'dump': -2.0,
            'pump': 1.5,
            'short': -1.0,
            'long': 1.0,
            'buy': 1.5,
            'sell': -1.5,
            'hold': 0.0,
            'profit': 1.5,
            'loss': -1.5,
            'gain': 1.5,
            'drop': -1.5,
            'rise': 1.5,
            'fall': -1.5,
            'beat': 1.5,
            'miss': -1.5,
            'upgrade': 1.5,
            'downgrade': -1.5,
            'positive': 1.5,
            'negative': -1.5,
            'strong': 1.0,
            'weak': -1.0,
            'growth': 1.5,
            'decline': -1.5
        }
        
        self.analyzer.lexicon.update(financial_words)
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with sentiment scores
        """
        if not text or pd.isna(text):
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 0.0}
        
        scores = self.analyzer.polarity_scores(text)
        return scores
    
    def get_sentiment_label(self, compound_score: float) -> str:
        """
        Convert compound score to sentiment label
        
        Args:
            compound_score: VADER compound score
        
        Returns:
            Sentiment label
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

class FinBERTSentimentAnalyzer:
    """FinBERT sentiment analysis for financial text"""
    
    def __init__(self):
        try:
            # Load FinBERT model
            self.model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
            self.available = True
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.warning(f"FinBERT model not available: {str(e)}")
            self.available = False
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with sentiment scores
        """
        if not self.available or not text or pd.isna(text):
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        try:
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            result = self.pipeline(text)[0]
            
            # Convert FinBERT output to standard format
            scores = {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 0.0
            }
            
            if result['label'] == 'positive':
                scores['positive'] = result['score']
            elif result['label'] == 'negative':
                scores['negative'] = result['score']
            else:
                scores['neutral'] = result['score']
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in FinBERT analysis: {str(e)}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
    
    def get_sentiment_label(self, scores: Dict[str, float]) -> str:
        """
        Get sentiment label from FinBERT scores
        
        Args:
            scores: FinBERT sentiment scores
        
        Returns:
            Sentiment label
        """
        max_score = max(scores.values())
        for label, score in scores.items():
            if score == max_score:
                return label
        return 'neutral'

class SentimentAnalyzer:
    """Main sentiment analysis orchestrator"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vader_analyzer = VADERSentimentAnalyzer()
        self.finbert_analyzer = FinBERTSentimentAnalyzer()
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, any]:
        """
        Perform comprehensive sentiment analysis on text
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with all sentiment analysis results
        """
        if not text or pd.isna(text):
            return self._empty_sentiment_result()
        
        # Preprocess text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Extract tickers
        tickers = self.preprocessor.extract_tickers(text)
        
        # VADER analysis
        vader_scores = self.vader_analyzer.analyze_sentiment(cleaned_text)
        vader_label = self.vader_analyzer.get_sentiment_label(vader_scores['compound'])
        
        # FinBERT analysis
        finbert_scores = self.finbert_analyzer.analyze_sentiment(cleaned_text)
        finbert_label = self.finbert_analyzer.get_sentiment_label(finbert_scores)
        
        # TextBlob analysis (additional baseline)
        blob = TextBlob(cleaned_text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'tickers': tickers,
            'vader_scores': vader_scores,
            'vader_label': vader_label,
            'finbert_scores': finbert_scores,
            'finbert_label': finbert_label,
            'textblob_polarity': textblob_polarity,
            'textblob_subjectivity': textblob_subjectivity,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _empty_sentiment_result(self) -> Dict[str, any]:
        """Return empty sentiment result structure"""
        return {
            'text': '',
            'cleaned_text': '',
            'tickers': [],
            'vader_scores': {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 0.0},
            'vader_label': 'neutral',
            'finbert_scores': {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0},
            'finbert_label': 'neutral',
            'textblob_polarity': 0.0,
            'textblob_subjectivity': 0.0,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def analyze_news_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for a list of news articles
        
        Args:
            articles: List of news article dictionaries
        
        Returns:
            List of articles with sentiment analysis added
        """
        analyzed_articles = []
        
        for article in articles:
            # Combine title and description for analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            
            sentiment_result = self.analyze_text_sentiment(text)
            
            # Merge sentiment results with article data
            analyzed_article = {**article, **sentiment_result}
            analyzed_articles.append(analyzed_article)
        
        return analyzed_articles
    
    def analyze_reddit_posts(self, posts: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for a list of Reddit posts
        
        Args:
            posts: List of Reddit post dictionaries
        
        Returns:
            List of posts with sentiment analysis added
        """
        analyzed_posts = []
        
        for post in posts:
            # Combine title and text for analysis
            text = f"{post.get('title', '')} {post.get('text', '')}"
            
            sentiment_result = self.analyze_text_sentiment(text)
            
            # Merge sentiment results with post data
            analyzed_post = {**post, **sentiment_result}
            analyzed_posts.append(analyzed_post)
        
        return analyzed_posts
    
    def analyze_reddit_comments(self, comments: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for a list of Reddit comments
        
        Args:
            comments: List of Reddit comment dictionaries
        
        Returns:
            List of comments with sentiment analysis added
        """
        analyzed_comments = []
        
        for comment in comments:
            text = comment.get('text', '')
            sentiment_result = self.analyze_text_sentiment(text)
            
            # Merge sentiment results with comment data
            analyzed_comment = {**comment, **sentiment_result}
            analyzed_comments.append(analyzed_comment)
        
        return analyzed_comments
    
    def calculate_daily_sentiment_summary(self, analyzed_data: List[Dict], date_column: str = 'date') -> pd.DataFrame:
        """
        Calculate daily sentiment summary statistics
        
        Args:
            analyzed_data: List of analyzed data points
            date_column: Column name containing the date
        
        Returns:
            DataFrame with daily sentiment summary
        """
        if not analyzed_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(analyzed_data)
        
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Group by date and calculate summary statistics
        daily_summary = df.groupby(date_column).agg({
            'vader_scores': lambda x: np.mean([item['compound'] for item in x]),
            'finbert_scores': lambda x: np.mean([item['positive'] for item in x]),
            'textblob_polarity': 'mean',
            'textblob_subjectivity': 'mean'
        }).reset_index()
        
        # Rename columns for clarity
        daily_summary.columns = [
            date_column,
            'avg_vader_compound',
            'avg_finbert_positive',
            'avg_textblob_polarity',
            'avg_textblob_subjectivity'
        ]
        
        return daily_summary

# Example usage
if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = SentimentAnalyzer()
    
    # Test text
    test_text = "AAPL stock is looking bullish today! The earnings report was amazing and the stock price surged 5%."
    
    result = analyzer.analyze_text_sentiment(test_text)
    
    print("Sentiment Analysis Results:")
    print(f"Text: {result['text']}")
    print(f"Tickers found: {result['tickers']}")
    print(f"VADER compound score: {result['vader_scores']['compound']:.3f}")
    print(f"VADER label: {result['vader_label']}")
    print(f"FinBERT positive score: {result['finbert_scores']['positive']:.3f}")
    print(f"FinBERT label: {result['finbert_label']}")
    print(f"TextBlob polarity: {result['textblob_polarity']:.3f}") 