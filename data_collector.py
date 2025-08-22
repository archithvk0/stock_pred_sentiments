"""
Data Collection Module for Stock Price Prediction
Handles collection of stock prices, news articles, and social media data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import praw
import time
import logging
from typing import List, Dict, Optional, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataCollector:
    """Collects stock price data using yfinance"""
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_stock_data(self, ticker: str, start_date: str, end_date: str = None) -> pd.DataFrame:
        """
        Fetch stock price data for a given ticker
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (defaults to today)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            
            # Add ticker column
            data['ticker'] = ticker
            data['date'] = data.index.date
            
            logger.info(f"Successfully collected {len(data)} days of data for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error collecting stock data for {ticker}: {str(e)}")
            return pd.DataFrame()

class NewsCollector:
    """Collects news articles from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_yahoo_finance_news(self, ticker: str, days_back: int = 30) -> List[Dict]:
        """
        Collect news from Yahoo Finance RSS feed
        
        Args:
            ticker: Stock symbol
            days_back: Number of days to look back
        
        Returns:
            List of news articles with metadata
        """
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            articles = []
            
            for item in soup.find_all('item'):
                article = {
                    'title': item.title.text if item.title else '',
                    'description': item.description.text if item.description else '',
                    'link': item.link.text if item.link else '',
                    'pubDate': item.pubDate.text if item.pubDate else '',
                    'ticker': ticker,
                    'source': 'yahoo_finance'
                }
                articles.append(article)
            
            logger.info(f"Collected {len(articles)} news articles for {ticker}")
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting Yahoo Finance news for {ticker}: {str(e)}")
            return []
    
    def get_google_news(self, ticker: str, days_back: int = 30) -> List[Dict]:
        """
        Collect news from Google News (basic scraping)
        
        Args:
            ticker: Stock symbol
            days_back: Number of days to look back
        
        Returns:
            List of news articles with metadata
        """
        try:
            query = f"{ticker} stock news"
            url = f"https://news.google.com/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Find news articles (this is a simplified version)
            for article in soup.find_all('article', limit=20):
                title_elem = article.find('h3')
                if title_elem:
                    article_data = {
                        'title': title_elem.get_text().strip(),
                        'description': '',
                        'link': '',
                        'pubDate': datetime.now().strftime('%Y-%m-%d'),
                        'ticker': ticker,
                        'source': 'google_news'
                    }
                    articles.append(article_data)
            
            logger.info(f"Collected {len(articles)} Google News articles for {ticker}")
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting Google News for {ticker}: {str(e)}")
            return []

class RedditCollector:
    """Collects Reddit posts and comments"""
    
    def __init__(self):
        # Initialize Reddit API client
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'StockSentimentBot/1.0')
        )
        
        self.subreddits = ['wallstreetbets', 'stocks', 'investing']
    
    def get_reddit_posts(self, ticker: str, days_back: int = 7, limit: int = 100) -> List[Dict]:
        """
        Collect Reddit posts mentioning a specific ticker
        
        Args:
            ticker: Stock symbol
            days_back: Number of days to look back
            limit: Maximum number of posts per subreddit
        
        Returns:
            List of Reddit posts with metadata
        """
        posts = []
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        try:
            for subreddit_name in self.subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for posts containing the ticker
                search_query = f"{ticker}"
                for post in subreddit.search(search_query, limit=limit, sort='new'):
                    post_time = datetime.fromtimestamp(post.created_utc)
                    
                    if post_time >= cutoff_time:
                        post_data = {
                            'title': post.title,
                            'text': post.selftext,
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments,
                            'created_utc': post.created_utc,
                            'subreddit': subreddit_name,
                            'ticker': ticker,
                            'url': post.url,
                            'author': str(post.author) if post.author else '[deleted]'
                        }
                        posts.append(post_data)
                
                time.sleep(1)  # Rate limiting
            
            logger.info(f"Collected {len(posts)} Reddit posts for {ticker}")
            return posts
            
        except Exception as e:
            logger.error(f"Error collecting Reddit posts for {ticker}: {str(e)}")
            return []
    
    def get_reddit_comments(self, ticker: str, days_back: int = 7, limit: int = 50) -> List[Dict]:
        """
        Collect Reddit comments mentioning a specific ticker
        
        Args:
            ticker: Stock symbol
            days_back: Number of days to look back
            limit: Maximum number of comments per subreddit
        
        Returns:
            List of Reddit comments with metadata
        """
        comments = []
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        try:
            for subreddit_name in self.subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Search for comments containing the ticker
                search_query = f"{ticker}"
                for comment in subreddit.search(search_query, limit=limit, sort='new'):
                    if hasattr(comment, 'body'):  # Check if it's a comment
                        comment_time = datetime.fromtimestamp(comment.created_utc)
                        
                        if comment_time >= cutoff_time:
                            comment_data = {
                                'text': comment.body,
                                'score': comment.score,
                                'created_utc': comment.created_utc,
                                'subreddit': subreddit_name,
                                'ticker': ticker,
                                'author': str(comment.author) if comment.author else '[deleted]'
                            }
                            comments.append(comment_data)
                
                time.sleep(1)  # Rate limiting
            
            logger.info(f"Collected {len(comments)} Reddit comments for {ticker}")
            return comments
            
        except Exception as e:
            logger.error(f"Error collecting Reddit comments for {ticker}: {str(e)}")
            return []

class DataCollector:
    """Main data collection orchestrator"""
    
    def __init__(self):
        self.stock_collector = StockDataCollector()
        self.news_collector = NewsCollector()
        self.reddit_collector = RedditCollector()
    
    def collect_all_data(self, ticker: str, start_date: str, end_date: str = None) -> Dict:
        """
        Collect all data types for a given ticker
        
        Args:
            ticker: Stock symbol
            start_date: Start date for stock data
            end_date: End date for stock data
        
        Returns:
            Dictionary containing all collected data
        """
        logger.info(f"Starting data collection for {ticker}")
        
        # Collect stock data
        stock_data = self.stock_collector.get_stock_data(ticker, start_date, end_date)
        
        # Collect news data
        news_articles = self.news_collector.get_yahoo_finance_news(ticker)
        google_news = self.news_collector.get_google_news(ticker)
        all_news = news_articles + google_news
        
        # Collect Reddit data
        reddit_posts = self.reddit_collector.get_reddit_posts(ticker)
        reddit_comments = self.reddit_collector.get_reddit_comments(ticker)
        
        return {
            'ticker': ticker,
            'stock_data': stock_data,
            'news_data': all_news,
            'reddit_posts': reddit_posts,
            'reddit_comments': reddit_comments,
            'collection_date': datetime.now().isoformat()
        }
    
    def collect_multiple_tickers(self, tickers: List[str], start_date: str, end_date: str = None) -> Dict:
        """
        Collect data for multiple tickers
        
        Args:
            tickers: List of stock symbols
            start_date: Start date for stock data
            end_date: End date for stock data
        
        Returns:
            Dictionary with data for all tickers
        """
        all_data = {}
        
        for ticker in tickers:
            logger.info(f"Collecting data for {ticker}")
            ticker_data = self.collect_all_data(ticker, start_date, end_date)
            all_data[ticker] = ticker_data
            
            # Add delay between tickers to avoid rate limiting
            time.sleep(2)
        
        return all_data

# Example usage
if __name__ == "__main__":
    # Test the data collector
    collector = DataCollector()
    
    # Test with a single ticker
    test_data = collector.collect_all_data(
        ticker="AAPL",
        start_date="2024-01-01",
        end_date="2024-01-31"
    )
    
    print(f"Collected {len(test_data['stock_data'])} stock records")
    print(f"Collected {len(test_data['news_data'])} news articles")
    print(f"Collected {len(test_data['reddit_posts'])} Reddit posts")
    print(f"Collected {len(test_data['reddit_comments'])} Reddit comments") 