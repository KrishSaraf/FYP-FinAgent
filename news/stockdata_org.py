import requests
import pandas as pd
import time
from datetime import datetime, timedelta

# StockData.org API configuration
API_TOKEN = "nDRGm7opKrax5WzG7HzNqaiRE06vrBVyskWeicni"  # Get from https://www.stockdata.org
BASE_URL = "https://api.stockdata.org/v1/news/all"

# Your Indian stock symbols
# indian_stocks = [
#     "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR",
#     "HDFC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "ASIANPAINT",
#     "AXISBANK", "MARUTI", "ULTRACEMCO", "TITAN", "BAJFINANCE",
#     "NESTLEIND", "ONGC", "POWERGRID", "NTPC", "ITC", "BAJAJFINSV",
#     "SUNPHARMA", "TECHM", "WIPRO", "HCLTECH", "JSWSTEEL", "TATAMOTORS"
# ]
indian_stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
    "HDFC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "MARUTI.NS", "ULTRACEMCO.NS", "TITAN.NS", "BAJFINANCE.NS",
    "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "ITC.NS", "BAJAJFINSV.NS",
    "SUNPHARMA.NS", "TECHM.NS", "WIPRO.NS", "HCLTECH.NS", "JSWSTEEL.NS", "TATAMOTORS.NS"
]


def fetch_stock_news(symbols, limit=50, date_from=None, date_to=None):
    """Fetch news for specific stock symbols"""
    params = {
        'api_token': API_TOKEN,
        'symbols': ','.join(symbols),
        'limit': limit,
        'language': 'en'
    }
    
    if date_from:
        params['published_after'] = date_from
    if date_to:
        params['published_before'] = date_to
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return None

# def save_news_to_csv(news_data, filename):
#     """Convert news data to DataFrame and save as CSV"""
#     if not news_data or 'data' not in news_data:
#         print("No news data to save")
#         return
    
#     articles = []
#     for article in news_data['data']:
#         articles.append({
#             'title': article.get('title', ''),
#             'description': article.get('description', ''),
#             'url': article.get('url', ''),
#             'published_at': article.get('published_at', ''),
#             'source': article.get('source', ''),
#             'symbols': ','.join(article.get('symbols', [])),
#             'sentiment': article.get('sentiment', ''),
#             'entities': ','.join(article.get('entities', []))
#         })
    
#     df = pd.DataFrame(articles)
#     df.to_csv(filename, index=False)
#     print(f"Saved {len(articles)} articles to {filename}")

def save_news_to_csv(news_data, filename):
    """Convert news data to DataFrame and save as CSV"""
    if not news_data or 'data' not in news_data:
        print("No news data to save")
        return
    
    articles = []
    for article in news_data['data']:
        # Safely extract entity names
        entities = article.get('entities', [])
        entity_names = [e['name'] for e in entities if isinstance(e, dict) and 'name' in e]
        
        articles.append({
            'title': article.get('title', ''),
            'description': article.get('description', ''),
            'url': article.get('url', ''),
            'published_at': article.get('published_at', ''),
            'source': article.get('source', ''),
            'symbols': ','.join(article.get('symbols', [])),
            'sentiment': article.get('sentiment', ''),
            'entities': ','.join(entity_names)
        })
    
    df = pd.DataFrame(articles)
    df.to_csv(filename, index=False)
    print(f"Saved {len(articles)} articles to {filename}")


# Fetch news for all stocks (batch processing to respect rate limits)
batch_size = 10  # Process 10 stocks at a time
date_from = "2021-07-12"
date_to = "2025-07-12"

for i in range(0, len(indian_stocks), batch_size):
    batch = indian_stocks[i:i+batch_size]
    print(f"Fetching news for batch {i//batch_size + 1}: {batch}")
    
    news_data = fetch_stock_news(
        symbols=batch,
        limit=100,
        date_from=date_from,
        date_to=date_to
    )
    
    if news_data:
        filename = f"indian_stocks_news_batch_{i//batch_size + 1}.csv"
        save_news_to_csv(news_data, filename)
    
    # Rate limiting - wait between requests
    time.sleep(2)

print("News data collection completed!")
