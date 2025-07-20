
import requests
import time
import csv
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import os
from dataclasses import dataclass, asdict
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_news_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Data class for storing news article information"""
    uuid: str
    title: str
    description: str
    url: str
    published_at: str
    source: str
    symbols: List[str]
    sentiment_score: Optional[float] = None
    keywords: str = ""
    api_source: str = ""

class RateLimiter:
    """Rate limiter to manage API call frequency"""
    
    def __init__(self, calls_per_minute: int, calls_per_day: int):
        self.calls_per_minute = calls_per_minute
        self.calls_per_day = calls_per_day
        self.minute_calls = []
        self.daily_calls = 0
        self.last_reset = datetime.now().date()
    
    def can_make_request(self) -> bool:
        """Check if we can make a request without exceeding limits"""
        now = datetime.now()
        current_date = now.date()
        
        # Reset daily counter if new day
        if current_date > self.last_reset:
            self.daily_calls = 0
            self.last_reset = current_date
        
        # Remove calls older than 1 minute
        self.minute_calls = [call_time for call_time in self.minute_calls 
                           if (now - call_time).seconds < 60]
        
        # Check limits
        return (len(self.minute_calls) < self.calls_per_minute and 
                self.daily_calls < self.calls_per_day)
    
    def record_request(self):
        """Record that a request was made"""
        self.minute_calls.append(datetime.now())
        self.daily_calls += 1
    
    def wait_time(self) -> int:
        """Calculate how long to wait before next request"""
        if not self.minute_calls:
            return 0
        
        oldest_call = min(self.minute_calls)
        seconds_since_oldest = (datetime.now() - oldest_call).seconds
        return max(0, 60 - seconds_since_oldest + 1)

class MarketauxAPI:
    """Marketaux API client - Primary source for Indian stocks"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.marketaux.com/v1"
        self.rate_limiter = RateLimiter(calls_per_minute=50, calls_per_day=100)  # Free tier
        
    def fetch_news(self, symbols: List[str] = None, 
                   published_after: str = None, 
                   published_before: str = None, 
                   limit: int = 3) -> List[NewsArticle]:
        """Fetch news from Marketaux API"""
        
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.wait_time()
            logger.info(f"Rate limit reached. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        
        params = {
            'api_token': self.api_key,
            'countries': 'in',  # India
            'language': 'en',
            'limit': min(limit, 3)  # Free tier max 3 articles per request
        }
        
        if symbols:
            # Convert Indian symbols to proper format
            formatted_symbols = self._format_indian_symbols(symbols)
            params['symbols'] = ','.join(formatted_symbols)
            params['filter_entities'] = 'true'
        
        if published_after:
            params['published_after'] = published_after
        
        if published_before:
            params['published_before'] = published_before
        
        try:
            response = requests.get(f"{self.base_url}/news/all", params=params, timeout=30)
            self.rate_limiter.record_request()
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_marketaux_response(data)
            elif response.status_code == 429:
                logger.warning("Marketaux rate limit exceeded")
                time.sleep(60)
                return []
            else:
                logger.error(f"Marketaux API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching from Marketaux: {str(e)}")
            return []
    
    def _format_indian_symbols(self, symbols: List[str]) -> List[str]:
        """Convert Indian stock symbols to Marketaux format"""
        formatted = []
        for symbol in symbols:
            if symbol.endswith('.NS'):
                # NSE format
                formatted.append(symbol)
            elif symbol.endswith('.BO'):
                # BSE format
                formatted.append(symbol)
            else:
                # Add NSE suffix if no exchange specified
                formatted.append(f"{symbol}.NS")
        return formatted
    
    def _parse_marketaux_response(self, data: Dict) -> List[NewsArticle]:
        """Parse Marketaux API response into NewsArticle objects"""
        articles = []
        
        for item in data.get('data', []):
            symbols = []
            sentiment_score = None
            
            # Extract symbols and sentiment from entities
            for entity in item.get('entities', []):
                if entity.get('country') == 'in':  # Indian entities
                    symbols.append(entity.get('symbol', ''))
                    if entity.get('sentiment_score') is not None:
                        sentiment_score = entity.get('sentiment_score')
            
            if not symbols:
                continue  # Skip if no Indian symbols found
            
            article = NewsArticle(
                uuid=item.get('uuid', ''),
                title=item.get('title', ''),
                description=item.get('description', ''),
                url=item.get('url', ''),
                published_at=item.get('published_at', ''),
                source=item.get('source', ''),
                symbols=symbols,
                sentiment_score=sentiment_score,
                keywords=item.get('keywords', ''),
                api_source='marketaux'
            )
            articles.append(article)
        
        return articles

class NewsAPIClient:
    """NewsAPI client - Secondary source for Indian business news"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.rate_limiter = RateLimiter(calls_per_minute=50, calls_per_day=100)  # Free tier
        
    def fetch_news(self, symbols: List[str] = None, 
                   from_date: str = None, 
                   to_date: str = None,
                   page_size: int = 20) -> List[NewsArticle]:
        """Fetch Indian business news from NewsAPI"""
        
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.wait_time()
            logger.info(f"NewsAPI rate limit reached. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        
        # Build search query for Indian stocks
        query_parts = ['India stock market', 'BSE', 'NSE', 'Indian stocks']
        if symbols:
            # Add specific company names/symbols
            query_parts.extend([self._symbol_to_company_name(s) for s in symbols[:5]])
        
        params = {
            'apiKey': self.api_key,
            'q': ' OR '.join(query_parts),
            'country': 'in',
            'category': 'business',
            'language': 'en',
            'pageSize': min(page_size, 20),  # Free tier limit
            'sortBy': 'publishedAt'
        }
        
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        try:
            response = requests.get(f"{self.base_url}/everything", params=params, timeout=30)
            self.rate_limiter.record_request()
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_newsapi_response(data, symbols)
            elif response.status_code == 429:
                logger.warning("NewsAPI rate limit exceeded")
                time.sleep(60)
                return []
            else:
                logger.error(f"NewsAPI error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {str(e)}")
            return []
    
    def _symbol_to_company_name(self, symbol: str) -> str:
        """Convert stock symbols to company names for better search"""
        symbol_map = {
            'RELIANCE.NS': 'Reliance Industries',
            'TCS.NS': 'Tata Consultancy Services',
            'HDFCBANK.NS': 'HDFC Bank',
            'INFY.NS': 'Infosys',
            'ICICIBANK.NS': 'ICICI Bank',
            'HINDUNILVR.NS': 'Hindustan Unilever',
            'HDFC.NS': 'HDFC',
            'SBIN.NS': 'State Bank of India',
            'BHARTIARTL.NS': 'Bharti Airtel',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank',
            'LT.NS': 'Larsen & Toubro',
            'ASIANPAINT.NS': 'Asian Paints',
            'AXISBANK.NS': 'Axis Bank',
            'MARUTI.NS': 'Maruti Suzuki',
            'ULTRACEMCO.NS': 'UltraTech Cement',
            'TITAN.NS': 'Titan Company',
            'BAJFINANCE.NS': 'Bajaj Finance',
            'NESTLEIND.NS': 'Nestle India',
            'ONGC.NS': 'Oil and Natural Gas Corporation',
            'POWERGRID.NS': 'Power Grid Corporation',
            'NTPC.NS': 'NTPC Limited',
            'ITC.NS': 'ITC Limited',
            'BAJAJFINSV.NS': 'Bajaj Finserv',
            'SUNPHARMA.NS': 'Sun Pharmaceutical',
            'TECHM.NS': 'Tech Mahindra',
            'WIPRO.NS': 'Wipro',
            'HCLTECH.NS': 'HCL Technologies',
            'JSWSTEEL.NS': 'JSW Steel',
            'TATAMOTORS.NS': 'Tata Motors',
            'ADANIPORTS.NS': 'Adani Ports',
            'BRITANNIA.NS': 'Britannia Industries',
            'SHREECEM.NS': 'Shree Cement',
            'TATACONSUM.NS': 'Tata Consumer Products',
            'BPCL.NS': 'Bharat Petroleum',
            'INDUSINDBK.NS': 'IndusInd Bank',
            'GRASIM.NS': 'Grasim Industries',
            'EICHERMOT.NS': 'Eicher Motors',
            'DRREDDY.NS': 'Dr Reddys Laboratories',
            'COALINDIA.NS': 'Coal India',
            'HEROMOTOCO.NS': 'Hero MotoCorp',
            'IOC.NS': 'Indian Oil Corporation',
            'UPL.NS': 'UPL Limited',
            'CIPLA.NS': 'Cipla',
            'TATASTEEL.NS': 'Tata Steel',
            'HDFCLIFE.NS': 'HDFC Life Insurance',
            'BAJAJ-AUTO.NS': 'Bajaj Auto'
        }
        return symbol_map.get(symbol, symbol.replace('.NS', '').replace('.BO', ''))
    
    def _parse_newsapi_response(self, data: Dict, symbols: List[str] = None) -> List[NewsArticle]:
        """Parse NewsAPI response into NewsArticle objects"""
        articles = []
        
        for item in data.get('articles', []):
            # Try to extract relevant symbols from content
            relevant_symbols = self._extract_symbols_from_content(
                item.get('title', '') + ' ' + item.get('description', ''), 
                symbols
            )
            
            article = NewsArticle(
                uuid=f"newsapi_{hash(item.get('url', ''))}", 
                title=item.get('title', ''),
                description=item.get('description', ''),
                url=item.get('url', ''),
                published_at=item.get('publishedAt', ''),
                source=item.get('source', {}).get('name', ''),
                symbols=relevant_symbols,
                keywords='',
                api_source='newsapi'
            )
            articles.append(article)
        
        return articles
    
    def _extract_symbols_from_content(self, content: str, symbols: List[str] = None) -> List[str]:
        """Extract relevant stock symbols from news content"""
        if not symbols:
            return ['GENERAL_INDIAN_MARKET']
        
        found_symbols = []
        content_upper = content.upper()
        
        for symbol in symbols:
            company_name = self._symbol_to_company_name(symbol).upper()
            if company_name in content_upper or symbol.replace('.NS', '').replace('.BO', '') in content_upper:
                found_symbols.append(symbol)
        
        return found_symbols if found_symbols else ['GENERAL_INDIAN_MARKET']

class AlphaVantageAPI:
    """Alpha Vantage API client - Backup source with sentiment analysis"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limiter = RateLimiter(calls_per_minute=5, calls_per_day=25)  # Free tier
        
    def fetch_news(self, symbols: List[str] = None, 
                   time_from: str = None, 
                   time_to: str = None,
                   limit: int = 50) -> List[NewsArticle]:
        """Fetch news with sentiment from Alpha Vantage"""
        
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.wait_time()
            logger.info(f"Alpha Vantage rate limit reached. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'apikey': self.api_key,
            'limit': min(limit, 50)  # Free tier limit
        }
        
        if symbols:
            # Focus on major Indian stocks that Alpha Vantage covers
            major_symbols = [s for s in symbols if s in self._get_supported_indian_symbols()]
            if major_symbols:
                params['tickers'] = ','.join(major_symbols[:5])  # Limit to 5 symbols
        
        if time_from:
            params['time_from'] = time_from
        if time_to:
            params['time_to'] = time_to
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            self.rate_limiter.record_request()
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_alphavantage_response(data)
            elif response.status_code == 429:
                logger.warning("Alpha Vantage rate limit exceeded")
                time.sleep(300)  # Wait 5 minutes
                return []
            else:
                logger.error(f"Alpha Vantage API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching from Alpha Vantage: {str(e)}")
            return []
    
    def _get_supported_indian_symbols(self) -> Set[str]:
        """Return set of Indian symbols supported by Alpha Vantage"""
        return {
            'INFY', 'WIT', 'TTM', 'VEDL', 'IBN', 'HDB', 'SIFY', 'INDA'
        }
    
    def _parse_alphavantage_response(self, data: Dict) -> List[NewsArticle]:
        """Parse Alpha Vantage response into NewsArticle objects"""
        articles = []
        
        for item in data.get('feed', []):
            # Extract ticker symbols
            symbols = []
            for ticker_info in item.get('ticker_sentiment', []):
                ticker = ticker_info.get('ticker', '')
                if ticker in self._get_supported_indian_symbols():
                    symbols.append(ticker)
            
            if not symbols:
                continue
            
            article = NewsArticle(
                uuid=f"alphavantage_{hash(item.get('url', ''))}",
                title=item.get('title', ''),
                description=item.get('summary', ''),
                url=item.get('url', ''),
                published_at=item.get('time_published', ''),
                source=item.get('source', ''),
                symbols=symbols,
                sentiment_score=float(item.get('overall_sentiment_score', 0)),
                keywords='',
                api_source='alphavantage'
            )
            articles.append(article)
        
        return articles

class StockNewsFetcher:
    """Main class orchestrating news fetching from multiple APIs"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config = self._load_config(config_file)
        self.apis = self._initialize_apis()
        self.fetched_articles = set()  # Track UUIDs to avoid duplicates
        self.results = []
        
        # Graceful shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "api_keys": {
                "marketaux": "YOUR_MARKETAUX_API_KEY",
                "newsapi": "YOUR_NEWSAPI_API_KEY", 
                "alphavantage": "YOUR_ALPHAVANTAGE_API_KEY"
            },
            "symbols": [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "HINDUNILVR.NS", "HDFC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
                "LT.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "ULTRACEMCO.NS",
                "TITAN.NS", "BAJFINANCE.NS", "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS",
                "NTPC.NS", "ITC.NS", "BAJAJFINSV.NS", "SUNPHARMA.NS", "TECHM.NS",
                "WIPRO.NS", "HCLTECH.NS", "JSWSTEEL.NS", "TATAMOTORS.NS", "ADANIPORTS.NS",
                "BRITANNIA.NS", "SHREECEM.NS", "TATACONSUM.NS", "BPCL.NS", "INDUSINDBK.NS",
                "GRASIM.NS", "EICHERMOT.NS", "DRREDDY.NS", "COALINDIA.NS", "HEROMOTOCO.NS",
                "IOC.NS", "UPL.NS", "CIPLA.NS", "TATASTEEL.NS", "HDFCLIFE.NS",
                "BAJAJ-AUTO.NS"
            ],
            "output_file": "indian_stock_news.csv",
            "start_date": "2024-06-06",
            "end_date": "2025-06-06",
            "batch_size_days": 30
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            # Merge with defaults
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
        else:
            config = default_config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Created default config file: {config_file}")
            logger.info("Please update the API keys in the config file before running.")
        
        return config
    
    def _initialize_apis(self) -> Dict:
        """Initialize API clients"""
        apis = {}
        
        if self.config['api_keys']['marketaux'] != "YOUR_MARKETAUX_API_KEY":
            apis['marketaux'] = MarketauxAPI(self.config['api_keys']['marketaux'])
            logger.info("Initialized Marketaux API")
        
        if self.config['api_keys']['newsapi'] != "YOUR_NEWSAPI_API_KEY":
            apis['newsapi'] = NewsAPIClient(self.config['api_keys']['newsapi'])
            logger.info("Initialized NewsAPI")
        
        if self.config['api_keys']['alphavantage'] != "YOUR_ALPHAVANTAGE_API_KEY":
            apis['alphavantage'] = AlphaVantageAPI(self.config['api_keys']['alphavantage'])
            logger.info("Initialized Alpha Vantage API")
        
        if not apis:
            logger.error("No valid API keys found. Please update config.json")
            sys.exit(1)
        
        return apis
    
    def fetch_historical_news(self):
        """Fetch news for the specified date range"""
        start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(self.config['end_date'], '%Y-%m-%d')
        batch_size = timedelta(days=self.config['batch_size_days'])
        
        current_date = start_date
        total_batches = int((end_date - start_date).days / self.config['batch_size_days']) + 1
        batch_count = 0
        
        logger.info(f"Starting historical fetch from {start_date.date()} to {end_date.date()}")
        logger.info(f"Processing in {total_batches} batches of {self.config['batch_size_days']} days each")
        
        while current_date < end_date:
            batch_count += 1
            batch_end = min(current_date + batch_size, end_date)
            
            logger.info(f"Processing batch {batch_count}/{total_batches}: {current_date.date()} to {batch_end.date()}")
            
            self._fetch_batch(current_date, batch_end)
            
            # Save progress after each batch
            self._save_results()
            
            current_date = batch_end
            
            # Wait between batches to respect rate limits
            if current_date < end_date:
                logger.info("Waiting 60 seconds between batches...")
                time.sleep(60)
        
        logger.info(f"Historical fetch completed. Total articles: {len(self.results)}")
    
    def _fetch_batch(self, start_date: datetime, end_date: datetime):
        """Fetch news for a specific date range"""
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch from each API
        for api_name, api_client in self.apis.items():
            logger.info(f"Fetching from {api_name} for {start_str} to {end_str}")
            
            try:
                if api_name == 'marketaux':
                    articles = api_client.fetch_news(
                        symbols=self.config['symbols'],
                        published_after=start_str,
                        published_before=end_str
                    )
                elif api_name == 'newsapi':
                    articles = api_client.fetch_news(
                        symbols=self.config['symbols'],
                        from_date=start_str,
                        to_date=end_str
                    )
                elif api_name == 'alphavantage':
                    # Alpha Vantage uses different date format
                    time_from = start_date.strftime('%Y%m%dT0000')
                    time_to = end_date.strftime('%Y%m%dT2359')
                    articles = api_client.fetch_news(
                        symbols=self.config['symbols'],
                        time_from=time_from,
                        time_to=time_to
                    )
                
                # Add unique articles
                new_articles = 0
                for article in articles:
                    if article.uuid not in self.fetched_articles:
                        self.fetched_articles.add(article.uuid)
                        self.results.append(article)
                        new_articles += 1
                
                logger.info(f"Added {new_articles} new articles from {api_name}")
                
                # Respect rate limits between APIs
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error fetching from {api_name}: {str(e)}")
                continue
    
    def _save_results(self):
        """Save results to CSV file"""
        if not self.results:
            return
        
        fieldnames = ['uuid', 'title', 'description', 'url', 'published_at', 
                     'source', 'symbols', 'sentiment_score', 'keywords', 'api_source']
        
        with open(self.config['output_file'], 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for article in self.results:
                row = asdict(article)
                row['symbols'] = '|'.join(row['symbols'])  # Join symbols with pipe
                writer.writerow(row)
        
        logger.info(f"Saved {len(self.results)} articles to {self.config['output_file']}")
    
    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info("Shutdown signal received. Saving current progress...")
        self._save_results()
        sys.exit(0)
    
    def get_statistics(self) -> Dict:
        """Get statistics about fetched data"""
        if not self.results:
            return {}
        
        stats = {
            'total_articles': len(self.results),
            'by_api': {},
            'by_source': {},
            'date_range': {
                'earliest': min(article.published_at for article in self.results),
                'latest': max(article.published_at for article in self.results)
            },
            'top_symbols': {}
        }
        
        # Count by API
        for article in self.results:
            api = article.api_source
            stats['by_api'][api] = stats['by_api'].get(api, 0) + 1
        
        # Count by source
        for article in self.results:
            source = article.source
            stats['by_source'][source] = stats['by_source'].get(source, 0) + 1
        
        # Count by symbols
        symbol_count = {}
        for article in self.results:
            for symbol in article.symbols:
                symbol_count[symbol] = symbol_count.get(symbol, 0) + 1
        
        # Top 10 symbols
        stats['top_symbols'] = dict(sorted(symbol_count.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True)[:10])
        
        return stats

def main():
    """Main function"""
    print("Indian Stock News Fetcher")
    print("=" * 50)
    print()
    
    # Check for config file
    if not os.path.exists('config.json'):
        print("Creating default config.json file...")
        fetcher = StockNewsFetcher()
        print("Please update the API keys in config.json and run again.")
        return
    
    # Initialize fetcher
    fetcher = StockNewsFetcher()
    
    print("Configuration loaded:")
    print(f"  Date range: {fetcher.config['start_date']} to {fetcher.config['end_date']}")
    print(f"  Symbols: {len(fetcher.config['symbols'])} stocks")
    print(f"  APIs: {', '.join(fetcher.apis.keys())}")
    print(f"  Output: {fetcher.config['output_file']}")
    print()
    
    # Start fetching
    try:
        fetcher.fetch_historical_news()
        
        # Print statistics
        stats = fetcher.get_statistics()
        print("\nFetching Statistics:")
        print("=" * 30)
        print(f"Total articles: {stats.get('total_articles', 0)}")
        print(f"By API: {stats.get('by_api', {})}")
        print(f"Top symbols: {dict(list(stats.get('top_symbols', {}).items())[:5])}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    main()