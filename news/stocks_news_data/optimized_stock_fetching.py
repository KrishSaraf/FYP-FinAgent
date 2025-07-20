
import requests
import time
import csv
import json
import logging
import hashlib
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_caching_fetcher.log'),
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
    search_strategy: str = ""

class SmartCache:
    """Simple but effective caching system"""
    
    def __init__(self, cache_file: str = 'fetch_cache.json'):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """Load cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"Loaded cache: {len(cache.get('completed_requests', []))} previous API calls")
                return cache
            except:
                logger.warning("Cache file corrupted, starting fresh")
        
        return {
            'completed_requests': [],  # List of completed API request hashes
            'date_ranges_fetched': [],  # List of [start_date, end_date, strategy] 
            'last_update': None,
            'total_articles_found': 0
        }
    
    def _save_cache(self):
        """Save cache to file"""
        self.cache['last_update'] = datetime.now().isoformat()
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def is_request_completed(self, api_source: str, strategy: str, start_date: str, end_date: str, params: Dict = None) -> bool:
        """Check if this exact API request was already made"""
        # Create unique hash for this request
        request_data = f"{api_source}|{strategy}|{start_date}|{end_date}|{str(sorted((params or {}).items()))}"
        request_hash = hashlib.md5(request_data.encode()).hexdigest()
        
        is_completed = request_hash in self.cache['completed_requests']
        
        if is_completed:
            logger.info(f"⏭️  Skipping cached request: {strategy} ({start_date} to {end_date})")
        
        return is_completed
    
    def mark_request_completed(self, api_source: str, strategy: str, start_date: str, end_date: str, 
                             params: Dict = None, articles_found: int = 0):
        """Mark an API request as completed"""
        request_data = f"{api_source}|{strategy}|{start_date}|{end_date}|{str(sorted((params or {}).items()))}"
        request_hash = hashlib.md5(request_data.encode()).hexdigest()
        
        if request_hash not in self.cache['completed_requests']:
            self.cache['completed_requests'].append(request_hash)
            self.cache['total_articles_found'] += articles_found
            
            # Also track date ranges for easy viewing
            self.cache['date_ranges_fetched'].append({
                'api': api_source,
                'strategy': strategy,
                'start_date': start_date,
                'end_date': end_date,
                'articles_found': articles_found,
                'timestamp': datetime.now().isoformat()
            })
            
            self._save_cache()
            logger.info(f"✅ Cached: {strategy} ({start_date} to {end_date}) - {articles_found} articles")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'total_requests_cached': len(self.cache['completed_requests']),
            'total_articles_found': self.cache['total_articles_found'],
            'last_update': self.cache.get('last_update'),
            'date_ranges_count': len(self.cache['date_ranges_fetched'])
        }
    
    def clear_cache(self):
        """Clear the cache (useful for fresh start)"""
        self.cache = {
            'completed_requests': [],
            'date_ranges_fetched': [],
            'last_update': None,
            'total_articles_found': 0
        }
        self._save_cache()
        logger.info("🗑️  Cache cleared")

class ArticleStorage:
    """Simple storage for articles with duplicate detection"""
    
    def __init__(self, storage_file: str = 'articles_storage.json'):
        self.storage_file = storage_file
        self.articles = []
        self.url_hashes = set()
        self.title_hashes = set()
        self._load_storage()
    
    def _load_storage(self):
        """Load existing articles from storage"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    
                for article_data in data:
                    article = NewsArticle(**article_data)
                    self.articles.append(article)
                    self.url_hashes.add(hashlib.md5(article.url.encode()).hexdigest())
                    self.title_hashes.add(hashlib.md5(article.title.encode()).hexdigest())
                
                logger.info(f"📚 Loaded {len(self.articles)} existing articles from storage")
            except Exception as e:
                logger.warning(f"Could not load storage: {e}")
    
    def add_articles(self, new_articles: List[NewsArticle]) -> int:
        """Add new articles, avoiding duplicates"""
        added_count = 0
        
        for article in new_articles:
            url_hash = hashlib.md5(article.url.encode()).hexdigest()
            title_hash = hashlib.md5(article.title.encode()).hexdigest()
            
            # Check for duplicates
            if url_hash not in self.url_hashes and title_hash not in self.title_hashes:
                self.articles.append(article)
                self.url_hashes.add(url_hash)
                self.title_hashes.add(title_hash)
                added_count += 1
        
        if added_count > 0:
            self._save_storage()
            logger.info(f"💾 Added {added_count} new unique articles (total: {len(self.articles)})")
        
        return added_count
    
    def _save_storage(self):
        """Save articles to storage"""
        with open(self.storage_file, 'w') as f:
            json.dump([asdict(article) for article in self.articles], f, indent=2)
    
    def export_to_csv(self, filename: str):
        """Export articles to CSV"""
        if not self.articles:
            logger.warning("No articles to export")
            return
        
        fieldnames = ['uuid', 'title', 'description', 'url', 'published_at', 
                     'source', 'symbols', 'sentiment_score', 'keywords', 
                     'api_source', 'search_strategy']
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for article in self.articles:
                row = asdict(article)
                row['symbols'] = '|'.join(row['symbols'])
                writer.writerow(row)
        
        logger.info(f"📊 Exported {len(self.articles)} articles to {filename}")

class SmartMarketauxAPI:
    """Marketaux API with smart caching"""
    
    def __init__(self, api_key: str, cache: SmartCache):
        self.api_key = api_key
        self.base_url = "https://api.marketaux.com/v1"
        self.cache = cache
        self.daily_calls = 0
        self.daily_limit = 100
        
    def fetch_with_caching(self, strategy: str, start_date: str, end_date: str, params: Dict) -> List[NewsArticle]:
        """Fetch articles with smart caching"""
        
        # Check if this request was already made
        if self.cache.is_request_completed('marketaux', strategy, start_date, end_date, params):
            return []  # Skip, already fetched
        
        # Check daily limit
        if self.daily_calls >= self.daily_limit:
            logger.warning("⚠️  Marketaux daily limit reached")
            return []
        
        # Make the API request
        params.update({
            'api_token': self.api_key,
            'published_after': start_date,
            'published_before': end_date,
            'language': 'en',
            'limit': 3
        })
        
        try:
            logger.info(f"🔍 API Call: {strategy} ({start_date} to {end_date})")
            response = requests.get(f"{self.base_url}/news/all", params=params, timeout=30)
            self.daily_calls += 1
            
            if response.status_code == 200:
                data = response.json()
                articles = self._parse_response(data, strategy)
                
                # Cache this request
                self.cache.mark_request_completed(
                    'marketaux', strategy, start_date, end_date, params, len(articles)
                )
                
                return articles
            
            elif response.status_code == 429:
                logger.warning("⚠️  Rate limit hit")
                time.sleep(60)
                return []
            else:
                logger.warning(f"⚠️  API error: {response.status_code}")
                # Still cache failed requests to avoid repeating them
                self.cache.mark_request_completed(
                    'marketaux', strategy, start_date, end_date, params, 0
                )
                return []
                
        except Exception as e:
            logger.error(f"❌ API error: {str(e)}")
            return []
    
    def _parse_response(self, data: Dict, strategy: str) -> List[NewsArticle]:
        """Parse API response"""
        articles = []
        
        for item in data.get('data', []):
            symbols = []
            sentiment_score = None
            
            # Extract symbols
            for entity in item.get('entities', []):
                symbol = entity.get('symbol', '')
                if symbol and (symbol.endswith('.NS') or symbol.endswith('.BO')):
                    symbols.append(symbol)
                    if entity.get('sentiment_score') is not None:
                        sentiment_score = entity.get('sentiment_score')
            
            if not symbols:
                symbols = ['INDIAN_MARKET_GENERAL']
            
            article = NewsArticle(
                uuid=item.get('uuid', f"marketaux_{hash(item.get('url', ''))}"),
                title=item.get('title', ''),
                description=item.get('description', ''),
                url=item.get('url', ''),
                published_at=item.get('published_at', ''),
                source=item.get('source', ''),
                symbols=symbols,
                sentiment_score=sentiment_score,
                keywords=item.get('keywords', ''),
                api_source='marketaux',
                search_strategy=strategy
            )
            articles.append(article)
        
        return articles

class SmartStockNewsFetcher:
    """Smart fetcher with caching and resume capabilities"""
    
    def __init__(self, config_file: str = 'smart_config.json'):
        self.config = self._load_config(config_file)
        self.cache = SmartCache()
        self.storage = ArticleStorage()
        self.marketaux_api = None
        
        self._initialize_apis()
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration"""
        default_config = {
            "api_keys": {
                "marketaux": "YOUR_MARKETAUX_API_KEY"
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
                "IOC.NS", "UPL.NS", "CIPLA.NS", "TATASTEEL.NS", "HDFCLIFE.NS", "BAJAJ-AUTO.NS"
            ],
            "start_date": "2024-06-06",
            "end_date": "2025-06-06",
            "batch_size_days": 7,
            "output_file": "smart_cached_news.csv"
        }
        
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
        else:
            config = default_config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Created config: {config_file}")
        
        return config
    
    def _initialize_apis(self):
        """Initialize APIs"""
        if self.config['api_keys']['marketaux'] != "YOUR_MARKETAUX_API_KEY":
            self.marketaux_api = SmartMarketauxAPI(self.config['api_keys']['marketaux'], self.cache)
            logger.info("✅ Smart Marketaux API initialized")
        else:
            logger.error("❌ Please update your Marketaux API key in config")
            exit(1)
    
    def fetch_smart(self):
        """Smart fetching with caching and resume"""
        logger.info("🧠 SMART CACHING NEWS FETCHER")
        logger.info("=" * 50)
        
        # Show cache stats
        cache_stats = self.cache.get_cache_stats()
        logger.info(f"📊 Cache Status:")
        logger.info(f"   Previous API calls: {cache_stats['total_requests_cached']}")
        logger.info(f"   Articles in storage: {len(self.storage.articles)}")
        logger.info(f"   Last update: {cache_stats.get('last_update', 'Never')}")
        
        # Define search strategies
        strategies = self._get_search_strategies()
        
        # Process date ranges
        start_date = datetime.strptime(self.config['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(self.config['end_date'], '%Y-%m-%d')
        batch_size = timedelta(days=self.config['batch_size_days'])
        
        current_date = start_date
        total_new_articles = 0
        skipped_requests = 0
        api_calls_made = 0
        
        while current_date < end_date:
            batch_end = min(current_date + batch_size, end_date)
            start_str = current_date.strftime('%Y-%m-%d')
            end_str = batch_end.strftime('%Y-%m-%d')
            
            logger.info(f"📅 Processing: {start_str} to {end_str}")
            
            # Try each strategy for this date range
            for strategy_name, strategy_params in strategies.items():
                
                # Check if already cached
                if self.cache.is_request_completed('marketaux', strategy_name, start_str, end_str, strategy_params):
                    skipped_requests += 1
                    continue
                
                # Make API call
                articles = self.marketaux_api.fetch_with_caching(
                    strategy_name, start_str, end_str, strategy_params
                )
                
                if articles:
                    new_count = self.storage.add_articles(articles)
                    total_new_articles += new_count
                    api_calls_made += 1
                
                time.sleep(1.5)  # Respectful delay
                
                # Check daily limit
                if self.marketaux_api.daily_calls >= self.marketaux_api.daily_limit:
                    logger.warning("🛑 Daily API limit reached")
                    break
            
            if self.marketaux_api.daily_calls >= self.marketaux_api.daily_limit:
                break
            
            current_date = batch_end
        
        # Final export and statistics
        self.storage.export_to_csv(self.config['output_file'])
        
        logger.info("🎉 SMART FETCHING COMPLETED!")
        logger.info("=" * 50)
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   Total articles in storage: {len(self.storage.articles)}")
        logger.info(f"   New articles this run: {total_new_articles}")
        logger.info(f"   API calls made: {api_calls_made}")
        logger.info(f"   API calls skipped (cached): {skipped_requests}")
        logger.info(f"   Efficiency: {total_new_articles/max(api_calls_made, 1):.1f} articles per API call")
        logger.info(f"📁 Results saved to: {self.config['output_file']}")
    
    def _get_search_strategies(self) -> Dict[str, Dict]:
        """Define search strategies"""
        return {
            'general_indian_market': {'countries': 'in'},
            'bse_sensex_trading': {'search': 'BSE Sensex trading volume'},
            'nse_nifty_index': {'search': 'NSE Nifty index performance'},
            'indian_banking_sector': {'search': 'Indian banking sector earnings'},
            'indian_it_companies': {'search': 'Indian IT companies revenue'},
            'indian_pharmaceutical': {'search': 'Indian pharmaceutical companies'},
            'indian_automobile': {'search': 'Indian automobile sector'},
            'reliance_industries': {'search': '"Reliance Industries"'},
            'tata_group_companies': {'search': 'Tata group companies TCS'},
            'hdfc_icici_banking': {'search': 'HDFC Bank ICICI Bank'},
            'infosys_wipro_it': {'search': 'Infosys Wipro IT services'},
            'indian_oil_gas': {'search': 'Indian oil gas ONGC'},
            'indian_steel_sector': {'search': 'Indian steel JSW Tata Steel'},
            'indian_cement_companies': {'search': 'Indian cement UltraTech'},
            'indian_telecom_airtel': {'search': 'Indian telecom Bharti Airtel'}
        }
    
    def clear_cache_and_restart(self):
        """Clear cache for fresh start"""
        self.cache.clear_cache()
        logger.info("🔄 Cache cleared. Run fetch_smart() for fresh start.")
    
    def show_status(self):
        """Show current status"""
        cache_stats = self.cache.get_cache_stats()
        
        print("📊 SMART FETCHER STATUS")
        print("=" * 40)
        print(f"Articles in storage: {len(self.storage.articles)}")
        print(f"Cached API requests: {cache_stats['total_requests_cached']}")
        print(f"Last update: {cache_stats.get('last_update', 'Never')}")
        print(f"Config file: smart_config.json")
        print(f"Cache file: fetch_cache.json")
        print(f"Storage file: articles_storage.json")

def main():
    """Main function"""
    print("🧠 Smart Caching Indian Stock News Fetcher")
    print("=" * 50)
    print("Features:")
    print("✅ Remembers previous API calls")
    print("✅ Skips duplicate requests")
    print("✅ Resumes from where you left off")
    print("✅ No wasted API calls")
    print()
    
    fetcher = SmartStockNewsFetcher()
    
    # Show current status
    fetcher.show_status()
    print()
    
    # Ask what to do
    print("Options:")
    print("1. Continue fetching (resume from cache)")
    print("2. Show current status only")
    print("3. Clear cache and start fresh")
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        fetcher.fetch_smart()
    elif choice == "2":
        fetcher.show_status()
    elif choice == "3":
        confirm = input("Are you sure you want to clear cache? (y/n): ").lower()
        if confirm == 'y':
            fetcher.clear_cache_and_restart()
            print("Cache cleared. Run again to start fresh.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()