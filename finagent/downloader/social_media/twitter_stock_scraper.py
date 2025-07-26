import os
import json
import logging
import requests
from typing import List, Optional
from dotenv import load_dotenv
from datetime import datetime, timedelta

BASE_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
USER_TWEETS_URL = "https://api.twitterapi.io/twitter/user/get_user_last_tweets"

class TwitterStockScraper:
    def __init__(self, root: str = "", api_key: str = None, stocks_path: str = None,
                 handles_path: str = None, workdir: str = "", tag: str = ""):
        self.root = root
        load_dotenv(os.path.join(root, "apikeys.env"))
        self.api_key = api_key if api_key else os.environ.get("TWITTER_API_KEY")

        if not self.api_key:
            raise ValueError("Twitter API key not provided. Please set TWITTER_API_KEY in apikeys.env.")

        self.stocks_path = os.path.join(root, stocks_path) if stocks_path else None
        self.handles_path = os.path.join(root, handles_path) if handles_path else None
        self.workdir = os.path.join(root, workdir, tag)
        self.tag = tag

        # Setup logging
        self.log_path = os.path.join(self.workdir, f"{tag}.log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.stocks = self._init_stocks()
        self.company_handles = self._load_company_handles()

    def _init_stocks(self) -> List[str]:
        """Initialize the list of stocks from the provided file."""
        try:
            if self.stocks_path and os.path.exists(self.stocks_path):
                with open(self.stocks_path, "r") as f:
                    return [line.strip() for line in f.readlines()]
            else:
                self.logger.warning("Stocks file not found. Using an empty stock list.")
                return []
        except Exception as e:
            self.logger.error(f"Error reading stocks file: {e}")
            return []

    def _load_company_handles(self) -> dict:
        """Load stock to company Twitter handle mapping."""
        try:
            if self.handles_path and os.path.exists(self.handles_path):
                with open(self.handles_path, "r") as f:
                    return json.load(f)
            else:
                self.logger.warning("Handles file not found. No user tweets will be fetched.")
                return {}
        except Exception as e:
            self.logger.error(f"Error reading handles file: {e}")
            return {}

    def _fetch_tweets(self, stock: str, limit: int = 100, start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> List[dict]:
        """Fetch tweets mentioning a specific stock."""
        query = f'({stock} OR ${stock.upper()}) lang:en filter:retweets false'
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "since": start_date,
            "until": end_date,
            "limit": limit,
            "sort_by": "favorite_count",
            "include_user_data": True,
            "include_metrics": True
        }

        try:
            self.logger.info(f"Fetching public tweets for stock: {stock}")
            response = requests.get(BASE_URL, headers=headers, params=payload)
            response.raise_for_status()
            return response.json().get("tweets", [])
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching tweets for stock {stock}: {e}")
            return []

    def _fetch_user_mentions_with_keywords(self, username: str, keywords: List[str], limit: int = 100,
                                       start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[dict]:
        """Search tweets from a user's handle that match specific keywords."""
        keyword_query = " OR ".join(keywords)
        query = f'from:{username} ({keyword_query}) lang:en'
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "since": start_date,
            "until": end_date,
            "limit": limit,
            "sort_by": "favorite_count",
            "include_user_data": True,
            "include_metrics": True
        }

        try:
            self.logger.info(f"Searching tweets from @{username} with keywords: {keywords}")
            response = requests.get(BASE_URL, headers=headers, params=payload)
            response.raise_for_status()
            return response.json().get("tweets", [])
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error searching tweets from @{username}: {e}")
            return []


    def scrape_twitter_posts(self, stocks: Optional[List[str]] = None, limit: int = 100,
                             start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Scrape tweets mentioning stocks and from official handles."""
        stocks = stocks if stocks else self.stocks
        if not stocks:
            self.logger.error("No stocks provided for scraping.")
            return

        if not start_date or not end_date:
            self.logger.error("Both start_date and end_date must be provided.")
            return

        os.makedirs(self.workdir, exist_ok=True)

        for stock in stocks:
            # Fetch public mentions
            tweets = self._fetch_tweets(stock, limit, start_date, end_date)
            public_file = os.path.join(self.workdir, f"{stock}_tweets.json")
            with open(public_file, "w") as f:
                json.dump(tweets, f, indent=4)
            self.logger.info(f"Saved {len(tweets)} public tweets for stock {stock} to {public_file}")

            # Fetch from company handles (if available)
            handles = self.company_handles.get(stock.upper(), [])
            for handle in handles:
                keywords = ["results", "profit", "earnings", "quarter", "q1", "q2", "q3", "q4"]
                user_tweets = self._fetch_user_mentions_with_keywords(handle, keywords, limit, start_date, end_date)
                user_file = os.path.join(self.workdir, f"{stock}_user_{handle}_tweets.json")
                with open(user_file, "w") as f:
                    json.dump(user_tweets, f, indent=4)
                self.logger.info(f"Saved {len(user_tweets)} tweets from @{handle} to {user_file}")

        self.logger.info("Twitter scraping completed successfully.")
