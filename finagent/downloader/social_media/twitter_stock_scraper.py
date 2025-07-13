import os
import time
import json
import logging
from typing import List, Optional, Union
from twscrape import API, gather
from twscrape.logger import set_log_level
from dotenv import load_dotenv
from datetime import datetime, timezone
import asyncio
from playwright_stealth import stealth_async
from playwright.async_api import async_playwright
from finagent.utils.proxy_rotator import ProxyRotator  # Import the ProxyRotator class

class TwitterStockScraper:
    def __init__(self,
                 root: str = "",
                 username: str = None,
                 password: str = None,
                 delay: int = 1,
                 stocks_path: str = None,
                 workdir: str = "",
                 tag: str = "",
                 proxy_username: str = None,
                 proxy_password: str = None,
                 proxy_server: str = None,
                 **kwargs):
        self.root = root
        load_dotenv(os.path.join(root, "apikeys.env"))
        self.username = username if username else os.environ.get("TWITTER_USERNAME")
        self.password = password if password else os.environ.get("TWITTER_PASSWORD")

        if not self.username or not self.password:
            raise ValueError("Twitter credentials not provided. Please set TWITTER_USERNAME and TWITTER_PASSWORD in apikeys.env.")

        self.delay = delay
        self.stocks_path = os.path.join(root, stocks_path) if stocks_path else None
        self.workdir = os.path.join(root, workdir, tag)
        self.tag = tag

        # Setup logging
        self.log_path = os.path.join(self.workdir, "{}.log".format(tag))
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)  # Ensure the directory exists
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_path),  # Log to file
                logging.StreamHandler()  # Log to terminal
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.stocks = self._init_stocks()

        self._proxy_username = proxy_username if proxy_username else os.environ.get("THORDATA_USERNAME")
        self._proxy_password = proxy_password if proxy_password else os.environ.get("THORDATA_PASSWORD")

        if not self._proxy_username or not self._proxy_password:
            raise ValueError("Proxy credentials not provided. Please set THORDATA_USERNAME and THORDATA_PASSWORD in apikeys.env.")

        # Initialize ProxyRotator with ThorData configuration
        self.proxy_rotator = ProxyRotator(
            username=self._proxy_username,
            password=self._proxy_password,
            proxy_server=proxy_server or "fiip79eu.pr.thordata.net:9999"  # Default ThorData endpoint
        )
        self.current_proxy = None

        # Initialize Twitter API client
        self.api = API()

    def _init_stocks(self) -> List[str]:
        """Initialize the list of stocks from the provided file."""
        try:
            if self.stocks_path and os.path.exists(self.stocks_path):
                with open(self.stocks_path) as op:
                    stocks = [line.strip() for line in op.readlines()]
                return stocks
            else:
                self.logger.warning("Stocks file not found. Using an empty stock list.")
                return []
        except Exception as e:
            self.logger.error(f"Error reading stocks file: {e}")
            return []

    async def initialize(self):
        """Initialize the scraper with proxies and accounts."""
        # Refresh proxies if ProxyRotator is enabled
        if self.proxy_rotator is None:
            self.logger.error("ProxyRotator is None. Ensure it is initialized.")
            raise ValueError("ProxyRotator is not initialized.")

        if self.proxy_rotator:
            self.proxy_rotator.refresh_proxies()
            await self.rotate_proxy()
        
        # Setup Twitter accounts
        await self._setup_accounts(account_proxy=self.current_proxy.to_url() if self.current_proxy else None)

    async def rotate_proxy(self):
        """Rotate to the next proxy."""
        if self.proxy_rotator:
            self.current_proxy = self.proxy_rotator.get_next_proxy()
            if self.current_proxy:
                proxy_url = self.current_proxy.to_url()
                self.logger.info(f"Rotating to proxy: {proxy_url}")
                self.api.proxy = proxy_url
            else:
                self.logger.warning("No available proxies")
        else:
            self.logger.info("Proxy rotation is disabled")

    async def generate_cookies_with_proxy(self, proxy_url: str, username: str, password: str, output_path: str = "cookies.json"):
        """Generate cookies by logging into Twitter using a proxy server.

        Args:
            proxy_url: Proxy server URL (e.g., "http://user:pass@proxy.com:8080").
            username: Twitter username.
            password: Twitter password.
            output_path: Path to save the generated cookies (default: "cookies.json").
        """
        try:
            # Parse proxy URL for ThorData format
            self.logger.info(f"Parsing proxy URL: {proxy_url}")
            
            if "://" not in proxy_url:
                raise ValueError(f"Invalid proxy URL format: {proxy_url}")
            
            protocol, rest = proxy_url.split("://", 1)
            
            if "@" in rest:
                auth_part, server_part = rest.split("@", 1)
                if ":" in auth_part:
                    proxy_username, proxy_password = auth_part.split(":", 1)
                else:
                    raise ValueError("Invalid authentication format in proxy URL")
            else:
                raise ValueError("No authentication found in proxy URL")
            
            if ":" in server_part:
                proxy_host, proxy_port = server_part.split(":", 1)
            else:
                raise ValueError("Invalid server format in proxy URL")

            self.logger.info(f"Parsed proxy - Host: {proxy_host}, Port: {proxy_port}, Username: {proxy_username}")

            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    proxy={
                        "server": f"http://{proxy_host}:{proxy_port}",
                        "username": proxy_username,
                        "password": proxy_password
                    },
                    headless=False  # Set to False for debugging purposes
                )
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                    locale="en-US"
                )
                page = await context.new_page()
                await stealth_async(page)  # Apply stealth mode to avoid detection
                
                # Navigate to Twitter login page
                self.logger.info("Navigating to https://x.com/login")
                await page.goto("https://x.com/login", wait_until="domcontentloaded", timeout=120000)

                # Wait for the username input field
                self.logger.info("Waiting for username input field...")
                await page.wait_for_selector('input[name="text"]', timeout=600000)

                # Enter username
                self.logger.info("Entering username...")
                await page.fill('input[name="text"]', username)

                # Click the "Next" button
                self.logger.info("Clicking the 'Next' button...")
                await page.click('button[role="button"]:has-text("Next")', timeout=600000)

                # Wait for the password input field
                self.logger.info("Waiting for password input field...")
                await page.wait_for_selector('input[name="password"]', timeout=600000)

                # Enter password
                self.logger.info("Entering password...")
                await page.fill('input[name="password"]', password)
                await page.click('button[data-testid="LoginForm_Login_Button"]', timeout=600000)

                # Wait for login to complete
                self.logger.info("Waiting for login to complete...")
                await page.wait_for_load_state("networkidle")

                # Check if login was successful
                if "login" in page.url:
                    self.logger.error("Login failed. Please check your credentials or the login page structure.")
                    raise ValueError("Login failed. Unable to proceed.")

                # Extract cookies
                self.logger.info("Extracting cookies...")
                await page.wait_for_timeout(600000)  # Wait for a few seconds to ensure cookies are set
                cookies = await context.cookies()
                with open(output_path, "w") as f:
                    json.dump(cookies, f, indent=4)

                self.logger.info(f"Cookies saved to {output_path}")
                await browser.close()
        except Exception as e:
            self.logger.error(f"Error generating cookies with proxy: {e}")
            raise

    async def _setup_accounts(self, account_proxy: str = None):
        """Setup Twitter accounts for the API.

        Args:
            account_proxy: Optional proxy for this specific account
        """
        try:
            # Generate cookies if not already available
            cookies_path = "cookies.json"
            if not os.path.exists(cookies_path):
                self.logger.info("Generating cookies...")
                await self.generate_cookies_with_proxy(
                    proxy_url=account_proxy,
                    username=self.username,
                    password=self.password,
                    output_path=cookies_path
                )

            # Load cookies
            with open(cookies_path, "r") as f:
                cookies = json.load(f)

            # Add account to the pool with cookies
            await self.api.pool.add_account_with_cookies(
                cookies=cookies,
                proxy=account_proxy
            )
            self.logger.info(f"Added account with cookies and proxy: {account_proxy}")

            # Log in
            await self.api.pool.login_all()
            self.logger.info("Successfully logged in to Twitter account.")
        except Exception as e:
            self.logger.error(f"Error setting up Twitter accounts: {e}")
            raise
        
    async def _fetch_tweets(self, stock: str, limit: int = 100, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[dict]:
        """Fetch tweets mentioning a specific stock within a date range."""
        try:
            # Build the search query
            query = f'"{stock}"'
            if start_date:
                query += f' since:{start_date}'
            if end_date:
                query += f' until:{end_date}'

            self.logger.info(f"Fetching tweets for stock: {stock} with query: {query}")

            # Use the updated search method
            tweets = []
            async for tweet in self.api.search(query, limit=limit):
                tweets.append(self._serialize_tweet(tweet))

            return tweets
        except Exception as e:
            self.logger.error(f"Error fetching tweets for stock {stock}: {e}")
            if self.current_proxy:
                self.proxy_rotator.mark_proxy_failed(self.current_proxy)
                await self.rotate_proxy()
            return []

    async def scrape_twitter_posts(self, stocks: Optional[List[str]] = None, limit: int = 100, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Scrape tweets mentioning stocks within a date range."""
        stocks = stocks if stocks else self.stocks
        if not stocks:
            self.logger.error("No stocks provided for scraping.")
            return

        # Initialize proxies and accounts
        await self.initialize()

        # Create output directory
        os.makedirs(self.workdir, exist_ok=True)

        for stock in stocks:
            try:
                tweets = await self._fetch_tweets(stock, limit=limit, start_date=start_date, end_date=end_date)
                if tweets:
                    # Save tweets to a JSON file
                    file_path = os.path.join(self.workdir, f"{stock}_twitter_posts.json")
                    with open(file_path, "w") as json_file:
                        json.dump(tweets, json_file, indent=4)
                    self.logger.info(f"Successfully scraped {len(tweets)} tweets for stock: {stock}")
                else:
                    self.logger.warning(f"No tweets found for stock: {stock}")
            except Exception as e:
                self.logger.error(f"Error scraping tweets for stock {stock}: {e}")

            await asyncio.sleep(self.delay)  # Rate limiting

        self.logger.info("Twitter scraping completed successfully.")

    def scrape_twitter_posts_sync(self, stocks: Optional[List[str]] = None, limit: int = 100, start_date: Optional[str] = None, end_date: Optional[str] = None, account_proxy: str = None):
        """Scrape tweets mentioning stocks within a date range (sync wrapper).
        
        Args:
            stocks: List of stock symbols to search for. If None, uses stocks from file.
            limit: Maximum number of tweets to collect per stock.
            start_date: Start date for search in 'YYYY-MM-DD' format (inclusive).
            end_date: End date for search in 'YYYY-MM-DD' format (exclusive).
            account_proxy: Optional proxy for account setup (overrides global proxy).
        
        Examples:
            # Search with ThorData proxy (basic)
            scraper.scrape_twitter_posts_sync(
                stocks=["AAPL"], 
                limit=50
            )
            
            # Search with ThorData proxy (US location)
            scraper.scrape_twitter_posts_sync(
                stocks=["TSLA"], 
                limit=100
            )
        """
        asyncio.run(self.scrape_twitter_posts(stocks, limit, start_date, end_date))
    
    def change_proxy(self, new_proxy: str):
        """Change the global proxy for the API at runtime.
        
        Args:
            new_proxy: New proxy URL (e.g., "http://td-customer-username:password@t.pr.thordata.net:9999")
        """
        self.api.proxy = new_proxy
        self.logger.info(f"Changed global proxy to: {new_proxy}")
    
    def remove_proxy(self):
        """Remove the current proxy and use direct connection."""
        self.api.proxy = None
        self.logger.info("Removed proxy - using direct connection")

    def _serialize_tweet(self, tweet) -> dict:
        """Serialize a tweet into a dictionary."""
        return {
            "id": tweet.id,
            "text": tweet.text,
            "author": tweet.user.username,
            "created_at": tweet.created_at.strftime('%Y-%m-%d %H:%M:%S') if tweet.created_at else None,
            "retweets": tweet.retweet_count,
            "likes": tweet.like_count,
            "replies": tweet.reply_count,
            "quote_tweets": tweet.quote_count,
            "permalink": f"https://twitter.com/{tweet.user.username}/status/{tweet.id}"
        }
    
    def get_date_range_helper(self, days_back: int = 7) -> tuple:
        """Helper method to generate date range for the last N days.
        
        Args:
            days_back: Number of days to go back from today (default: 7)
            
        Returns:
            tuple: (start_date, end_date) in 'YYYY-MM-DD' format
        """
        from datetime import timedelta
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')