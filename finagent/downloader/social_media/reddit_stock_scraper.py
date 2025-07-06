import os
import time
import json
import logging
from typing import List, Optional
from praw import Reddit
from praw.models import Submission
from dotenv import load_dotenv
from datetime import datetime, timezone

class RedditStockScraper:
    def __init__(self,
                 root: str = "",
                 client_id: str = None,
                 client_secret: str = None,
                 user_agent: str = None,
                 delay: int = 1,
                 stocks_path: str = None,
                 workdir: str = "",
                 tag: str = "",
                 **kwargs):
        self.root = root
        load_dotenv(os.path.join(root, "apikeys.env"))
        self.client_id = client_id if client_id else os.environ.get("REDDIT_CLIENT_ID")
        self.client_secret = client_secret if client_secret else os.environ.get("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent if user_agent else os.environ.get("REDDIT_USER_AGENT")

        if not self.client_id or not self.client_secret:
            raise ValueError("Reddit API credentials not provided. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in apikeys.env.")

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

        # Initialize Reddit API client
        self.reddit = Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )

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

    def _fetch_posts(self, stock: str, subreddits: List[str], limit: int = 100, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Submission]:
        """Fetch posts mentioning a specific stock within a date range from specific subreddits."""
        try:
            filtered_posts = []
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()) if start_date else None
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) if end_date else None

            for subreddit in subreddits:
                query = f'"{stock}"'
                self.logger.info(f"Fetching posts for stock: {stock} in subreddit: {subreddit}")
                posts = self.reddit.subreddit(subreddit).search(query, limit=limit)

                for post in posts:
                    if start_timestamp and post.created_utc < start_timestamp:
                        continue
                    if end_timestamp and post.created_utc > end_timestamp:
                        continue
                    filtered_posts.append(post)

            return filtered_posts
        except Exception as e:
            self.logger.error(f"Error fetching posts for stock {stock}: {e}")
            return []

    def scrape_reddit_posts(self, stocks: Optional[List[str]] = None, subreddits: List[str] = None, limit: int = 100, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Scrape Reddit posts mentioning stocks within a date range from specific subreddits."""
        stocks = stocks if stocks else self.stocks
        subreddits = subreddits if subreddits else []
        if not stocks:
            self.logger.error("No stocks provided for scraping.")
            return
        if not subreddits:
            self.logger.error("No subreddits provided for scraping.")
            return

        # Create output directory
        os.makedirs(self.workdir, exist_ok=True)

        for stock in stocks:
            try:
                posts = self._fetch_posts(stock, subreddits=subreddits, limit=limit, start_date=start_date, end_date=end_date)
                if posts:
                    # Save posts to a JSON file
                    file_path = os.path.join(self.workdir, f"{stock}_reddit_posts.json")
                    with open(file_path, "w") as json_file:
                        json.dump([self._serialize_post(post) for post in posts], json_file, indent=4)
                    self.logger.info(f"Successfully scraped posts for stock: {stock}")
                else:
                    self.logger.warning(f"No posts found for stock: {stock}")
            except Exception as e:
                self.logger.error(f"Error scraping posts for stock {stock}: {e}")

            time.sleep(self.delay)  # Rate limiting

        self.logger.info("Reddit scraping completed successfully.")

    def _serialize_post(self, post: Submission) -> dict:
        """Serialize a Reddit post into a dictionary, including comments."""
        return {
            "id": post.id,
            "title": post.title,
            "selftext": post.selftext,
            "author": str(post.author),
            "created_utc": post.created_utc,
            "created_timestamp": datetime.fromtimestamp(post.created_utc, timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            "score": post.score,
            "num_comments": post.num_comments,
            "permalink": f"https://reddit.com{post.permalink}",
            "subreddit": post.subreddit.display_name,
            "comments": self._fetch_comments(post)
        }

    def _fetch_comments(self, post: Submission, top_n: int = 5) -> List[dict]:
        """Fetch and serialize the top N comments for a Reddit post."""
        try:
            post.comments.replace_more(limit=None)  # Load all comments
            all_comments = post.comments.list()

            # Sort comments by score in descending order
            sorted_comments = sorted(all_comments, key=lambda c: c.score, reverse=True)

            # Limit to the top N comments
            top_comments = sorted_comments[:top_n]

            return [
                {
                    "id": comment.id,
                    "author": str(comment.author),
                    "body": comment.body,
                    "created_utc": comment.created_utc,
                    "created_timestamp": datetime.fromtimestamp(comment.created_utc, timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "score": comment.score
                }
                for comment in top_comments
            ]
        except Exception as e:
            self.logger.error(f"Error fetching comments for post {post.id}: {e}")
            return []