import os
from dotenv import load_dotenv
from finagent.downloader.social_media.reddit_stock_scraper import RedditStockScraper

# Load environment variables
load_dotenv()

# List of subreddits to scrape
subreddits = [
    "IndianStockMarket",
    "IndiaInvestments",
    "StockMarketIndia",
    "IndianStreetBets",
    "trading",
    "stockmarket",
    "investing",
    "stocks",
]

# Initialize RedditStockScraper
scraper = RedditStockScraper(
    root="/Users/puneetgarg/Documents/Pranav Material/FYP/Indian_FinAgent",
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT", "RedditStockScraper"),
    stocks_path="finagent/stocks.txt",
    workdir="social_media_data/uncleaned_data",
    tag="reddit_scraper"
)

# Scrape Reddit posts for all stocks in stocks.txt
scraper.scrape_reddit_posts(subreddits=subreddits, limit=100, start_date="2024-06-06", end_date="2025-06-06")