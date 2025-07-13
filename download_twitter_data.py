import os
from dotenv import load_dotenv
from finagent.downloader.social_media.twitter_stock_scraper import TwitterStockScraper
import asyncio

# Load environment variables
load_dotenv()

# Initialize TwitterStockScraper
scraper = TwitterStockScraper(
    root="/Users/puneetgarg/Documents/Pranav Material/FYP/Indian_FinAgent",
    username=os.getenv("TWITTER_USERNAME"),
    password=os.getenv("TWITTER_PASSWORD"),
    stocks_path="finagent/stocks.txt",
    workdir="social_media_data/uncleaned_data",
    tag="twitter_scraper",
    proxy_username=os.getenv("THORDATA_USERNAME"),
    proxy_password=os.getenv("THORDATA_PASSWORD"),
    proxy_server="fiip79eu.pr.thordata.net:9999"
)

scraper.proxy_rotator.set_location(country="GB")
scraper.proxy_rotator.use_sessions = False  # Use same IP for all requests vs new IP for each request

# Scrape Twitter posts for all stocks in stocks.txt
asyncio.run(scraper.scrape_twitter_posts(
    limit=500,
    start_date="2024-06-06",
    end_date="2025-06-06"
))