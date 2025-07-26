import os
from dotenv import load_dotenv
from finagent.downloader.social_media.twitter_stock_scraper import TwitterStockScraper

# Load environment variables
load_dotenv()

# Initialize TwitterStockScraper
scraper = TwitterStockScraper(
    root="/Users/puneetgarg/Documents/Pranav Material/FYP/Indian_FinAgent",
    api_key=os.getenv("TWITTER_API_KEY"),
    stocks_path="finagent/stocks.txt",
    handles_path="finagent/stock_handles.json",
    workdir="social_media_data/uncleaned_data",
    tag="twitter_scraper"
)

# Specify the date range for scraping
start_date = "2024-06-06T00:00:00Z"  # Start date in ISO 8601 format
end_date = "2025-06-06T23:59:59Z"    # End date in ISO 8601 format

# Scrape Twitter posts for all stocks in stocks.txt
scraper.scrape_twitter_posts(
    limit=500,
    start_date=start_date,
    end_date=end_date
)