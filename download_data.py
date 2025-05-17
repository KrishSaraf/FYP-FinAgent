import os
from dotenv import load_dotenv
from finagent.downloader.prices import IndianMarketDownloader

# Load environment variables
load_dotenv()

# Initialize downloader
downloader = IndianMarketDownloader(
    root="/Users/puneetgarg/Documents/Pranav Material/FYP/Indian_FinAgent",
    token=os.getenv("INDIAN_API_KEY"),
    stocks_path="finagent/stocks.txt",
    workdir="market_data",
    tag="indian_market"
)

# Download data
downloader.download()