import os
import logging
from pathlib import Path
from finagent.cleaners.price_cleaner import PriceCleaner
from finagent.cleaners.stock_details_cleaner import StockDetailsCleaner
from finagent.cleaners.historical_stats_cleaner import HistoricalStatsCleaner
from finagent.cleaners.stock_forecasts_cleaner import StockForecastsCleaner
from finagent.cleaners.reddit_data_cleaner import RedditDataCleaner
import json
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def clean_historical_data():
    """
    Function to clean historical data JSON files and save them as CSVs using PriceCleaner.
    """
    # Define input and output directories
    input_dir = Path("market_data/indian_market")
    output_dir = Path("market_data/cleaned_data")

    # Initialize the PriceCleaner
    cleaner = PriceCleaner()

    # Iterate through each stock directory in the input directory
    for stock_dir in input_dir.iterdir():
        if stock_dir.is_dir():  # Ensure it's a directory
            stock_name = stock_dir.name
            logger.info(f"Processing historical data for stock: {stock_name}")

            # Path to the historical_data.json file
            json_file_path = stock_dir / "historical_data.json"
            if not json_file_path.exists():
                logger.warning(f"No historical_data.json found for {stock_name}, skipping.")
                continue

            # Parse the JSON file using PriceCleaner
            dataframes = cleaner.parse_price_json(str(json_file_path))

            # Create the output directory for the stock
            stock_output_dir = output_dir / stock_name
            stock_output_dir.mkdir(parents=True, exist_ok=True)

            # Save each DataFrame as a CSV
            for label, df in dataframes.items():
                output_csv_path = stock_output_dir / f"{label}.csv"
                df.to_csv(output_csv_path, index=False)
                logger.info(f"Saved cleaned historical data to: {output_csv_path}")

def clean_stock_details():
    """
    Function to clean stock details JSON files and save them as CSVs using StockDetailsCleaner.
    """
    # Define input and output directories
    input_dir = Path("market_data/indian_market")
    output_dir = Path("market_data/cleaned_data")

    # Initialize the StockDetailsCleaner
    cleaner = StockDetailsCleaner()

    # Iterate through each stock directory in the input directory
    for stock_dir in input_dir.iterdir():
        if stock_dir.is_dir():  # Ensure it's a directory
            stock_name = stock_dir.name
            logger.info(f"Processing stock details for stock: {stock_name}")

            # Path to the stock_details.json file
            json_file_path = stock_dir / "stock_details.json"
            if not json_file_path.exists():
                logger.warning(f"No stock_details.json found for {stock_name}, skipping.")
                continue

            # Parse the JSON file using StockDetailsCleaner
            dataframes = cleaner.parse_stock_details_json(str(json_file_path))

            # Create the output directory for the stock
            stock_output_dir = output_dir / stock_name
            stock_output_dir.mkdir(parents=True, exist_ok=True)

            # Save each DataFrame as a CSV
            for label, df in dataframes.items():
                output_csv_path = stock_output_dir / f"{label}.csv"
                df.to_csv(output_csv_path, index=False)
                logger.info(f"Saved cleaned stock details to: {output_csv_path}")

def clean_historical_stats():
    """
    Function to clean historical stats JSON files and save them as CSVs using HistoricalStatsCleaner.
    """
    # Define input and output directories
    input_dir = Path("market_data/indian_market")
    output_dir = Path("market_data/cleaned_data")

    # Initialize the HistoricalStatsCleaner
    cleaner = HistoricalStatsCleaner()

    # Iterate through each stock directory in the input directory
    for stock_dir in input_dir.iterdir():
        if stock_dir.is_dir():  # Ensure it's a directory
            stock_name = stock_dir.name
            logger.info(f"Processing historical stats for stock: {stock_name}")

            # Path to the historical_stats.json file
            json_file_path = stock_dir / "historical_stats.json"
            if not json_file_path.exists():
                logger.warning(f"No historical_stats.json found for {stock_name}, skipping.")
                continue

            # Parse the JSON file using HistoricalStatsCleaner
            df = cleaner.parse_historical_stats_json(str(json_file_path))

            # Create the output directory for the stock
            stock_output_dir = output_dir / stock_name
            stock_output_dir.mkdir(parents=True, exist_ok=True)

            # Save the DataFrame as a CSV
            output_csv_path = stock_output_dir / "historical_stats.csv"
            df.to_csv(output_csv_path, index=False)
            logger.info(f"Saved cleaned historical stats to: {output_csv_path}")

def clean_stock_forecasts():
    """
    Function to clean stock forecasts JSON files and save them as CSVs using StockForecastsCleaner.
    """
    # Define input and output directories
    input_dir = Path("market_data/indian_market")
    output_dir = Path("market_data/cleaned_data")

    # Initialize the StockForecastsCleaner
    cleaner = StockForecastsCleaner()

    # Iterate through each stock directory in the input directory
    for stock_dir in input_dir.iterdir():
        if stock_dir.is_dir():  # Ensure it's a directory
            stock_name = stock_dir.name
            logger.info(f"Processing stock forecasts for stock: {stock_name}")

            # Path to the stock_forecasts.json file
            json_file_path = stock_dir / "stock_forecasts.json"
            if not json_file_path.exists():
                logger.warning(f"No stock_forecasts.json found for {stock_name}, skipping.")
                continue

            # Parse the JSON file using StockForecastsCleaner
            estimates_actuals_df, snapshots_df = cleaner.parse_stock_forecasts_json(str(json_file_path))

            # Create the output directory for the stock
            stock_output_dir = output_dir / stock_name
            stock_output_dir.mkdir(parents=True, exist_ok=True)

            # Save the estimates and actuals DataFrame as a CSV
            estimates_actuals_csv_path = stock_output_dir / "estimates_actuals.csv"
            estimates_actuals_df.to_csv(estimates_actuals_csv_path, index=False)
            logger.info(f"Saved cleaned estimates and actuals to: {estimates_actuals_csv_path}")

            # Save the snapshots DataFrame as a CSV
            snapshots_csv_path = stock_output_dir / "snapshots.csv"
            snapshots_df.to_csv(snapshots_csv_path, index=False)
            logger.info(f"Saved cleaned snapshots to: {snapshots_csv_path}")

def clean_52week_high_low():
    """
    Function to clean the 52week_high_low.json file and save the data as CSVs.
    """
    # Define the input file path and output directory
    input_file = Path("market_data/indian_market/market_data/52week_high_low.json")
    output_dir = Path("market_data/cleaned_data/market_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if the input file exists
    if not input_file.exists():
        logger.error(f"52week_high_low.json file not found at {input_file}")
        return

    try:
        # Load the JSON file
        logger.info(f"Parsing JSON file: {input_file}")
        with open(input_file, 'r') as file:
            data = json.load(file)

        # Initialize dictionaries to store DataFrames
        dataframes = {}

        # Process BSE data
        if "BSE_52WeekHighLow" in data:
            bse_data = data["BSE_52WeekHighLow"]
            high_df = pd.DataFrame(bse_data.get("high52Week", []))
            low_df = pd.DataFrame(bse_data.get("low52Week", []))
            dataframes["BSE_high"] = high_df
            dataframes["BSE_low"] = low_df

        # Process NSE data
        if "NSE_52WeekHighLow" in data:
            nse_data = data["NSE_52WeekHighLow"]
            high_df = pd.DataFrame(nse_data.get("high52Week", []))
            low_df = pd.DataFrame(nse_data.get("low52Week", []))
            dataframes["NSE_high"] = high_df
            dataframes["NSE_low"] = low_df

        # Save each DataFrame as a CSV
        for label, df in dataframes.items():
            output_csv_path = output_dir / f"{label}.csv"
            df.to_csv(output_csv_path, index=False)
            logger.info(f"Saved cleaned 52-week high/low data to: {output_csv_path}")

    except Exception as e:
        logger.error(f"Error parsing 52week_high_low.json file: {e}")

def clean_commodities():
    """
    Function to clean the commodities.json file and save the data as a CSV.
    """
    # Define the input file path and output directory
    input_file = Path("market_data/indian_market/market_data/commodities.json")
    output_dir = Path("market_data/cleaned_data/market_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if the input file exists
    if not input_file.exists():
        logger.error(f"commodities.json file not found at {input_file}")
        return

    try:
        # Load the JSON file
        logger.info(f"Parsing JSON file: {input_file}")
        with open(input_file, 'r') as file:
            data = json.load(file)

        # Convert the JSON data into a DataFrame
        df = pd.DataFrame(data)

        # Save the DataFrame as a CSV
        output_csv_path = output_dir / "commodities.csv"
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved cleaned commodities data to: {output_csv_path}")

    except Exception as e:
        logger.error(f"Error parsing commodities.json file: {e}")

def clean_ipo():
    """
    Function to clean the ipo.json file and save the data as separate CSVs for upcoming, listed, active, and closed IPOs.
    """
    # Define the input file path and output directory
    input_file = Path("market_data/indian_market/market_data/ipo.json")
    output_dir = Path("market_data/cleaned_data/market_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if the input file exists
    if not input_file.exists():
        logger.error(f"ipo.json file not found at {input_file}")
        return

    try:
        # Load the JSON file
        logger.info(f"Parsing JSON file: {input_file}")
        with open(input_file, 'r') as file:
            data = json.load(file)

        # Process each section of the IPO data
        for section in ["upcoming", "listed", "active", "closed"]:
            if section in data:
                df = pd.DataFrame(data[section])
                # Handle empty rows (if any)
                df.fillna("", inplace=True)

                # Save the DataFrame as a CSV
                output_csv_path = output_dir / f"{section}.csv"
                df.to_csv(output_csv_path, index=False)
                logger.info(f"Saved cleaned {section} IPO data to: {output_csv_path}")

    except Exception as e:
        logger.error(f"Error parsing ipo.json file: {e}")

def clean_most_active():
    """
    Function to clean the most_active_bse.json and most_active_nse.json files and save the data as separate CSVs.
    """
    # Define the input file paths and output directory
    bse_file = Path("market_data/indian_market/most_active_bse.json")
    nse_file = Path("market_data/indian_market/most_active_nse.json")
    output_dir = Path("market_data/cleaned_data/market_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process most_active_bse.json
    if not bse_file.exists():
        logger.warning(f"most_active_bse.json file not found at {bse_file}")
    else:
        try:
            logger.info(f"Parsing JSON file: {bse_file}")
            with open(bse_file, 'r') as file:
                bse_data = json.load(file)

            # Convert the JSON data into a DataFrame
            bse_df = pd.DataFrame(bse_data)

            # Save the DataFrame as a CSV
            bse_csv_path = output_dir / "most_active_bse.csv"
            bse_df.to_csv(bse_csv_path, index=False)
            logger.info(f"Saved cleaned most active BSE data to: {bse_csv_path}")
        except Exception as e:
            logger.error(f"Error parsing most_active_bse.json file: {e}")

    # Process most_active_nse.json
    if not nse_file.exists():
        logger.warning(f"most_active_nse.json file not found at {nse_file}")
    else:
        try:
            logger.info(f"Parsing JSON file: {nse_file}")
            with open(nse_file, 'r') as file:
                nse_data = json.load(file)

            # Convert the JSON data into a DataFrame
            nse_df = pd.DataFrame(nse_data)

            # Save the DataFrame as a CSV
            nse_csv_path = output_dir / "most_active_nse.csv"
            nse_df.to_csv(nse_csv_path, index=False)
            logger.info(f"Saved cleaned most active NSE data to: {nse_csv_path}")
        except Exception as e:
            logger.error(f"Error parsing most_active_nse.json file: {e}")

def clean_mutual_funds():
    """
    Function to clean the mutual_funds.json file and save the data as separate CSVs for each category and subcategory.
    """
    # Define the input file path and output directory
    input_file = Path("market_data/indian_market/market_data/mutual_funds.json")
    output_dir = Path("market_data/cleaned_data/market_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if the input file exists
    if not input_file.exists():
        logger.error(f"mutual_funds.json file not found at {input_file}")
        return

    try:
        # Load the JSON file
        logger.info(f"Parsing JSON file: {input_file}")
        with open(input_file, 'r') as file:
            data = json.load(file)

        # Iterate through each category in the JSON data
        for category, subcategories in data.items():
            if isinstance(subcategories, dict):
                for subcategory, funds in subcategories.items():
                    # Convert the list of funds into a DataFrame
                    df = pd.DataFrame(funds)
                    # Handle empty rows (if any)
                    df.fillna("", inplace=True)

                    # Save the DataFrame as a CSV
                    output_csv_path = output_dir / f"{category}_{subcategory.replace(' ', '_')}.csv"
                    df.to_csv(output_csv_path, index=False)
                    logger.info(f"Saved cleaned mutual funds data for {category} - {subcategory} to: {output_csv_path}")

    except Exception as e:
        logger.error(f"Error parsing mutual_funds.json file: {e}")

def clean_price_shockers():
    """
    Function to clean the price_shockers.json file and save the data as separate CSVs for BSE and NSE price shockers.
    """
    # Define the input file path and output directory
    input_file = Path("market_data/indian_market/market_data/price_shockers.json")
    output_dir = Path("market_data/cleaned_data/market_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if the input file exists
    if not input_file.exists():
        logger.error(f"price_shockers.json file not found at {input_file}")
        return

    try:
        # Load the JSON file
        logger.info(f"Parsing JSON file: {input_file}")
        with open(input_file, 'r') as file:
            data = json.load(file)

        # Process BSE price shockers
        if "BSE_PriceShocker" in data:
            bse_data = data["BSE_PriceShocker"]
            bse_df = pd.DataFrame(bse_data)
            if not bse_df.empty:
                bse_csv_path = output_dir / "bse_price_shockers.csv"
                bse_df.to_csv(bse_csv_path, index=False)
                logger.info(f"Saved cleaned BSE price shockers data to: {bse_csv_path}")
            else:
                logger.warning("No data found for BSE price shockers.")

        # Process NSE price shockers
        if "NSE_PriceShocker" in data:
            nse_data = data["NSE_PriceShocker"]
            nse_df = pd.DataFrame(nse_data)
            if not nse_df.empty:
                nse_csv_path = output_dir / "nse_price_shockers.csv"
                nse_df.to_csv(nse_csv_path, index=False)
                logger.info(f"Saved cleaned NSE price shockers data to: {nse_csv_path}")
            else:
                logger.warning("No data found for NSE price shockers.")

    except Exception as e:
        logger.error(f"Error parsing price_shockers.json file: {e}")

def clean_trending():
    """
    Function to clean the trending.json file and save the data as separate CSVs for top gainers and top losers.
    """
    # Define the input file path and output directory
    input_file = Path("market_data/indian_market/market_data/trending.json")
    output_dir = Path("market_data/cleaned_data/market_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if the input file exists
    if not input_file.exists():
        logger.error(f"trending.json file not found at {input_file}")
        return

    try:
        # Load the JSON file
        logger.info(f"Parsing JSON file: {input_file}")
        with open(input_file, 'r') as file:
            data = json.load(file)

        # Process top gainers
        if "trending_stocks" in data and "top_gainers" in data["trending_stocks"]:
            top_gainers_data = data["trending_stocks"]["top_gainers"]
            top_gainers_df = pd.DataFrame(top_gainers_data)
            if not top_gainers_df.empty:
                top_gainers_csv_path = output_dir / "top_gainers.csv"
                top_gainers_df.to_csv(top_gainers_csv_path, index=False)
                logger.info(f"Saved cleaned top gainers data to: {top_gainers_csv_path}")
            else:
                logger.warning("No data found for top gainers.")

        # Process top losers
        if "trending_stocks" in data and "top_losers" in data["trending_stocks"]:
            top_losers_data = data["trending_stocks"]["top_losers"]
            top_losers_df = pd.DataFrame(top_losers_data)
            if not top_losers_df.empty:
                top_losers_csv_path = output_dir / "top_losers.csv"
                top_losers_df.to_csv(top_losers_csv_path, index=False)
                logger.info(f"Saved cleaned top losers data to: {top_losers_csv_path}")
            else:
                logger.warning("No data found for top losers.")

    except Exception as e:
        logger.error(f"Error parsing trending.json file: {e}")

def clean_reddit_data():
    """
    Function to clean Reddit JSON files and save them as CSVs using RedditDataCleaner.
    """
    # Define input and output directories
    input_dir = Path("social_media_data/uncleaned_data/reddit_scraper")
    output_dir = Path("social_media_data/cleaned_data")

    # Initialize the RedditDataCleaner
    cleaner = RedditDataCleaner()

    # Iterate through all JSON files in the input directory
    for json_file_path in input_dir.glob("*_reddit_posts.json"):
        # Extract the stock name from the file name
        stock_name = json_file_path.stem.replace("_reddit_posts", "")
        logger.info(f"Processing Reddit data for stock: {stock_name}")

        # Parse the JSON file using RedditDataCleaner
        df = cleaner.parse_reddit_json(str(json_file_path))
        if df is None or df.empty:
            logger.warning(f"No data found in {json_file_path}, skipping.")
            continue

        # Create the output directory for the stock
        stock_output_dir = output_dir / stock_name
        stock_output_dir.mkdir(parents=True, exist_ok=True)

        # Save the DataFrame as a CSV
        output_csv_path = stock_output_dir / "reddit.csv"
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved cleaned Reddit data to: {output_csv_path}")

def main():
    """
    Main function to parse all JSON files and clean the data.
    """
    clean_historical_data()
    clean_stock_details()
    clean_historical_stats()
    clean_stock_forecasts()
    clean_52week_high_low()
    clean_commodities()
    clean_ipo()
    clean_most_active()
    clean_mutual_funds()
    clean_price_shockers()
    clean_trending()
    clean_reddit_data()

if __name__ == "__main__":
    main()