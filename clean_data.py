import os
import logging
from pathlib import Path
from finagent.cleaners.price_cleaner import PriceCleaner
from finagent.cleaners.stock_details_cleaner import StockDetailsCleaner
from finagent.cleaners.historical_stats_cleaner import HistoricalStatsCleaner
from finagent.cleaners.stock_forecasts_cleaner import StockForecastsCleaner

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
            dataframes = cleaner.parse_stock_forecasts_json(str(json_file_path))

            # Create the output directory for the stock
            stock_output_dir = output_dir / stock_name
            stock_output_dir.mkdir(parents=True, exist_ok=True)

            # Save each DataFrame as a CSV
            for label, df in dataframes.items():
                output_csv_path = stock_output_dir / f"{label}.csv"
                df.to_csv(output_csv_path, index=False)
                logger.info(f"Saved cleaned stock forecasts to: {output_csv_path}")


def main():
    """
    Main function to clean historical data, stock details, historical stats and stock forecasts JSON files.
    """
    clean_historical_data()
    clean_stock_details()
    clean_historical_stats()
    clean_stock_forecasts()

if __name__ == "__main__":
    main()