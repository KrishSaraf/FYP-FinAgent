import os
import logging
from pathlib import Path
from finagent.cleaners.price_cleaner import PriceCleaner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to clean historical data JSON files and save them as CSVs.
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
            logger.info(f"Processing stock: {stock_name}")

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
                logger.info(f"Saved cleaned data to: {output_csv_path}")

if __name__ == "__main__":
    main()