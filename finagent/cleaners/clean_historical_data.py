"""
Script to clean historical market data files.
"""

import logging
from pathlib import Path
from finagent.cleaners.historical_data_cleaner import HistoricalDataCleaner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to clean historical data files."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Define input and output directories
    input_dir = project_root / "market_data" / "indian_market"
    output_dir = project_root / "market_data" / "indian_market" / "cleaned_data"
    
    # Create cleaner instance
    cleaner = HistoricalDataCleaner(input_dir, output_dir)
    
    # Clean all historical data files
    logger.info("Starting to clean historical data files...")
    cleaned_data = cleaner.clean_all_files()
    
    # Log results
    logger.info(f"Successfully cleaned {len(cleaned_data)} files")
    for file_name in cleaned_data:
        logger.info(f"Cleaned: {file_name}")

if __name__ == "__main__":
    main() 