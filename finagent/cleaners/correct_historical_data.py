"""
Script to correct historical market data CSV files.
"""

import logging
from pathlib import Path
from finagent.cleaners.csv_data_corrector import CSVDataCorrector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to correct historical data CSV files."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Define input and output directories
    input_dir = project_root / "market_data" / "indian_market"
    output_dir = project_root / "market_data" / "corrected_data"
    
    # Create corrector instance
    corrector = CSVDataCorrector(input_dir, output_dir)
    
    # Correct all files
    corrector.correct_all_files()

if __name__ == "__main__":
    main()