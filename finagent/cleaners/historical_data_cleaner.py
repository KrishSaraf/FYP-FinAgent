"""
Historical data cleaner module for processing and cleaning historical market data files.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
from finagent.registry import CLEANER

# Configure logging
output_dir = Path('market_data/cleaned_data')
log_file_path = output_dir / 'historical_data_cleaner.log'
log_file_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@CLEANER.register_module()
class HistoricalDataCleaner:
    """Class for cleaning historical market data files."""
    
    def __init__(self, input_dir: Union[str, Path], output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the historical data cleaner.
        
        Args:
            input_dir: Directory containing the historical data files
            output_dir: Directory to save cleaned data (if None, will use input_dir)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else Path('market_data/cleaned_data')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _parse_historical_data(self, file_path: Path) -> List[Dict]:
        """Parse a historical data file and return a list of dictionaries containing the parsed data."""
        try:
            # Read the file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Remove the 'datasets' header
            content = content.replace('datasets\n', '')
            
            # Replace single quotes with double quotes for valid JSON
            content = content.replace("'", '"')
            
            # Replace 'None' with 'null' for valid JSON
            content = content.replace("'None'", "null")
            
            # Strip any extra whitespace and newlines
            content = content.strip()
            
            # Remove any BOM or special characters at the start
            content = content.lstrip('\ufeff')
            
            # Fix improperly closed arrays (e.g., "values": [...]], "meta": {...})
            content = content.replace("]], \"meta\":", "], \"meta\":")
            
            # Validate JSON structure
            if content.count('{') != content.count('}'):
                logger.error("Mismatched curly braces in JSON content.")
                return []
            if content.count('[') != content.count(']'):
                logger.error("Mismatched square brackets in JSON content.")
                return []
            
            # Find the first '[' and last ']' to extract just the JSON array
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                content = content[start_idx:end_idx]
            else:
                logger.warning(f"Malformed JSON structure in file: {file_path}")
                return []
            
            # Parse the JSON string
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error in file {file_path}: {str(e)}")
                return []
            
            # Validate the parsed data
            if not isinstance(data, list):
                logger.warning(f"Unexpected JSON structure in file: {file_path}")
                return []
            
            return data
        
        except Exception as e:
            logger.error(f"Error parsing historical data file {file_path}: {str(e)}")
            return []
            
    def _convert_to_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """Convert the parsed data into a pandas DataFrame.
        
        Args:
            data (List[Dict]): List of dictionaries containing the parsed data
            
        Returns:
            pd.DataFrame: DataFrame containing the cleaned data
        """
        try:
            # Initialize empty DataFrame
            df = pd.DataFrame()
            
            if not data:
                logger.warning("No data to convert to DataFrame")
                return pd.DataFrame()
            
            # Process each metric
            for metric in data:
                try:
                    metric_name = metric.get('metric', 'Unknown')
                    values = metric.get('values', [])
                    
                    if not values:
                        logger.warning(f"No values found for metric {metric_name}")
                        continue
                    
                    # Convert values to DataFrame
                    if metric_name == 'Volume':
                        # Handle Volume data structure with delivery percentages
                        dates = [v[0] for v in values if len(v) > 0]
                        volumes = [v[1] if len(v) > 1 else np.nan for v in values]
                        deliveries = [v[2].get('delivery') if len(v) > 2 and isinstance(v[2], dict) else np.nan for v in values]
                        
                        temp_df = pd.DataFrame({
                            'date': dates,
                            'volume': volumes,
                            'delivery_percentage': deliveries
                        })
                        temp_df.set_index('date', inplace=True)
                        
                        # Add to main DataFrame
                        df[f'{metric_name}'] = temp_df['volume']
                        df[f'{metric_name}_delivery'] = temp_df['delivery_percentage']
                    else:
                        # Handle other metrics
                        temp_df = pd.DataFrame(values, columns=['date', 'value'])
                        temp_df.set_index('date', inplace=True)
                        df[metric_name] = pd.to_numeric(temp_df['value'], errors='coerce')
                except Exception as e:
                    logger.error(f"Error processing metric {metric_name}: {str(e)}")
                    continue
            
            # Sort by date
            if not df.empty:
                df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error converting data to DataFrame: {str(e)}")
            return pd.DataFrame()
    
    def clean_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Clean a single historical data file.
        
        Args:
            file_path: Path to the historical data file
            
        Returns:
            DataFrame containing the cleaned data
        """
        file_path = Path(file_path)
        logger.info(f"Cleaning file: {file_path}")
        
        # Get stock symbol from the parent directory name
        stock_symbol = file_path.parent.name
        
        # Create stock-specific output directory
        stock_output_dir = self.output_dir / stock_symbol / "historical_data"
        stock_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse the data
        data = self._parse_historical_data(file_path)
        
        # Convert to DataFrame
        df = self._convert_to_dataframe(data)
        
        # Save the cleaned data
        if not df.empty:
            output_path = stock_output_dir / "historical_data_cleaned.csv"
            df.to_csv(output_path)
            logger.info(f"Saved cleaned data to: {output_path}")
        else:
            logger.warning(f"No cleaned data generated for file: {file_path}")
        
        return df
    
    def clean_all_files(self, pattern: str = "historical_data.csv") -> Dict[str, pd.DataFrame]:
        """
        Clean all historical data files in the input directory.
        
        Args:
            pattern: File pattern to match (default: "historical_data.csv")
            
        Returns:
            Dictionary mapping file names to their cleaned DataFrames
        """
        cleaned_data = {}
        
        # Find all matching files
        for file_path in self.input_dir.rglob(pattern):
            try:
                df = self.clean_file(file_path)
                cleaned_data[file_path.name] = df
            except Exception as e:
                logger.error(f"Error cleaning file {file_path}: {str(e)}")
                continue
        
        return cleaned_data
    
    def test_data_correction(self, file_path: Union[str, Path]) -> None:
        """
        Test if the entries in the datasets column are being corrected properly.
        
        Args:
            file_path: Path to the historical data file to test
        """
        file_path = Path(file_path)
        logger.info(f"Testing data correction for file: {file_path}")
        
        try:
            # Read the file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Original content before corrections
            logger.info("Original content (character 6700 onwards):")
            logger.info(content[6700:6800])  # Log content around the problematic area
            
            # Apply corrections
            corrected_content = content.replace('datasets\n', '')
            corrected_content = corrected_content.replace("'", '"')
            corrected_content = corrected_content.replace("'None'", "null")
            corrected_content = corrected_content.strip()
            corrected_content = corrected_content.lstrip('\ufeff')
            
            # Log corrected content
            logger.info("Corrected content (character 6700 onwards):")
            logger.info(corrected_content[6700:6800])  # Log corrected content around the problematic area
            
            # Validate JSON parsing
            start_idx = corrected_content.find('[')
            end_idx = corrected_content.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                corrected_content = corrected_content[start_idx:end_idx]
            else:
                logger.warning(f"Malformed JSON structure in file: {file_path}")
                return
            
            # Parse the JSON string
            try:
                data = json.loads(corrected_content)
                logger.info("JSON parsing successful. Sample data:")
                logger.info(data[:2])  # Log the first two entries for verification
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error in file {file_path}: {str(e)}")
                return
            
            # Validate the parsed data structure
            if isinstance(data, list) and all(isinstance(entry, dict) for entry in data):
                logger.info("Data structure validation successful.")
            else:
                logger.warning("Unexpected data structure after corrections.")
        
        except Exception as e:
            logger.error(f"Error testing data correction for file {file_path}: {str(e)}")