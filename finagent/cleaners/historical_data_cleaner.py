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

# Create the output directory first
output_dir = Path('market_data/cleaned_data')
output_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
log_file_path = output_dir / 'historical_data_cleaner.log'
log_file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

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
        """Parse a historical data file and return a list of dictionaries containing the parsed data.
        
        Args:
            file_path (Path): Path to the historical data file
            
        Returns:
            List[Dict]: List of dictionaries containing the parsed data
        """
        try:
            # Read the file content
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Remove the 'datasets' header
            content = content.replace('datasets\n', '')
            
            # Replace single quotes with double quotes for valid JSON
            content = content.replace("'", '"')
            
            # Strip any extra whitespace and newlines
            content = content.strip()
            
            # Remove any BOM or special characters at the start
            content = content.lstrip('\ufeff')
            
            # Find the first '[' and last ']' to extract just the JSON array
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                content = content[start_idx:end_idx]
            
            # Parse the JSON string twice due to double-encoding
            try:
                # First try parsing as is
                data = json.loads(content)
            except json.JSONDecodeError as e:
                try:
                    # If that fails, try parsing the string representation
                    # First, remove any extra quotes at the start and end
                    content = content.strip('"')
                    data = json.loads(json.loads(content))
                except json.JSONDecodeError as e:
                    # Try to recover partial data
                    error_pos = e.pos
                    error_line = e.lineno
                    error_col = e.colno
                    
                    logger.error(f"JSON parsing error at line {error_line}, column {error_col} (char {error_pos})")
                    
                    # Find the last complete object before the error
                    last_complete = content.rfind('}', 0, error_pos)
                    if last_complete != -1:
                        # Extract the valid portion of the JSON
                        valid_content = content[:last_complete + 1] + ']'
                        try:
                            # Try to parse the valid portion
                            data = json.loads(valid_content)
                            logger.info(f"Recovered partial data up to char {last_complete}")
                        except json.JSONDecodeError:
                            try:
                                # If that fails, try parsing the string representation of the valid portion
                                data = json.loads(json.loads(valid_content))
                                logger.info(f"Recovered partial data up to char {last_complete}")
                            except json.JSONDecodeError:
                                logger.error("Could not recover partial data")
                                return []
                    else:
                        logger.error("Could not find complete object before error")
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
        stock_output_dir = self.output_dir / stock_symbol
        stock_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse the data
        data = self._parse_historical_data(file_path)
        
        # Convert to DataFrame
        df = self._convert_to_dataframe(data)
        
        # Save the cleaned data
        output_path = stock_output_dir / "historical_data_cleaned.csv"
        df.to_csv(output_path)
        logger.info(f"Saved cleaned data to: {output_path}")
        
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