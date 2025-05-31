"""
CSV data corrector module for fixing malformed historical market data files.
"""

import json
from pathlib import Path
from typing import Union
import logging

# Configure logging
output_dir = Path('market_data/corrected_data')
log_file_path = output_dir / 'csv_data_corrector.log'
log_file_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Fix the format string
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class CSVDataCorrector:
    """Class for correcting malformed historical market data CSV files."""
    
    def __init__(self, input_dir: Union[str, Path], output_dir: Union[str, Path]):
        """
        Initialize the CSV data corrector.
        
        Args:
            input_dir: Directory containing the historical data files
            output_dir: Directory to save corrected data
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _correct_csv_content(self, content: str) -> str:
        """Correct the content of a CSV file."""
        try:
            # Remove the 'datasets' header
            content = content.replace('datasets\n', '')

            # Replace single quotes with double quotes
            content = content.replace("'", '"')

            # Replace 'None' with 'null' for valid JSON
            content = content.replace("'None'", "null")

            # Replace Python boolean values with JSON boolean values
            content = content.replace(" True", " true").replace(" False", " false")

            # Strip any extra whitespace and newlines
            content = content.strip()

            # Remove any BOM or special characters at the start
            content = content.lstrip('\ufeff')

            # Fix improperly closed arrays
            content = content.replace("]], \"meta\":", "], \"meta\":")

            # Validate JSON structure
            try:
                json.loads(content)  # Ensure the content is valid JSON
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}. Content: {content[:100]}")  # Log the first 100 characters
                return ""

            return content

        except Exception as e:
            logger.error(f"Error correcting CSV content: {str(e)}")
            return ""

    def _fix_mismatched_brackets(self, content: str) -> str:
        """Fix mismatched square brackets in the content.
        
        Args:
            content (str): The raw content of the CSV file
            
        Returns:
            str: The content with corrected square brackets
        """
        stack = []
        corrected_content = []
        unmatched_closing_brackets = 0

        for i, char in enumerate(content):
            if char == '[':
                stack.append(i)  # Track the position of the opening bracket
                corrected_content.append(char)
            elif char == ']':
                if stack:
                    stack.pop()  # Match found, pop from stack
                    corrected_content.append(char)
                else:
                    unmatched_closing_brackets += 1  # Count unmatched closing brackets
                    logger.warning(f"Unmatched closing bracket at position {i}, ignoring it.")
            else:
                corrected_content.append(char)

        # Add missing closing brackets at the end
        while stack:
            unmatched_opening_position = stack.pop()
            logger.warning(f"Unmatched opening bracket at position {unmatched_opening_position}, adding a closing bracket.")
            corrected_content.append(']')

        # Log the number of unmatched brackets
        if unmatched_closing_brackets > 0:
            logger.warning(f"Total unmatched closing brackets ignored: {unmatched_closing_brackets}")
        if len(stack) > 0:
            logger.warning(f"Total unmatched opening brackets fixed: {len(stack)}")

        corrected_content_str = ''.join(corrected_content)
        logger.info("Mismatched brackets corrected.")
        return corrected_content_str
    
    def correct_file(self, file_path: Union[str, Path]) -> None:
        """
        Correct a single CSV file.
        
        Args:
            file_path: Path to the CSV file to correct
        """
        file_path = Path(file_path)
        logger.info(f"Correcting file: {file_path}")
        
        # Read the file content
        try:
            with open(file_path, 'r') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return
        
        # Correct the content
        corrected_content = self._correct_csv_content(content)
        if not corrected_content:
            logger.warning(f"No corrections made for file: {file_path}")
            return
        
        # Save the corrected content
        stock_symbol = file_path.parent.name
        stock_output_dir = self.output_dir / stock_symbol
        stock_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = stock_output_dir / file_path.name
        
        try:
            with open(output_path, 'w') as f:
                f.write(corrected_content)
            logger.info(f"Saved corrected file to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving corrected file {output_path}: {str(e)}")
    
    def correct_all_files(self, pattern: str = "historical_data.csv") -> None:
        """
        Correct all CSV files in the input directory.
        
        Args:
            pattern: File pattern to match (default: "historical_data.csv")
        """
        for file_path in self.input_dir.rglob(pattern):
            self.correct_file(file_path)