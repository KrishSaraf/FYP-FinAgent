import json
import pandas as pd
import logging
from typing import Dict
from ..registry import CLEANER  # Import the CLEANER registry

@CLEANER.register_module()
class PriceCleaner:
    """
    Class for cleaning and parsing stock price JSON data.
    """

    def __init__(self):
        """
        Initialize the PriceCleaner with logging.
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def parse_price_json(self, json_file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Parses stock price JSON data into multiple DataFrames.

        Args:
            json_file_path (str): Path to the JSON file containing stock price data.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of DataFrames for each dataset.
        """
        try:
            self.logger.info(f"Parsing JSON file: {json_file_path}")
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            dataframes = {}

            for dataset in data["datasets"]:
                label = dataset["label"]
                values = dataset["values"]

                # Handle Volume data with nested delivery info
                if label == "Volume":
                    df = pd.DataFrame(values, columns=["Date", "Volume", "Meta"])
                    df["Delivery (%)"] = df["Meta"].apply(
                        lambda x: x.get("delivery") if isinstance(x, dict) else None
                    )
                    df.drop(columns=["Meta"], inplace=True)
                else:
                    df = pd.DataFrame(values, columns=["Date", label])
                    df[label] = pd.to_numeric(df[label], errors="coerce")  # Convert to numeric

                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")  # Convert to datetime
                dataframes[label] = df

            self.logger.info(f"Successfully parsed JSON file: {json_file_path}")
            return dataframes

        except Exception as e:
            self.logger.error(f"Error parsing JSON file {json_file_path}: {e}")
            return {}
