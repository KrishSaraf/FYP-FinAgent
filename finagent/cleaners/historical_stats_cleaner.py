import json
import pandas as pd
import logging
from typing import Dict
from ..registry import CLEANER  # Import the CLEANER registry

@CLEANER.register_module()
class HistoricalStatsCleaner:
    """
    Class for cleaning and parsing historical stats JSON data.
    """

    def __init__(self):
        """
        Initialize the HistoricalStatsCleaner with logging.
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def parse_historical_stats_json(self, json_file_path: str) -> pd.DataFrame:
        """
        Parses historical stats JSON data into a single DataFrame.

        Args:
            json_file_path (str): Path to the JSON file containing historical stats data.

        Returns:
            pd.DataFrame: A DataFrame containing the cleaned historical stats data.
        """
        try:
            self.logger.info(f"Parsing JSON file: {json_file_path}")
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            # Initialize an empty DataFrame
            df = pd.DataFrame()

            # Iterate through each key in the JSON data
            for key, values in data.items():
                # Convert the nested dictionary into a DataFrame
                temp_df = pd.DataFrame(values.items(), columns=["Date", key])
                if df.empty:
                    df = temp_df
                else:
                    df = pd.merge(df, temp_df, on="Date", how="outer")

            # Sort the DataFrame by Date
            df["Date"] = pd.to_datetime(df["Date"], format="%b %Y", errors="coerce")
            df = df.sort_values(by="Date").reset_index(drop=True)

            self.logger.info(f"Successfully parsed JSON file: {json_file_path}")
            return df

        except Exception as e:
            self.logger.error(f"Error parsing JSON file {json_file_path}: {e}")
            return pd.DataFrame()
