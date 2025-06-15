import json
import pandas as pd
import logging
from typing import Dict, Tuple
from ..registry import CLEANER  # Import the CLEANER registry

@CLEANER.register_module()
class StockForecastsCleaner:
    """
    Class for cleaning and parsing stock forecasts JSON data.
    """

    def __init__(self):
        """
        Initialize the StockForecastsCleaner with logging.
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def parse_stock_forecasts_json(self, json_file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parses stock forecasts JSON data into two DataFrames: one for estimates and actuals,
        and another for snapshots.

        Args:
            json_file_path (str): Path to the JSON file containing stock forecasts data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - DataFrame for estimates and actuals
                - DataFrame for snapshots
        """
        try:
            self.logger.info(f"Parsing JSON file: {json_file_path}")
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            # Debugging: Log the loaded JSON content
            self.logger.debug(f"Loaded JSON content: {data}")

            # Validate the JSON structure
            if not isinstance(data, dict) or 'periods' not in data:
                self.logger.error("Invalid JSON structure: Expected a dictionary with 'periods'.")
                return pd.DataFrame(), pd.DataFrame()

            estimates_actuals_rows = []
            snapshots_rows = []

            # Iterate through periods
            for period in data['periods']:
                if not isinstance(period, dict):
                    self.logger.warning(f"Invalid period structure: {period}")
                    continue

                base_info = {
                    'CalendarMonth': period.get('CalendarMonth'),
                    'CalendarYear': period.get('CalendarYear'),
                    'FiscalYear': period.get('FiscalPeriod', {}).get('Year'),
                    'ActualReportDate': period.get('ActualReportDate')
                }

                # Handle Actuals
                if period.get('Actuals') and period['Actuals'].get('Actual'):
                    for actual in period['Actuals']['Actual']:
                        row = base_info.copy()
                        row.update({
                            'Type': 'Actual',
                            'CurrencyCode': actual.get('CurrencyCode', ""),
                            'Reported': actual.get('Reported', ""),
                            'ReportedDate': actual.get('ReportedDate', ""),
                            'SurprisePercent': actual.get('SurprisePercent', ""),
                            'SurpriseMean': actual.get('SurpriseMean', ""),
                            'SUE': actual.get('StandardizedUnexpectedEarnings', ""),
                            'NumEstimates': actual.get('NumberOfEstimates', "")
                        })
                        estimates_actuals_rows.append(row)

                # Handle Estimates
                if period.get('Estimates') and period['Estimates'].get('Estimate'):
                    for estimate in period['Estimates']['Estimate']:
                        row = base_info.copy()
                        row.update({
                            'Type': 'Estimate',
                            'CurrencyCode': estimate.get('CurrencyCode', ""),
                            'Mean': estimate.get('Mean', ""),
                            'High': estimate.get('High', ""),
                            'Low': estimate.get('Low', ""),
                            'Median': estimate.get('Median', ""),
                            'StandardDeviation': estimate.get('StandardDeviation', ""),
                            'SmartEstimate': estimate.get('SmartEstimate', ""),
                            'NumEstimates': estimate.get('NumberOfEstimates', "")
                        })
                        estimates_actuals_rows.append(row)

                # Handle EstimateSnapshots
                if period.get('EstimateSnapshots') and period['EstimateSnapshots'].get('EstimateSnapshot'):
                    for snapshot in period['EstimateSnapshots']['EstimateSnapshot']:
                        row = {
                            'Age': snapshot.get('Age', ""),
                            'CurrencyCode': snapshot.get('CurrencyCode', ""),
                            'Mean': snapshot.get('Mean', ""),
                            'High': snapshot.get('High', ""),
                            'Low': snapshot.get('Low', ""),
                            'Median': snapshot.get('Median', ""),
                            'StandardDeviation': snapshot.get('StandardDeviation', ""),
                            'SmartEstimate': snapshot.get('SmartEstimate', ""),
                            'NumEstimates': snapshot.get('NumberOfEstimates', "")
                        }
                        snapshots_rows.append(row)

            # Convert rows to DataFrames
            estimates_actuals_df = pd.DataFrame(estimates_actuals_rows)
            snapshots_df = pd.DataFrame(snapshots_rows)

            self.logger.info(f"Successfully parsed JSON file: {json_file_path}")
            return estimates_actuals_df, snapshots_df

        except Exception as e:
            self.logger.error(f"Error parsing JSON file {json_file_path}: {e}")
            return pd.DataFrame(), pd.DataFrame()
