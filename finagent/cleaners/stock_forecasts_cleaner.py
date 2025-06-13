import json
import pandas as pd
import logging
from typing import Dict
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

    def parse_stock_forecasts_json(self, json_file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Parses stock forecasts JSON data into multiple DataFrames.

        Args:
            json_file_path (str): Path to the JSON file containing stock forecasts data.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of DataFrames for each relevant section.
        """
        try:
            self.logger.info(f"Parsing JSON file: {json_file_path}")
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            # Debugging: Log the loaded JSON content
            self.logger.debug(f"Loaded JSON content: {data}")

            # Validate the JSON structure
            if not isinstance(data, dict):
                self.logger.error("Invalid JSON structure: Expected a dictionary.")
                return {}

            dataframes = {}

            # Extract measure information
            measure_code = data.get("measureCode", "Unknown")
            measure_name = data.get("measureName", "Unknown")
            self.logger.debug(f"Measure Code: {measure_code}, Measure Name: {measure_name}")

            # Iterate through periods
            periods = data.get("periods", [])
            if not periods:
                self.logger.warning("No periods found in the JSON file.")
                return {}

            actuals_data = []
            estimates_data = []
            snapshots_data = []

            for period in periods:
                if not isinstance(period, dict):
                    self.logger.warning(f"Invalid period structure: {period}")
                    continue

                fiscal_year = period.get("FiscalPeriod", {}).get("Year", "Unknown")
                calendar_year = period.get("CalendarYear", "Unknown")
                calendar_month = period.get("CalendarMonth", "Unknown")
                actual_report_date = period.get("ActualReportDate", None)

                # Debugging: Log the period details
                self.logger.debug(f"Processing period: FiscalYear={fiscal_year}, CalendarYear={calendar_year}, CalendarMonth={calendar_month}")

                # Extract Actuals
                actuals = period.get("Actuals", {}).get("Actual", [])
                if actuals is None:
                    self.logger.warning(f"No actuals found for period: {period}")
                else:
                    for actual in actuals:
                        actuals_data.append({
                            "FiscalYear": fiscal_year,
                            "CalendarYear": calendar_year,
                            "CalendarMonth": calendar_month,
                            "ActualReportDate": actual_report_date,
                            "CurrencyCode": actual.get("CurrencyCode", None),
                            "Reported": actual.get("Reported", None),
                            "ReportedDate": actual.get("ReportedDate", None),
                            "SurprisePercent": actual.get("SurprisePercent", None),
                            "SurpriseMean": actual.get("SurpriseMean", None),
                            "StandardizedUnexpectedEarnings": actual.get("StandardizedUnexpectedEarnings", None),
                            "NumberOfEstimates": actual.get("NumberOfEstimates", None)
                        })

                # Extract Estimates
                estimates = period.get("Estimates", {}).get("Estimate", [])
                if estimates is None:
                    self.logger.warning(f"No estimates found for period: {period}")
                else:
                    for estimate in estimates:
                        estimates_data.append({
                            "FiscalYear": fiscal_year,
                            "CalendarYear": calendar_year,
                            "CalendarMonth": calendar_month,
                            "CurrencyCode": estimate.get("CurrencyCode", None),
                            "Mean": estimate.get("Mean", None),
                            "High": estimate.get("High", None),
                            "Low": estimate.get("Low", None),
                            "NumberOfEstimates": estimate.get("NumberOfEstimates", None),
                            "Median": estimate.get("Median", None),
                            "StandardDeviation": estimate.get("StandardDeviation", None),
                            "SmartEstimate": estimate.get("SmartEstimate", None)
                        })

                # Extract EstimateSnapshots
                snapshots = period.get("EstimateSnapshots", {}).get("EstimateSnapshot", [])
                if snapshots is None:
                    self.logger.warning(f"No estimate snapshots found for period: {period}")
                else:
                    for snapshot in snapshots:
                        snapshots_data.append({
                            "FiscalYear": fiscal_year,
                            "CalendarYear": calendar_year,
                            "CalendarMonth": calendar_month,
                            "CurrencyCode": snapshot.get("CurrencyCode", None),
                            "Mean": snapshot.get("Mean", None),
                            "High": snapshot.get("High", None),
                            "Low": snapshot.get("Low", None),
                            "NumberOfEstimates": snapshot.get("NumberOfEstimates", None),
                            "Median": snapshot.get("Median", None),
                            "StandardDeviation": snapshot.get("StandardDeviation", None),
                            "Age": snapshot.get("Age", None),
                            "SmartEstimate": snapshot.get("SmartEstimate", None)
                        })

            # Convert lists to DataFrames
            if actuals_data:
                df_actuals = pd.DataFrame(actuals_data)
                dataframes["Actuals"] = df_actuals

            if estimates_data:
                df_estimates = pd.DataFrame(estimates_data)
                dataframes["Estimates"] = df_estimates

            if snapshots_data:
                df_snapshots = pd.DataFrame(snapshots_data)
                dataframes["EstimateSnapshots"] = df_snapshots

            self.logger.info(f"Successfully parsed JSON file: {json_file_path}")
            return dataframes

        except Exception as e:
            self.logger.error(f"Error parsing JSON file {json_file_path}: {e}")
            return {}
