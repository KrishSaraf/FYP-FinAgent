import json
import pandas as pd
import logging
from typing import Dict
from ..registry import CLEANER  # Import the CLEANER registry

@CLEANER.register_module()
class StockDetailsCleaner:
    """
    Class for cleaning and parsing stock details JSON data.
    """

    def __init__(self):
        """
        Initialize the StockDetailsCleaner with logging.
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def parse_stock_details_json(self, json_file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Parses stock details JSON data into multiple DataFrames.

        Args:
            json_file_path (str): Path to the JSON file containing stock details data.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary of DataFrames for each relevant section.
        """
        try:
            self.logger.info(f"Parsing JSON file: {json_file_path}")
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            dataframes = {}

            # Extract company profile
            if "companyProfile" in data:
                company_profile = data["companyProfile"]

                # Handle top-level fields in companyProfile
                top_level_fields = {key: value for key, value in company_profile.items() if not isinstance(value, (dict, list))}
                df_company_profile = pd.DataFrame([top_level_fields])
                dataframes["CompanyProfile"] = df_company_profile

                # Handle nested officers data
                if "officers" in company_profile and isinstance(company_profile["officers"], dict):
                    officers = company_profile["officers"].get("officer", [])
                    df_officers = pd.DataFrame(officers)
                    dataframes["CompanyProfile_Officers"] = df_officers

                # Handle peer company list
                if "peerCompanyList" in company_profile and isinstance(company_profile["peerCompanyList"], list):
                    peer_company_list = company_profile["peerCompanyList"]
                    df_peer_companies = pd.DataFrame(peer_company_list)
                    dataframes["CompanyProfile_PeerCompanies"] = df_peer_companies

            # Extract current price
            if "currentPrice" in data:
                current_price = data["currentPrice"]
                df_current_price = pd.DataFrame([current_price])
                dataframes["CurrentPrice"] = df_current_price

            # Extract stock technical data
            if "stockTechnicalData" in data:
                stock_technical_data = data["stockTechnicalData"]
                df_stock_technical_data = pd.DataFrame(stock_technical_data)
                dataframes["StockTechnicalData"] = df_stock_technical_data

            # Extract financials
            if "financials" in data:
                financials = data["financials"]

                # Iterate through each financial record
                for record in financials:
                    fiscal_year = record.get("FiscalYear", "Unknown")
                    statement_type = record.get("Type", "Unknown")
                    stock_financial_map = record.get("stockFinancialMap", {})

                    # Separate `CAS` (Cash Flow Statement)
                    if "CAS" in stock_financial_map:
                        df_cas = pd.DataFrame(stock_financial_map["CAS"])
                        df_cas["FiscalYear"] = fiscal_year
                        df_cas["Type"] = statement_type
                        dataframes[f"Financials_CAS_{fiscal_year}_{statement_type}"] = df_cas

                    # Separate `BAL` (Balance Sheet)
                    if "BAL" in stock_financial_map:
                        df_bal = pd.DataFrame(stock_financial_map["BAL"])
                        df_bal["FiscalYear"] = fiscal_year
                        df_bal["Type"] = statement_type
                        dataframes[f"Financials_BAL_{fiscal_year}_{statement_type}"] = df_bal

                    # Separate `INC` (Income Statement)
                    if "INC" in stock_financial_map:
                        df_inc = pd.DataFrame(stock_financial_map["INC"])
                        df_inc["FiscalYear"] = fiscal_year
                        df_inc["Type"] = statement_type
                        dataframes[f"Financials_INC_{fiscal_year}_{statement_type}"] = df_inc

            # Extract key metrics
            if "keyMetrics" in data:
                key_metrics = data["keyMetrics"]
                for key, values in key_metrics.items():
                    df_key_metrics = pd.DataFrame(values)
                    dataframes[f"KeyMetrics_{key}"] = df_key_metrics

            # Extract recent news
            if "recentNews" in data:
                recent_news = data["recentNews"]
                df_recent_news = pd.DataFrame(recent_news)
                dataframes["RecentNews"] = df_recent_news

            # Extract corporate actions
            if "stockCorporateActionData" in data:
                corporate_actions = data["stockCorporateActionData"]

                # Handle bonus
                if "bonus" in corporate_actions and isinstance(corporate_actions["bonus"], list):
                    df_bonus = pd.DataFrame(corporate_actions["bonus"])
                    dataframes["CorporateActions_Bonus"] = df_bonus

                # Handle dividend
                if "dividend" in corporate_actions and isinstance(corporate_actions["dividend"], list):
                    df_dividend = pd.DataFrame(corporate_actions["dividend"])
                    dataframes["CorporateActions_Dividend"] = df_dividend

                # Handle rights
                if "rights" in corporate_actions and isinstance(corporate_actions["rights"], list):
                    df_rights = pd.DataFrame(corporate_actions["rights"])
                    dataframes["CorporateActions_Rights"] = df_rights

                # Handle splits
                if "splits" in corporate_actions and isinstance(corporate_actions["splits"], list):
                    df_splits = pd.DataFrame(corporate_actions["splits"])
                    dataframes["CorporateActions_Splits"] = df_splits

                # Handle annual general meetings
                if "annualGeneralMeeting" in corporate_actions and isinstance(corporate_actions["annualGeneralMeeting"], list):
                    df_agm = pd.DataFrame(corporate_actions["annualGeneralMeeting"])
                    dataframes["CorporateActions_AGM"] = df_agm

                # Handle board meetings
                if "boardMeetings" in corporate_actions and isinstance(corporate_actions["boardMeetings"], list):
                    df_board_meetings = pd.DataFrame(corporate_actions["boardMeetings"])
                    dataframes["CorporateActions_BoardMeetings"] = df_board_meetings


            self.logger.info(f"Successfully parsed JSON file: {json_file_path}")
            return dataframes

        except Exception as e:
            self.logger.error(f"Error parsing JSON file {json_file_path}: {e}")
            return {}
