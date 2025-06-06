from datetime import datetime
import os
import time
import pandas as pd
from tqdm import tqdm
import requests
from urllib.request import urlopen
import certifi
import json
import logging
from typing import Any, Dict, List, Optional, Literal
from ..custom import Downloader
from ...registry import DOWNLOADER
from dotenv import load_dotenv

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

# Type definitions based on API schema
PeriodEnum = Literal["1m", "6m", "1yr", "3yr", "5yr", "10yr", "max"]
FilterEnum = Literal["default", "price", "pe", "sm", "evebitda", "ptb", "mcs"]
PeriodType = Literal["Annual", "Interim"]
DataType = Literal["Actuals", "Estimates"]
DataAge = Literal["OneWeekAgo", "ThirtyDaysAgo", "SixtyDaysAgo", "NinetyDaysAgo", "Current"]
MeasureCode = Literal["EPS", "CPS", "CPX", "DPS", "EBI", "EBT", "GPS", "GRM", "NAV", "NDT", "NET", "PRE", "ROA", "ROE", "SAL"]

@DOWNLOADER.register_module(force=True)
class IndianMarketDownloader(Downloader):
    def __init__(self,
                 root: str = "",
                 token: str = None,
                 delay: int = 1,
                 start_date: str = "2020-01-01",
                 end_date: str = "2024-01-01",
                 interval: str = "1d",
                 stocks_path: str = None,
                 workdir: str = "",
                 tag: str = "",
                 **kwargs):
        self.root = root
        load_dotenv(os.path.join(root, "apikeys.env"))
        self.token = token if token is not None else os.environ.get("INDIAN_API_KEY")
        if not self.token:
            raise ValueError("API token not provided. Please set INDIAN_API_KEY in apikeys.env.")
            
        self.delay = delay
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.stocks_path = os.path.join(root, stocks_path)
        self.tag = tag
        self.workdir = os.path.join(root, workdir, tag)

        # Setup logging
        self.log_path = os.path.join(self.workdir, "{}.log".format(tag))
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)  # Ensure the directory exists
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        self.stocks = self._init_stocks()

        # Base URL for the API
        self.base_url = "https://stock.indianapi.in"

        super().__init__(**kwargs)

    def _init_stocks(self):
        try:
            with open(self.stocks_path) as op:
                stocks = [line.strip() for line in op.readlines()]
            return stocks
        except Exception as e:
            self.logger.error(f"Error reading stocks file: {e}")
            return []

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Generic method to make API requests with improved error handling and a delay."""
        try:
            headers = {
                "x-api-key": self.token,
                "Accept": "application/json"
            }
            response = requests.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            time.sleep(self.delay)  # Add a delay after each request
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed for {endpoint}: {e}")
            if hasattr(e.response, 'text'):
                self.logger.error(f"Response text: {e.response.text}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error during API request to {endpoint}: {e}")
            return None

    def get_stock_details(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a stock"""
        return self._make_request("stock", {"name": name})

    def get_ipo_data(self) -> Optional[Dict[str, Any]]:
        """Get IPO data"""
        return self._make_request("ipo")

    def get_news_data(self) -> Optional[Dict[str, Any]]:
        """Get news data"""
        return self._make_request("news")

    def get_trending_stocks(self) -> Optional[Dict[str, Any]]:
        """Get trending stocks"""
        return self._make_request("trending")

    def get_statement(self, stock_name: str, stats: str) -> Optional[Dict[str, Any]]:
        """Get financial statements"""
        return self._make_request("statement", {
            "stock_name": stock_name,
            "stats": stats
        })

    def get_commodities_data(self) -> Optional[Dict[str, Any]]:
        """Get commodities data"""
        return self._make_request("commodities")

    def get_mutual_funds_data(self) -> Optional[Dict[str, Any]]:
        """Get mutual funds data"""
        return self._make_request("mutual_funds")

    def get_price_shockers(self) -> Optional[Dict[str, Any]]:
        """Get price shockers data"""
        return self._make_request("price_shockers")

    def get_most_active_bse(self) -> Optional[List[Dict[str, Any]]]:
        """Get most active stocks on BSE"""
        return self._make_request("BSE_most_active")

    def get_most_active_nse(self) -> Optional[List[Dict[str, Any]]]:
        """Get most active stocks on NSE"""
        return self._make_request("NSE_most_active")

    def get_historical_data(self,
                          stock_name: str,
                          period: PeriodEnum = "5yr",
                          filter: FilterEnum = "default") -> Optional[Dict[str, Any]]:
        """Get historical data for a stock"""
        return self._make_request("historical_data", {
            "stock_name": stock_name,
            "period": period,
            "filter": filter
        })

    def get_industry_search(self, query: str) -> Optional[Dict[str, Any]]:
        """Search for companies in an industry"""
        return self._make_request("industry_search", {"query": query})

    def get_stock_forecasts(self,
                          stock_id: str,
                          measure_code: MeasureCode,
                          period_type: PeriodType,
                          data_type: DataType,
                          age: DataAge) -> Optional[Dict[str, Any]]:
        """Get stock forecasts"""
        return self._make_request("stock_forecasts", {
            "stock_id": stock_id,
            "measure_code": measure_code,
            "period_type": period_type,
            "data_type": data_type,
            "age": age
        })

    def get_historical_stats(self,
                           stock_name: str,
                           stats: str) -> Optional[Dict[str, Any]]:
        """Get historical statistics"""
        return self._make_request("historical_stats", {
            "stock_name": stock_name,
            "stats": stats
        })

    def get_corporate_actions(self, stock_name: str) -> Optional[Dict[str, Any]]:
        """Get corporate actions data"""
        return self._make_request("corporate_actions", {"stock_name": stock_name})

    def get_mutual_fund_search(self, query: str) -> Optional[Dict[str, Any]]:
        """Search for mutual funds"""
        return self._make_request("mutual_fund_search", {"query": query})

    def get_stock_target_price(self, stock_id: str) -> Optional[Dict[str, Any]]:
        """Get stock target price"""
        return self._make_request("stock_target_price", {"stock_id": stock_id})

    def get_mutual_fund_details(self, stock_name: str) -> Optional[Dict[str, Any]]:
        """Get mutual fund details"""
        return self._make_request("mutual_funds_details", {"stock_name": stock_name})

    def get_recent_announcements(self, stock_name: str) -> Optional[Dict[str, Any]]:
        """Get recent announcements"""
        return self._make_request("recent_announcements", {"stock_name": stock_name})

    def get_52_week_high_low(self) -> Optional[Dict[str, Any]]:
        """Get 52-week high/low data"""
        return self._make_request("fetch_52_week_high_low_data")

    def download(self,
                 stocks: Optional[List[str]] = None,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None):
        """Enhanced main download method with additional data types"""
        start_date = datetime.strptime(start_date if start_date else self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date if end_date else self.end_date, "%Y-%m-%d")
        
        stocks = stocks if stocks else self.stocks
        if not stocks:
            self.logger.error("No stocks provided for download")
            return

        # Create base directories
        os.makedirs(os.path.join(self.workdir, "market_data"), exist_ok=True)
        os.makedirs(os.path.join(self.workdir, "mutual_funds"), exist_ok=True)
        os.makedirs(os.path.join(self.workdir, "commodities"), exist_ok=True)
        os.makedirs(os.path.join(self.workdir, "ipo"), exist_ok=True)

        # Download market-wide data
        market_data = {
            "trending": self.get_trending_stocks(),
            "52week_high_low": self.get_52_week_high_low(),
            "most_active_nse": self.get_most_active_nse(),
            "most_active_bse": self.get_most_active_bse(),
            "price_shockers": self.get_price_shockers(),
            "commodities": self.get_commodities_data(),
            "mutual_funds": self.get_mutual_funds_data(),
            "ipo": self.get_ipo_data()
        }

        for data_type, data in market_data.items():
            if data is not None:
                file_path = os.path.join(self.workdir, "market_data", f"{data_type}.json")
                with open(file_path, "w") as json_file:
                    json.dump(data, json_file, indent=4)
                self.logger.info(f"Successfully downloaded {data_type}")

        # Download stock-specific data
        for stock in tqdm(stocks, desc="Downloading stock data"):
            stock_dir = os.path.join(self.workdir, stock)
            os.makedirs(stock_dir, exist_ok=True)

            # Download all data types for each stock
            data_types = {
                "stock_details": lambda s: self.get_stock_details(s),
                "historical_data": lambda s: self.get_historical_data(s, period="5yr"),
                "historical_stats": lambda s: self.get_historical_stats(s, "quarter_results"),
                "corporate_actions": lambda s: self.get_corporate_actions(s),
                "recent_announcements": lambda s: self.get_recent_announcements(s),
                "stock_forecasts": lambda s: self.get_stock_forecasts(
                    s, "EPS", "Annual", "Actuals", "Current"
                )
            }

            for data_type, download_func in data_types.items():
                try:
                    data = download_func(stock)
                    
                    if data is not None:
                        file_path = os.path.join(stock_dir, f"{data_type}.json")
                        with open(file_path, "w") as json_file:
                            json.dump(data, json_file, indent=4)
                        self.logger.info(f"Successfully downloaded {data_type} for {stock}")
                    else:
                        self.logger.warning(f"Failed to download {data_type} for {stock}")
                except Exception as e:
                    self.logger.error(f"Error downloading {data_type} for {stock}: {e}")

            time.sleep(self.delay)  # Rate limiting

        self.logger.info("Download completed successfully")