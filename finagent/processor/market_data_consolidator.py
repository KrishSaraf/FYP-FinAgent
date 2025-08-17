import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from finagent.registry import PROCESSOR

@PROCESSOR.register_module()
class MarketDataConsolidator:
    """
    Consolidates market data from multiple sources into unified time-series format
    for RL trading environment.
    """
    
    def __init__(self, data_root: str = "market_data"):
        self.data_root = Path(data_root)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Define data source paths
        self.market_data_path = self.data_root / "indian_market"
        self.cleaned_data_path = self.data_root / "cleaned_data"
        self.social_media_path = Path("social_media_data/cleaned_data")
        
        # Stock list
        self.stocks = self._load_stock_list()

        # Load FinBERT model and tokenizer
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
    
    def _load_stock_list(self) -> List[str]:
        """Load list of stocks from stocks.txt"""
        stocks_file = Path("finagent/stocks.txt")
        if stocks_file.exists():
            with open(stocks_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        self.logger.warning("Stock list file not found.")
        return []
    
    def consolidate_stock_data(self, stock_symbol: str) -> pd.DataFrame:
        """
        Consolidate all available data for a single stock into unified DataFrame.
        
        Args:
            stock_symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            
        Returns:
            Consolidated DataFrame with all market data aligned by date
        """
        try:
            self.logger.info(f"Consolidating data for {stock_symbol}")
            
            # Load price and technical data
            price_data = self._load_price_data(stock_symbol)
            fundamental_data = self._load_fundamental_data(stock_symbol)
            sentiment_data = self._load_sentiment_data(stock_symbol)
            corporate_data = self._load_corporate_data(stock_symbol)
            
            # Merge all data sources
            consolidated_df = self._merge_data_sources(
                price_data, fundamental_data, sentiment_data, corporate_data
            )
            
            # Fill missing values and align timestamps
            consolidated_df = self._clean_and_align_data(consolidated_df)
            
            self.logger.info(f"Successfully consolidated data for {stock_symbol}: {len(consolidated_df)} rows")
            return consolidated_df
            
        except Exception as e:
            self.logger.error(f"Error consolidating data for {stock_symbol}: {e}")
            return pd.DataFrame()
    
    def _load_price_data(self, stock_symbol: str) -> pd.DataFrame:
        """Load price and technical data"""
        stock_path = self.cleaned_data_path / stock_symbol
        price_data = {}
        
        try:
            # Load price data
            price_file = stock_path / "Price on NSE.csv"
            if price_file.exists():
                df = pd.read_csv(price_file)
                if 'Date' in df.columns and 'Price on NSE' in df.columns:
                    df['date'] = pd.to_datetime(df['Date'])
                    df.set_index('date', inplace=True)
                    price_data['price'] = df['Price on NSE']
                else:
                    self.logger.warning(f"Missing required columns in {price_file}")
            
            # Load volume data
            volume_file = stock_path / "Volume.csv"
            if volume_file.exists():
                df = pd.read_csv(volume_file)
                if 'Date' in df.columns and 'Volume' in df.columns:
                    df['date'] = pd.to_datetime(df['Date'])
                    df.set_index('date', inplace=True)
                    price_data['volume'] = df['Volume']
                    price_data['delivery_percentage'] = df.get('Delivery (%)', np.nan)
                else:
                    self.logger.warning(f"Missing required columns in {volume_file}")
            
            # Load DMA data
            dma50_file = stock_path / "50 DMA.csv"
            if dma50_file.exists():
                df = pd.read_csv(dma50_file)
                if 'Date' in df.columns and '50 DMA' in df.columns:
                    df['date'] = pd.to_datetime(df['Date'])
                    df.set_index('date', inplace=True)
                    price_data['dma50'] = df['50 DMA']
                else:
                    self.logger.warning(f"Missing required columns in {dma50_file}")
            
            dma200_file = stock_path / "200 DMA.csv"
            if dma200_file.exists():
                df = pd.read_csv(dma200_file)
                if 'Date' in df.columns and '200 DMA' in df.columns:
                    df['date'] = pd.to_datetime(df['Date'])
                    df.set_index('date', inplace=True)
                    price_data['dma200'] = df['200 DMA']
                else:
                    self.logger.warning(f"Missing required columns in {dma200_file}")
            
            if price_data:
                # print(price_data)
                return pd.DataFrame(price_data)
            else:
                self.logger.warning(f"No price data found for {stock_symbol}")
                return pd.DataFrame()
        
        except Exception as e:
            self.logger.error(f"Error loading price data for {stock_symbol}: {e}")
            return pd.DataFrame()
    
    def _load_fundamental_data(self, stock_symbol: str) -> pd.DataFrame:
        """
        Loads and consolidates fundamental data to create a time-series dataset.

        This method integrates three types of data:
        1. KeyMetrics files for baseline, non-time-specific metrics.
        2. Annual financial files, with the period ending on March 31st of the year in the filename.
        3. Interim (quarterly) financial files, with the period end date parsed from the filename.

        The data is then aligned to the dates of board meetings where financial results were announced,
        creating a historical view of the available data at the time of each announcement.
        """
        stock_path = self.cleaned_data_path / stock_symbol
        if not stock_path.exists():
            self.logger.warning(f"Data directory not found for symbol: {stock_symbol}")
            return pd.DataFrame()

        # --- 1. Load Base KeyMetrics ---
        base_metrics_dict: Dict[str, any] = {}
        key_metrics_files = list(stock_path.glob("KeyMetrics_*.csv"))
        for file_path in key_metrics_files:
            try:
                df = pd.read_csv(file_path)
                if 'key' not in df.columns or 'value' not in df.columns:
                    self.logger.warning(f"Skipping {file_path}: missing 'key' or 'value' columns.")
                    continue
                df.dropna(subset=['key', 'value'], inplace=True)
                base_metrics_dict.update(pd.Series(df.value.values, index=df.key).to_dict())
            except Exception as e:
                self.logger.error(f"Error processing KeyMetrics file {file_path}: {e}")

        # --- 2. Pre-load all time-stamped Financial Data into a Master Lookup ---
        data_by_period_end: Dict[pd.Timestamp, dict] = {}

        # Part A: Process Annual Files (from Snippet 1 logic)
        annual_files = list(stock_path.glob("Financials_*_Annual.csv"))
        for file_path in annual_files:
            match = re.search(r'_(\d{4})_Annual', file_path.name)
            if not match:
                continue
            try:
                year = int(match.group(1))
                # Assumption: Annual reports for the fiscal year ending March 31st.
                period_end_date = pd.Timestamp(year=year, month=3, day=31)
                metrics_series = pd.read_csv(file_path).set_index('key')['value']
                
                if period_end_date not in data_by_period_end:
                    data_by_period_end[period_end_date] = {}
                data_by_period_end[period_end_date].update(metrics_series.to_dict())
            except Exception as e:
                self.logger.error(f"Error processing annual file {file_path}: {e}")

        # Part B: Process single-observation Interim Files (calculating date from CSV content)
        interim_files = list(stock_path.glob("Financials_*_Interim.csv"))
        for file_path in interim_files:
            try:
                # --- Step 1: Read the CSV file, including the separate FiscalYear column ---
                df = pd.read_csv(file_path)
                if 'key' not in df.columns or 'value' not in df.columns or 'FiscalYear' not in df.columns:
                    self.logger.warning(f"Skipping {file_path}: missing key, value, or FiscalYear columns.")
                    continue
                
                # --- Step 2: Extract metadata from the file content ---
                metrics_series = df.set_index('key')['value']
                
                # Get FiscalYear from its dedicated column (using the first row's value).
                csv_fiscal_year = int(df['FiscalYear'].iloc[0])
                
                # Extract period details. periodNumber is essential for this calculation.
                period_length = pd.to_numeric(metrics_series.get('periodLength'), errors='coerce')
                period_type = metrics_series.get('periodType', 'Months').lower() # Default to months

                # Validate that we have the necessary data to proceed
                if pd.isna(period_length):
                    self.logger.warning(f"Skipping {file_path}: 'periodLength' key not found or invalid.")
                    continue

                # --- Step 3: Calculate the period-end date from the fiscal year start ---
                # Define a multiplier to handle different period types (e.g., months, quarters).
                unit_multiplier = {'months': 1, 'quarters': 3}.get(period_type, 1)
                total_months_offset = int(period_length * unit_multiplier)

                # A fiscal year (e.g., FY2024) starts on April 1 of the previous calendar year (2023).
                fy_start_date = pd.Timestamp(year=csv_fiscal_year - 1, month=4, day=1)
                
                # Calculate the end date by adding the total month offset to the fiscal year start
                # Example: FY24-Q1 -> 2023-04-01 + 3 months -> 2023-06-30
                target_end_date = fy_start_date + pd.DateOffset(months=total_months_offset)
                period_end_date = target_end_date - pd.DateOffset(days=1) + pd.offsets.MonthEnd(0)

                # --- Step 4: Store the loaded metrics with the calculated period-end date ---
                if period_end_date not in data_by_period_end:
                    data_by_period_end[period_end_date] = {}
                data_by_period_end[period_end_date].update(metrics_series.to_dict())

            except Exception as e:
                self.logger.error(f"Error processing interim file {file_path}: {e}")

        if not data_by_period_end:
            self.logger.warning(f"No time-stamped financial data could be loaded for {stock_symbol}.")
            # Still might return base metrics if available, but without a date context.
            # Returning empty for consistency with the method's time-series goal.
            return pd.DataFrame()

        sorted_period_end_dates = sorted(data_by_period_end.keys())
        
        # --- 3. Load Board Meeting Announcements and Match to Data ---
        actions_file = stock_path / "CorporateActions_BoardMeetings.csv"
        if not actions_file.exists():
            self.logger.warning(f"CorporateActions_BoardMeetings.csv not found for {stock_symbol}.")
            return pd.DataFrame()

        all_periods_data = []
        try:
            actions_df = pd.read_csv(actions_file, usecols=['boardMeetDate', 'purpose'])
            actions_df['date'] = pd.to_datetime(actions_df['boardMeetDate'], errors='coerce')
            actions_df.dropna(subset=['date'], inplace=True)
            
            # Find announcements for quarterly, half-yearly, or annual results
            relevant_announcements = actions_df[
                actions_df['purpose'].str.contains('result', case=False, na=False)
            ].copy()

            for _, announcement in relevant_announcements.iterrows():
                announcement_date = announcement['date']
                
                # Find the latest financial data available *before* this announcement date
                matching_period_end = None
                for end_date in sorted_period_end_dates:
                    if end_date < announcement_date:
                        matching_period_end = end_date
                    else:
                        break # Stop once we've passed the announcement date
                
                if matching_period_end:
                    # Start with the base metrics, then update with period-specific data
                    record = base_metrics_dict.copy()
                    record.update(data_by_period_end[matching_period_end])
                    record['date'] = announcement_date # This is the announcement date
                    record['period_end_date'] = matching_period_end # Add for clarity
                    all_periods_data.append(record)

        except Exception as e:
            self.logger.error(f"Error processing board meetings for {stock_symbol}: {e}")

        # --- 4. Create and Clean the Final DataFrame ---
        if not all_periods_data:
            self.logger.warning(f"No fundamental data points could be constructed for {stock_symbol}.")
            return pd.DataFrame()

        fundamental_df = pd.DataFrame(all_periods_data).sort_values('date')
        fundamental_df.drop_duplicates(subset=['date'], keep='last', inplace=True)
        fundamental_df.set_index('date', inplace=True)

        # Convert all potential metric columns to numeric, leaving date/ID columns alone
        for col in fundamental_df.columns:
            if fundamental_df[col].dtype == 'object' and col not in ['purpose']:
                fundamental_df[col] = pd.to_numeric(fundamental_df[col], errors='coerce')
        
        # Add a 'metric_' prefix to all columns except for the period end date
        fundamental_df.columns = [
            f"metric_{c}" if c != 'period_end_date' else c for c in fundamental_df.columns
        ]
        
        return fundamental_df
    
    def _load_sentiment_data(self, stock_symbol: str) -> pd.DataFrame:
        """Load and merge social media sentiment data into a single time-indexed DataFrame."""
        
        daily_reddit_df = pd.DataFrame()
        daily_twitter_df = pd.DataFrame()

        # Load Reddit data
        reddit_file = self.social_media_path / stock_symbol / "reddit.csv"
        if reddit_file.exists():
            try:
                df = pd.read_csv(reddit_file)
                if 'created_timestamp' in df.columns and not df['created_timestamp'].isnull().all():
                    # Correctly parse date strings, handle potential errors
                    df['date'] = pd.to_datetime(df['created_timestamp'], errors='coerce').dt.date
                    df.dropna(subset=['date'], inplace=True) # Drop rows where date conversion failed
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Perform FinBERT sentiment analysis
                    sentiments_title = self._analyze_sentiment(df['title'].tolist() if 'title' in df.columns else [])
                    sentiments_body = self._analyze_sentiment(df['body'].tolist() if 'body' in df.columns else [])
                    df['sentiment_title'] = sentiments_title if 'title' in df.columns else ['neutral'] * len(df)
                    df['sentiment_body'] = sentiments_body if 'body' in df.columns else ['neutral'] * len(df)

                    # Aggregate daily sentiment metrics
                    daily_reddit = df.groupby('date').agg({
                        'sentiment_title': lambda x: x.value_counts().to_dict(),  # Count sentiment labels
                        'sentiment_body': lambda x: x.value_counts().to_dict(),  # Count sentiment labels
                        'score': ['mean', 'sum', 'count'],
                        'num_comments': 'sum'
                    }).round(2)
                    
                    # Flatten the MultiIndex columns
                    daily_reddit.columns = ['_'.join(col).strip() for col in daily_reddit.columns.values]

                    # Now rename the flattened columns
                    daily_reddit.rename(columns={
                        'sentiment_title_<lambda>': 'reddit_title_sentiments',
                        'sentiment_body_<lambda>': 'reddit_body_sentiments',
                        'score_mean': 'reddit_score_mean',
                        'score_sum': 'reddit_score_sum',
                        'score_count': 'reddit_posts_count', # score_count is the number of posts
                        'num_comments_sum': 'reddit_comments_sum'
                    }, inplace=True)

                    daily_reddit_df = daily_reddit
            except Exception as e:
                self.logger.warning(f"Error loading Reddit data for {stock_symbol}: {e}")
        
        # Load Twitter data
        twitter_file = self.social_media_path / stock_symbol / "tweets.csv"
        if twitter_file.exists():
            try:
                df = pd.read_csv(twitter_file)
                if 'created_at' in df.columns:
                    df['date'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce').dt.date
                    df.dropna(subset=['date'], inplace=True)
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Perform FinBERT sentiment analysis
                    sentiments = self._analyze_sentiment(df['text'].tolist())
                    df['sentiment'] = sentiments

                    # Aggregate daily Twitter metrics
                    daily_twitter = df.groupby('date').agg({
                        'sentiment': lambda x: x.value_counts().to_dict(),  # Count sentiment labels
                        'like_count': ['mean', 'sum'],
                        'retweet_count': ['mean', 'sum'],
                        'reply_count': ['mean', 'sum'],
                        'view_count': ['mean', 'sum']
                    }).round(2)
                    
                    daily_twitter.columns = ['_'.join(col).strip() for col in daily_twitter.columns.values]

                    daily_twitter.rename(columns={
                        'sentiment_<lambda>': 'twitter_sentiments',
                        'like_count_mean': 'twitter_like_mean',
                        'like_count_sum': 'twitter_like_sum',
                        'retweet_count_mean': 'twitter_retweet_mean',
                        'retweet_count_sum': 'twitter_retweet_sum',
                        'reply_count_mean': 'twitter_reply_mean',
                        'reply_count_sum': 'twitter_reply_sum',
                        'view_count_mean': 'twitter_view_mean',
                        'view_count_sum': 'twitter_view_sum'
                    }, inplace=True)

            except Exception as e:
                self.logger.warning(f"Error loading Twitter data for {stock_symbol}: {e}")

        # Join the two DataFrames using their common date index
        if not daily_reddit_df.empty and not daily_twitter_df.empty:
            # If both have data, join them. 'outer' keeps all dates from both sources.
            return daily_reddit_df.join(daily_twitter_df, how='outer')
        elif not daily_reddit_df.empty:
            # If only Reddit has data, return it
            return daily_reddit_df
        elif not daily_twitter_df.empty:
            # If only Twitter has data, return it
            return daily_twitter_df
        else:
            # If no sentiment data was found, return an empty DataFrame
            return pd.DataFrame()
        
    def _analyze_sentiment(self, texts: List[str]) -> List[float]:
        """
        Analyze sentiment of text data using FinBERT and output numeric sentiment scores.

        Args:
            texts: A list of strings to analyze.

        Returns:
            A list of sentiment scores, where each score is between -1.0 and 1.0.
        """
        original_text_count = len(texts)
        try:
            sanitized_texts = [str(text) if text and isinstance(text, str) else "." for text in texts]
            if not sanitized_texts:
                return []

            # Tokenize texts in batches
            batch_size = 32  # Define batch size
            all_logits = []

            # Perform inference in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = sanitized_texts[i:i+batch_size]
                inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                all_logits.append(outputs.logits)

            # Concatenate all outputs
            logits = torch.cat(all_logits, dim=0)

            # Convert logits to probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)

            # Calculate sentiment scores
            positive_probs = probabilities[:, 0]  # Positive probability
            negative_probs = probabilities[:, 1]  # Negative probability

            # Score = Positive Probability - Negative Probability
            scores = positive_probs - negative_probs

            # Convert the result from a PyTorch tensor to a simple Python list
            return scores.tolist()
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return [0.0] * original_text_count  # Default to neutral score if error occurs
    
    def _load_corporate_data(self, stock_symbol: str) -> pd.DataFrame:
        """Load corporate actions and news data with FinBERT sentiment analysis."""
        stock_path = self.cleaned_data_path / stock_symbol
        action_files = list(stock_path.glob("CorporateActions_*.csv"))
        all_actions_dfs = []

        # --- Process Corporate Actions ---
        for file_path in action_files:
            try:
                df = pd.read_csv(file_path)
                if 'dateOfAnnouncement' not in df.columns:
                    continue

                # Set the announcement date as the index
                df['date'] = pd.to_datetime(df['dateOfAnnouncement'], errors='coerce')
                df.dropna(subset=['date'], inplace=True)
                df.set_index('date', inplace=True)
                
                action_type = file_path.stem.replace('CorporateActions_', '').lower()
                
                # Create a new DataFrame with just the data for this action
                if action_type == 'dividend' and 'value' in df.columns:
                    action_df = df[['value', 'interimOrFinal']].copy()
                    action_df.rename(columns={'value': 'dividend_amount', 'interimOrFinal': 'dividend_type'}, inplace=True)
                    all_actions_dfs.append(action_df)

                elif action_type == 'bonus' and 'remarks' in df.columns:
                    action_df = df[['remarks']].copy()
                    action_df.rename(columns={'remarks': 'bonus_remarks'}, inplace=True)
                    all_actions_dfs.append(action_df)

                elif action_type == 'splits' and 'oldFaceValue' in df.columns and 'newFaceValue' in df.columns:
                    action_df = df[['oldFaceValue', 'newFaceValue']].copy()
                    action_df['split_ratio'] = action_df['oldFaceValue'] / action_df['newFaceValue']
                    all_actions_dfs.append(action_df[['split_ratio']])

            except Exception as e:
                self.logger.warning(f"Error loading corporate actions {file_path}: {e}")
                continue
        
        # Concatenate all individual action DataFrames into one
        corporate_df = pd.concat(all_actions_dfs) if all_actions_dfs else pd.DataFrame()

        # --- Process Recent News ---
        news_df = pd.DataFrame()
        news_file = stock_path / "RecentNews.csv"
        if news_file.exists():
            try:
                df = pd.read_csv(news_file)
                if 'date' in df.columns and 'text' in df.columns:
                    # Parse dates and drop invalid rows
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df.dropna(subset=['date'], inplace=True)

                    # Perform FinBERT sentiment analysis on the news text
                    sentiments = self._analyze_sentiment(df['text'].tolist())
                    df['sentiment'] = sentiments

                    # Aggregate sentiment metrics by date
                    sentiment_counts = df.groupby('date')['sentiment'].apply(
                        lambda x: x.value_counts().to_dict()
                    ).reset_index(name='sentiment_counts')

                    # Convert sentiment counts into separate columns for positive, negative, and neutral
                    sentiment_counts = sentiment_counts.join(
                        pd.DataFrame(sentiment_counts['sentiment_counts'].tolist(), index=sentiment_counts.index)
                    ).fillna(0)

                    # Rename columns for clarity
                    sentiment_counts.rename(columns={
                        'positive': 'news_positive_count',
                        'negative': 'news_negative_count',
                        'neutral': 'news_neutral_count'
                    }, inplace=True)

                    # Set the date as the index
                    sentiment_counts['date'] = pd.to_datetime(sentiment_counts['date'])
                    news_df = sentiment_counts.set_index('date')
            except Exception as e:
                self.logger.warning(f"Error loading news data for {stock_symbol}: {e}")

        # --- Merge Corporate Actions and News ---
        if corporate_df.empty and news_df.empty:
            return pd.DataFrame()
        elif corporate_df.empty:
            return news_df
        elif news_df.empty:
            return corporate_df
        else:
            # Join both dataframes on their date index
            return corporate_df.join(news_df, how='outer')

    
    def _merge_data_sources(self, price_data: pd.DataFrame, fundamental_data: pd.DataFrame, 
                        sentiment_data: pd.DataFrame, corporate_data: pd.DataFrame) -> pd.DataFrame:
        """Merge all data sources into a single DataFrame."""
        try:
            # Ensure price data is not empty
            if price_data.empty:
                self.logger.warning("Price data is empty, cannot merge.")
                return pd.DataFrame()

            # Start with price data as the base DataFrame
            merged_df = price_data.copy()

            # Merge fundamental data
            if not fundamental_data.empty:
                merged_df = merged_df.join(fundamental_data, how='outer')

            # Merge sentiment data
            if not sentiment_data.empty:
                merged_df = merged_df.join(sentiment_data, how='outer')

            # Merge corporate data
            if not corporate_data.empty:
                merged_df = merged_df.join(corporate_data, how='outer')

            return merged_df

        except Exception as e:
            self.logger.error(f"Error merging data sources: {e}")
            return pd.DataFrame()
    
    def _clean_and_align_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and align data for RL environment"""
        if df.empty:
            self.logger.warning("DataFrame is empty. Returning as is.")
            return df
        
        # Ensure the DataFrame has a valid date index
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.warning("DataFrame index is not a DatetimeIndex. Attempting to convert.")
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                self.logger.error(f"Error converting index to DatetimeIndex: {e}")
                return df
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Forward fill missing values for fundamental data (changes slowly)
        fundamental_cols = [col for col in df.columns if col.startswith('metric_') or 
                            col in ['revenue', 'net_income', 'total_assets', 'total_equity']]
        if fundamental_cols:
            df[fundamental_cols] = df[fundamental_cols].fillna(method='ffill')
        else:
            self.logger.warning("No fundamental columns found in DataFrame.")

        # Forward fill corporate action data
        corporate_cols = [col for col in df.columns if col.startswith(('dividend_', 'bonus_', 'split_', 'news_'))]
        if corporate_cols:
            df[corporate_cols] = df[corporate_cols].fillna(0)
        else:
            self.logger.warning("No corporate action columns found in DataFrame.")

        # Fill remaining NaN values with 0 for sentiment and other metrics
        sentiment_cols = [col for col in df.columns if col.startswith(('reddit_', 'twitter_'))]
        if sentiment_cols:
            df[sentiment_cols] = df[sentiment_cols].fillna(0)
        else:
            self.logger.warning("No sentiment columns found in DataFrame.")

        # Fill remaining NaN values with appropriate defaults
        df = df.fillna(method='ffill').fillna(method='bfill')

        # Ensure all numeric columns are float
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.any():
            df[numeric_cols] = df[numeric_cols].astype(float)

        return df
    
    def consolidate_all_stocks(self) -> Dict[str, pd.DataFrame]:
        """Consolidate data for all stocks"""
        consolidated_data = {}
        
        def process_stock(stock):
            try:
                stock_data = self.consolidate_stock_data(stock)
                if not stock_data.empty:
                    self.logger.info(f"Consolidated {stock}: {len(stock_data)} rows, {len(stock_data.columns)} features")
                    return stock, stock_data
                else:
                    self.logger.warning(f"No data consolidated for {stock}")
                    return stock, pd.DataFrame()
            except Exception as e:
                self.logger.error(f"Error consolidating {stock}: {e}")
                return stock, pd.DataFrame()
        
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_stock, self.stocks)
        
        for stock, data in results:
            consolidated_data[stock] = data
        
        return consolidated_data
    
    def save_consolidated_data(self, consolidated_data: Dict[str, pd.DataFrame], 
                             output_dir: str = "market_data/consolidated_data"):
        """Save consolidated data to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for stock, df in consolidated_data.items():
            if not df.empty:
                file_path = output_path / f"{stock}_consolidated.csv"
                df.to_csv(file_path)
                self.logger.info(f"Saved consolidated data for {stock} to {file_path}")