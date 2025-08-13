import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
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
        """Load fundamental data from various sources"""
        stock_path = self.cleaned_data_path / stock_symbol
        
        # Load key metrics
        metrics_files = list(stock_path.glob("KeyMetrics_*.csv"))
        all_metrics_dict = {}
        # Iterate over all files that match the pattern to consolidate all available metrics.
        for metrics_file in metrics_files:
            try:
                df = pd.read_csv(metrics_file)

                # Validate expected columns for the current file.
                if 'key' not in df.columns or 'value' not in df.columns:
                    self.logger.warning(f"Metrics file {metrics_file} is missing 'key' or 'value' columns. Skipping.")
                    continue

                # Drop rows where key or value is missing to avoid errors.
                df.dropna(subset=['key', 'value'], inplace=True)

                # Pivot the current file's data into a dictionary.
                current_metrics = pd.Series(df.value.values, index=df.key).to_dict()
                
                # Update the master dictionary with the data from the current file.
                # This will add new keys and overwrite existing ones with newer values.
                all_metrics_dict.update(current_metrics)

            except Exception as e:
                self.logger.warning(f"Could not process metrics file {metrics_file}: {e}")
                continue
        
        # Load financial data (latest available)
        financial_files = list(stock_path.glob("Financials_*2025_Annual.csv"))
    
        # Use a temporary dict to collect all financial data from multiple files
        financial_data_dict = {}
        for file_path in financial_files:
            try:
                df = pd.read_csv(file_path)
                # Check for the key-value structure from your screenshot
                if 'key' in df.columns and 'value' in df.columns:
                    # Create a mapping of financial metric keys to their values for easy lookup
                    financial_series = df.set_index('key')['value']
                    
                    # Define the keys you want to extract from the files
                    keys_to_extract = {
                        'revenue': 'Revenue',
                        'net_income': 'NetIncome',
                        'gross_profit': 'GrossProfit',
                        'operating_income': 'OperatingIncome',
                        'total_assets': 'TotalAssets',
                        'total_equity': 'TotalEquity',
                        'total_debt': 'TotalDebt',
                        'net_cash_change': 'NetChangeinCash'
                    }
                    
                    # Extract data for each key if it exists in the file
                    for friendly_name, file_key in keys_to_extract.items():
                        if file_key in financial_series.index:
                            financial_data_dict[friendly_name] = financial_series.get(file_key)
                else:
                    self.logger.warning(f"Financials file {file_path} is missing 'key' or 'value' columns.")
            except Exception as e:
                self.logger.warning(f"Error loading financials {file_path}: {e}")
                continue
                
        # Add the successfully loaded financial data to the main dictionary
        all_metrics_dict.update(financial_data_dict)
        
        if all_metrics_dict:
            # Combine all metrics into a single DataFrame
            all_metrics_df = pd.DataFrame([all_metrics_dict])
            all_metrics_df.columns = [f"metric_{col}" for col in all_metrics_df.columns]
            return all_metrics_df
        return pd.DataFrame()
    
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
                    df['date'] = pd.to_datetime(df['created_at'], errors='coerce').dt.date
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
        
    def _analyze_sentiment(self, texts: List[str]) -> List[str]:
        """Analyze sentiment of text data using FinBERT"""
        try:
            # Tokenize the text data
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get probabilities and predicted labels
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            
            # Map label indices to sentiment labels
            labels = ["positive", "negative", "neutral"]
            return [labels[i] for i in predicted_labels]
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return ["neutral"] * len(texts)  # Default to neutral if error occurs
    
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
            df[corporate_cols] = df[corporate_cols].fillna(method='ffill')
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