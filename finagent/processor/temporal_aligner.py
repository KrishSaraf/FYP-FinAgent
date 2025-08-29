import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
from finagent.registry import PROCESSOR

@PROCESSOR.register_module()
class TemporalDataAligner:
    """
    Aligns market data from multiple sources with consistent timestamps
    and handles different update frequencies for RL environment.
    """
    
    def __init__(self, market: str = "NSE"):
        self.market = market
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Initialize market calendar
        self.market_calendar = self._init_market_calendar()
        
        # Define trading hours for Indian market
        self.trading_start = "09:15"
        self.trading_end = "15:30"
        self.timezone = "Asia/Kolkata"
    
    def _init_market_calendar(self):
        """Initialize market calendar for Indian market"""
        try:
            if self.market == "NSE":
                return mcal.get_calendar('NSE')
            elif self.market == "BSE":
                return mcal.get_calendar('BSE')
            else:
                return mcal.get_calendar('NSE')  # Default to NSE
        except Exception as e:
            self.logger.warning(f"Could not initialize market calendar: {e}")
            return None
    
    def align_to_trading_days(self, df: pd.DataFrame, start_date: Optional[str] = None, 
                             end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Align data to trading days only.
        
        Args:
            df: DataFrame with datetime index
            start_date: Start date for alignment (YYYY-MM-DD)
            end_date: End date for alignment (YYYY-MM-DD)
            
        Returns:
            DataFrame aligned to trading days
        """
        if df.empty:
            self.logger.warning("Input DataFrame is empty.")
            return df
        
        try:
            # Convert dates if provided
            if start_date:
                start_date = pd.to_datetime(start_date)
            if end_date:
                end_date = pd.to_datetime(end_date)
            
            # Get trading days
            trading_days = self._get_trading_days(start_date, end_date)
            
            if trading_days is None:
                self.logger.warning("Could not get trading days, returning original data")
                return df
            
            # Reindex to trading days
            aligned_df = df.reindex(trading_days)
            
            # Forward fill missing values for fundamental data
            fundamental_cols = [col for col in df.columns if col.startswith('metric_') or 
                              col in ['revenue', 'net_income', 'total_assets', 'total_equity']]
            aligned_df[fundamental_cols] = aligned_df[fundamental_cols].fillna(method='ffill')
            
            # Forward fill corporate action data
            corporate_cols = [col for col in df.columns if col.startswith(('dividend_', 'bonus_', 'split_',))]
            aligned_df[corporate_cols] = aligned_df[corporate_cols].fillna(method='ffill')
            
            # Fill remaining NaN values
            aligned_df = aligned_df.fillna(method='ffill').fillna(method='bfill')
            
            return aligned_df
        
        except Exception as e:
            self.logger.error(f"Error aligning to trading days: {e}")
            return df
    
    def _get_trading_days(self, start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None) -> Optional[pd.DatetimeIndex]:
        """Get trading days for the specified period"""
        try:
            if self.market_calendar is None:
                return None
            
            # Use provided dates or default to last 2 years
            if start_date is None:
                start_date = datetime(2024, 6, 6)
            if end_date is None:
                end_date = datetime(2025, 6, 6)
            
            # Get trading schedule
            schedule = self.market_calendar.schedule(start_date=start_date, end_date=end_date)
            
            # Extract trading days
            trading_days = schedule.index.date
            trading_days = pd.to_datetime(trading_days)
            
            return trading_days
            
        except Exception as e:
            self.logger.error(f"Error getting trading days: {e}")
            return None
    
    def resample_to_daily(self, df: pd.DataFrame, method: str = 'last') -> pd.DataFrame:
        """
        Resample data to daily frequency.
        
        Args:
            df: DataFrame with datetime index
            method: Resampling method ('last', 'first', 'mean', 'sum')
            
        Returns:
            Daily resampled DataFrame
        """
        if df.empty:
            return df
        
        # Group by date (remove time component)
        df_daily = df.copy()
        df_daily.index = df_daily.index.date
        df_daily.index = pd.to_datetime(df_daily.index)
        
        # Apply appropriate aggregation method for each column
        resampled_data = {}
        
        for col in df_daily.columns:
            if col.startswith(('metric_', 'revenue_', 'net_income_', 'total_assets_', 'total_equity_')):
                # Fundamental data: use last value
                resampled_data[col] = df_daily[col].groupby(df_daily.index).last()
            elif col.startswith(('dividend_', 'bonus_', 'split_', 'news_')):
                # Corporate actions: use last value
                resampled_data[col] = df_daily[col].groupby(df_daily.index).last()
            elif col.startswith(('reddit_', 'twitter_')):
                # Social media: use sum for counts, mean for scores
                if 'count' in col or 'sum' in col:
                    resampled_data[col] = df_daily[col].groupby(df_daily.index).sum()
                else:
                    resampled_data[col] = df_daily[col].groupby(df_daily.index).mean()
            else:
                # Price/volume data: use last value
                resampled_data[col] = df_daily[col].groupby(df_daily.index).last()
        
        resampled_df = pd.DataFrame(resampled_data)
        
        # Sort by date
        resampled_df.sort_index(inplace=True)
        
        return resampled_df
    
    def create_lagged_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
        """
        Create lagged features for time series analysis.
        
        Args:
            df: DataFrame with datetime index
            lags: List of lag periods to create
            
        Returns:
            DataFrame with lagged features
        """
        if df.empty:
            return df
        
        df_with_lags = df.copy()
        
        # Create lagged features for price and volume data
        price_volume_cols = ['price', 'volume', 'dma50', 'dma200']
        
        for col in price_volume_cols:
            if col in df.columns:
                for lag in lags:
                    df_with_lags[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Create rolling statistics
        for col in price_volume_cols:
            if col in df.columns:
                # Rolling mean
                df_with_lags[f'{col}_rolling_mean_5'] = df[col].rolling(window=5).mean()
                df_with_lags[f'{col}_rolling_mean_20'] = df[col].rolling(window=20).mean()
                
                # Rolling std
                df_with_lags[f'{col}_rolling_std_20'] = df[col].rolling(window=20).std()
                
                # Price momentum
                if col == 'price':
                    df_with_lags[f'{col}_momentum_5'] = df[col] / df[col].shift(5) - 1
                    df_with_lags[f'{col}_momentum_20'] = df[col] / df[col].shift(20) - 1
        
        # Create technical indicators
        if 'price' in df.columns and 'volume' in df.columns:
            # RSI-like indicator
            price_diff = df['price'].diff()
            gain = price_diff.where(price_diff > 0, 0)
            loss = -price_diff.where(price_diff < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            df_with_lags['rsi_14'] = rsi
            
            # Volume-price trend
            df_with_lags['volume_price_trend'] = (df['price'] - df['price'].shift(1)) * df['volume']
        
        # Create cross-sectional features
        if 'dma50' in df.columns and 'dma200' in df.columns:
            df_with_lags['dma_cross'] = (df['dma50'] > df['dma200']).astype(int)
            df_with_lags['dma_distance'] = (df['dma50'] - df['dma200']) / df['dma200']
        
        return df_with_lags
    
    def align_multiple_stocks(self, stock_data: Dict[str, pd.DataFrame], 
                            start_date: Optional[str] = None, 
                            end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Align multiple stocks to the same time period and trading days.
        
        Args:
            stock_data: Dictionary of stock DataFrames
            start_date: Start date for alignment
            end_date: End date for alignment
            
        Returns:
            Dictionary of aligned stock DataFrames
        """
        aligned_data = {}
        
        # Get common trading days
        trading_days = self._get_trading_days(
            pd.to_datetime(start_date) if start_date else None,
            pd.to_datetime(end_date) if end_date else None
        )
        
        if trading_days is None:
            self.logger.warning("Could not get trading days, returning original data")
            return stock_data
        
        # Align each stock
        for stock, df in stock_data.items():
            try:
                if not df.empty:
                    # Resample to daily
                    daily_df = self.resample_to_daily(df)
                    
                    # Align to trading days
                    aligned_df = daily_df.reindex(trading_days)
                    
                    # Fill missing values
                    aligned_df = aligned_df.fillna(method='ffill').fillna(method='bfill')
                    
                    # Create lagged features
                    aligned_df = self.create_lagged_features(aligned_df)
                    
                    aligned_data[stock] = aligned_df
                    self.logger.info(f"Aligned {stock}: {len(aligned_df)} rows")
                else:
                    self.logger.warning(f"Empty data for {stock}")
            except Exception as e:
                self.logger.error(f"Error aligning {stock}: {e}")
                continue
        
        return aligned_data
    
    def create_market_features(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create market-wide features from individual stock data.
        
        Args:
            stock_data: Dictionary of aligned stock DataFrames
            
        Returns:
            DataFrame with market-wide features
        """
        market_features = {}
        
        # Get common dates
        common_dates = None
        for stock, df in stock_data.items():
            if not df.empty:
                if common_dates is None:
                    common_dates = set(df.index)
                else:
                    common_dates = common_dates.intersection(set(df.index))
        
        if not common_dates:
            return pd.DataFrame()
        
        common_dates = sorted(list(common_dates))
        
        # Create market-wide features
        for date in common_dates:
            market_features[date] = {}
            
            # Price-based features
            prices = [stock_data[stock].loc[date, 'price'] for stock in stock_data 
                     if 'price' in stock_data[stock].columns and np.all(pd.notna(stock_data[stock].loc[date, 'price']))]
            
            if prices:
                flat_prices = []
                for item in prices:
                    if isinstance(item, pd.Series):
                        flat_prices.extend(item.tolist())
                    else:
                        flat_prices.append(item)
            if flat_prices:
                market_features[date]['market_price_mean'] = np.mean(flat_prices)
                market_features[date]['market_price_std'] = np.std(flat_prices)
                market_features[date]['market_price_change'] = np.mean([p - flat_prices[0] for p in flat_prices[1:]]) if len(flat_prices) > 1 else 0
            
            # Volume-based features
            volumes = [stock_data[stock].loc[date, 'volume'] for stock in stock_data 
                      if 'volume' in stock_data[stock].columns and np.all(pd.notna(stock_data[stock].loc[date, 'volume']))]
            
            if volumes:
                flat_volumes = []
                for item in volumes:
                    if isinstance(item, pd.Series):
                        flat_volumes.extend(item.tolist())
                    else:
                        flat_volumes.append(item)
                market_features[date]['market_volume_mean'] = np.mean(flat_volumes)
                market_features[date]['market_volume_std'] = np.std(flat_volumes)
            
            # Sentiment-based features
            reddit_scores = [stock_data[stock].loc[date, 'reddit_score_mean'] for stock in stock_data 
                           if 'reddit_score_mean' in stock_data[stock].columns and np.all(pd.notna(stock_data[stock].loc[date, 'reddit_score_mean']))]
            
            if reddit_scores:
                flat_reddit_scores = []
                for item in reddit_scores:
                    if isinstance(item, pd.Series):
                        flat_reddit_scores.extend(item.tolist())
                    else:
                        flat_reddit_scores.append(item)
                market_features[date]['market_reddit_sentiment'] = np.mean(flat_reddit_scores)
        
        market_df = pd.DataFrame.from_dict(market_features, orient='index')
        market_df.index = pd.to_datetime(market_df.index)
        market_df.sort_index(inplace=True)
        
        return market_df
    
    def save_aligned_data(self, aligned_data: Dict[str, pd.DataFrame], 
                         market_features: pd.DataFrame,
                         output_dir: str = "market_data/aligned_data"):
        """Save aligned data to CSV files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual stock data
        for stock, df in aligned_data.items():
            if not df.empty:
                file_path = output_path / f"{stock}_aligned.csv"
                df.to_csv(file_path)
                self.logger.info(f"Saved aligned data for {stock} to {file_path}")
        
        # Save market features
        if not market_features.empty:
            market_file = output_path / "market_features.csv"
            market_features.to_csv(market_file)
            self.logger.info(f"Saved market features to {market_file}")