"""
Data Integration Module for LLM Enhanced Strategy

This module integrates with the existing JAX portfolio environment data loader
to provide properly formatted data for the LLM enhanced strategy.
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Import existing data loader
from finagent.environment.portfolio_env import JAXPortfolioDataLoader

logger = logging.getLogger(__name__)

class LLMDataIntegrator:
    """Integrates existing JAX data loader with LLM strategy requirements"""
    
    def __init__(self, data_root: str, stocks: List[str], features: Optional[List[str]] = None):
        self.data_root = Path(data_root)
        self.stocks = stocks
        self.features = features
        
        # Initialize the existing JAX data loader
        self.jax_loader = JAXPortfolioDataLoader(
            data_root=str(data_root),
            stocks=stocks,
            features=features,
            use_all_features=True
        )
        
        # Cache for loaded data
        self._data_cache = {}
        
        logger.info(f"Initialized LLM Data Integrator with {len(stocks)} stocks")
    
    def load_market_data(self, start_date: str, end_date: str, 
                        cache_key: Optional[str] = None) -> pd.DataFrame:
        """Load market data compatible with LLM strategy requirements"""
        
        if cache_key and cache_key in self._data_cache:
            logger.info(f"Using cached data for {cache_key}")
            return self._data_cache[cache_key]
        
        try:
            # Load data using existing JAX loader
            data_array, day_indices, valid_dates, n_features = self.jax_loader.load_and_preprocess_data(
                start_date=start_date,
                end_date=end_date,
                preload_to_gpu=False,  # Keep on CPU for pandas processing
                save_cache=True,
                force_reload=False
            )
            
            # Convert JAX array to numpy if needed
            if hasattr(data_array, 'device'):
                data_array = np.array(data_array)
            
            # Get feature names
            feature_names = self.jax_loader.features
            
            # Reshape data for pandas processing
            # data_array shape: (n_days, n_stocks, n_features)
            n_days, n_stocks, n_features = data_array.shape
            
            # Create multi-level column structure
            columns = []
            for stock in self.stocks:
                for feature in feature_names:
                    columns.append(f"{stock}_{feature}")
            
            # Reshape to (n_days, n_stocks * n_features)
            reshaped_data = data_array.reshape(n_days, n_stocks * n_features)
            
            # Create DataFrame
            df = pd.DataFrame(
                data=reshaped_data,
                index=valid_dates,
                columns=columns
            )
            
            # Cache the result
            if cache_key:
                self._data_cache[cache_key] = df
            
            logger.info(f"Loaded market data: {df.shape} for period {start_date} to {end_date}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            # Return empty DataFrame with expected structure
            return self._create_empty_dataframe(start_date, end_date)
    
    def _create_empty_dataframe(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Create empty DataFrame with expected structure"""
        
        # Create date range
        dates = pd.date_range(start_date, end_date, freq='B')  # Business days
        
        # Create columns
        columns = []
        basic_features = ['close', 'returns_1d', 'volatility_20d', 'momentum_5d', 
                         'momentum_20d', 'rsi_14', 'volume_ratio_20d', 'dma_50', 'dma_200']
        
        for stock in self.stocks:
            for feature in basic_features:
                columns.append(f"{stock}_{feature}")
        
        # Create empty DataFrame
        df = pd.DataFrame(index=dates, columns=columns)
        df = df.fillna(0.0)
        
        logger.warning(f"Created empty DataFrame with shape {df.shape}")
        return df
    
    def get_stock_list(self) -> List[str]:
        """Get list of available stocks"""
        return self.stocks.copy()
    
    def get_feature_list(self) -> List[str]:
        """Get list of available features"""
        return self.jax_loader.features.copy() if hasattr(self.jax_loader, 'features') else []
    
    def validate_data_availability(self, start_date: str, end_date: str) -> Dict[str, bool]:
        """Validate data availability for each stock in the date range"""
        
        availability = {}
        
        for stock in self.stocks:
            csv_path = self.data_root / f"{stock}_aligned.csv"
            
            if csv_path.exists():
                try:
                    # Read sample to check date range
                    df = pd.read_csv(csv_path, parse_dates=[0], index_col=0, nrows=1000)
                    
                    if len(df) > 0:
                        # Check if date range overlaps
                        data_start = df.index.min()
                        data_end = df.index.max()
                        
                        req_start = pd.to_datetime(start_date)
                        req_end = pd.to_datetime(end_date)
                        
                        has_overlap = not (req_end < data_start or req_start > data_end)
                        availability[stock] = has_overlap
                    else:
                        availability[stock] = False
                        
                except Exception as e:
                    logger.error(f"Error checking {stock}: {e}")
                    availability[stock] = False
            else:
                availability[stock] = False
        
        available_count = sum(availability.values())
        logger.info(f"Data availability: {available_count}/{len(self.stocks)} stocks available")
        
        return availability
    
    def get_stock_data_for_date(self, market_data: pd.DataFrame, 
                               stock: str, date: datetime) -> Optional[Dict[str, float]]:
        """Extract stock data for a specific date in LLM-compatible format"""
        
        try:
            if date not in market_data.index:
                # Find nearest available date
                available_dates = market_data.index
                nearest_date = min(available_dates, key=lambda x: abs((x - date).days))
                if abs((nearest_date - date).days) > 5:  # More than 5 days away
                    return None
                date = nearest_date
            
            # Extract stock-specific columns
            stock_columns = [col for col in market_data.columns if col.startswith(f"{stock}_")]
            
            if not stock_columns:
                return None
            
            row = market_data.loc[date, stock_columns]
            
            # Convert to standard format
            stock_data = {}
            for col in stock_columns:
                feature = col.replace(f"{stock}_", "")
                stock_data[feature] = float(row[col]) if not pd.isna(row[col]) else 0.0
            
            # Add stock symbol
            stock_data['symbol'] = stock
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error extracting data for {stock} on {date}: {e}")
            return None
    
    def calculate_market_indicators(self, market_data: pd.DataFrame, 
                                  date: datetime) -> Dict[str, float]:
        """Calculate market-wide indicators for LLM context"""
        
        try:
            # Get returns for all stocks
            returns_cols = [col for col in market_data.columns if col.endswith('_returns_1d')]
            
            if date not in market_data.index or not returns_cols:
                return {'market_return': 0.0, 'market_volatility': 0.2, 'market_sentiment': 'Neutral'}
            
            returns = market_data.loc[date, returns_cols]
            
            # Calculate market metrics
            market_return = returns.mean()
            market_volatility = returns.std()
            
            # Simple sentiment based on return distribution
            positive_returns = (returns > 0).sum()
            total_returns = len(returns)
            positive_ratio = positive_returns / total_returns if total_returns > 0 else 0.5
            
            if positive_ratio > 0.6:
                sentiment = 'Positive'
            elif positive_ratio < 0.4:
                sentiment = 'Negative'
            else:
                sentiment = 'Neutral'
            
            return {
                'market_return': float(market_return),
                'market_volatility': float(market_volatility),
                'market_sentiment': sentiment,
                'positive_ratio': float(positive_ratio)
            }
            
        except Exception as e:
            logger.error(f"Error calculating market indicators: {e}")
            return {'market_return': 0.0, 'market_volatility': 0.2, 'market_sentiment': 'Neutral'}
    
    def prepare_rolling_windows(self, start_date: str, end_date: str, 
                              window_days: int = 90) -> List[Tuple[str, str]]:
        """Prepare rolling window date ranges for backtesting"""
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        windows = []
        current_date = start_dt
        
        while current_date < end_dt:
            window_end = min(current_date + timedelta(days=window_days), end_dt)
            
            windows.append((
                current_date.strftime('%Y-%m-%d'),
                window_end.strftime('%Y-%m-%d')
            ))
            
            # Move forward by a smaller step to create overlapping windows
            current_date += timedelta(days=30)  # 30-day step for overlapping windows
        
        logger.info(f"Created {len(windows)} rolling windows")
        return windows
    
    def get_data_statistics(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive statistics about the loaded data"""
        
        stats = {
            'shape': market_data.shape,
            'date_range': (market_data.index.min(), market_data.index.max()),
            'n_stocks': len(self.stocks),
            'n_features_per_stock': len([col for col in market_data.columns 
                                       if col.startswith(f"{self.stocks[0]}_")]) if self.stocks else 0,
            'missing_data_pct': (market_data.isnull().sum().sum() / market_data.size) * 100,
            'stocks_with_data': []
        }
        
        # Check which stocks have data
        for stock in self.stocks:
            stock_cols = [col for col in market_data.columns if col.startswith(f"{stock}_")]
            if stock_cols:
                non_null_pct = (market_data[stock_cols].count().sum() / 
                              (len(market_data) * len(stock_cols))) * 100
                if non_null_pct > 50:  # At least 50% data available
                    stats['stocks_with_data'].append(stock)
        
        logger.info(f"Data statistics: {stats}")
        return stats

# Example usage and testing
def test_data_integration():
    """Test the data integration module"""
    
    # Get stock list from file
    stocks_file = Path("finagent/stocks.txt")
    if stocks_file.exists():
        with open(stocks_file, 'r') as f:
            stocks = [line.strip() for line in f.readlines() if line.strip()][:10]  # First 10 stocks
    else:
        stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
    
    # Initialize integrator
    integrator = LLMDataIntegrator(
        data_root="processed_data/",
        stocks=stocks
    )
    
    # Test data availability
    print("Testing data availability...")
    availability = integrator.validate_data_availability("2024-01-01", "2024-12-31")
    print(f"Available stocks: {sum(availability.values())}/{len(stocks)}")
    
    # Load sample data
    print("\nLoading sample data...")
    try:
        market_data = integrator.load_market_data("2024-06-01", "2024-09-01")
        print(f"Loaded data shape: {market_data.shape}")
        
        # Get statistics
        stats = integrator.get_data_statistics(market_data)
        print(f"Data statistics: {stats}")
        
        # Test stock data extraction
        test_date = market_data.index[0]
        stock_data = integrator.get_stock_data_for_date(market_data, stocks[0], test_date)
        print(f"Sample stock data for {stocks[0]}: {list(stock_data.keys()) if stock_data else 'No data'}")
        
        print("Data integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in data integration test: {e}")
        return False

if __name__ == "__main__":
    test_data_integration()