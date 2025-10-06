"""
Data Utilities for FinRL Integration
Handles data preparation and preprocessing for Indian stock market data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os
import glob
import warnings
warnings.filterwarnings('ignore')

class IndianStockDataProcessor:
    """
    Data processor for Indian stock market data
    """
    
    def __init__(self, data_path: str = "processed_data"):
        self.data_path = data_path
        self.stock_data = {}
        self.combined_data = None
    
    def load_stock_data(self, stock_symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Load stock data from CSV files - loads ALL available stocks by default"""
        if stock_symbols is None:
            # Load ALL available stocks
            csv_files = glob.glob(os.path.join(self.data_path, "*_aligned.csv"))
            stock_symbols = [os.path.basename(f).replace("_aligned.csv", "") for f in csv_files]
            print(f"🚀 Loading ALL {len(stock_symbols)} available stocks...")
        
        for symbol in stock_symbols:
            file_path = os.path.join(self.data_path, f"{symbol}_aligned.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Handle the unnamed first column (date)
                    if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
                        df.columns = ['date'] + list(df.columns[1:])
                    
                    # Ensure date column is datetime
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        # Don't set as index yet, keep as column for combining
                    
                    # Add stock symbol column
                    df['tic'] = symbol
                    self.stock_data[symbol] = df
                    print(f"Loaded data for {symbol}: {len(df)} records")
                except Exception as e:
                    print(f"Error loading {symbol}: {e}")
            else:
                print(f"File not found: {file_path}")
        
        print(f"✅ Successfully loaded {len(self.stock_data)} stocks")
        return self.stock_data
    
    def combine_stock_data(self, stock_symbols: List[str] = None) -> pd.DataFrame:
        """Combine all stock data into a single DataFrame"""
        if stock_symbols is None:
            stock_symbols = list(self.stock_data.keys())
        
        combined_data = []
        for symbol in stock_symbols:
            if symbol in self.stock_data:
                df = self.stock_data[symbol].copy()
                df['tic'] = symbol
                combined_data.append(df)
        
        if combined_data:
            self.combined_data = pd.concat(combined_data, ignore_index=True)
            # Sort by date and stock symbol
            self.combined_data = self.combined_data.sort_values(['date', 'tic'])
            print(f"Combined data shape: {self.combined_data.shape}")
        
        return self.combined_data
    
    def prepare_finrl_data(
        self, 
        stock_symbols: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        feature_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Prepare data in FinRL format
        
        Args:
            stock_symbols: List of stock symbols to include
            start_date: Start date for data filtering
            end_date: End date for data filtering
            feature_columns: List of feature columns to include
        
        Returns:
            DataFrame in FinRL format
        """
        if self.combined_data is None:
            self.combine_stock_data(stock_symbols)
        
        if self.combined_data is None:
            raise ValueError("No data available. Please load stock data first.")
        
        # Filter by date range
        if start_date:
            self.combined_data = self.combined_data[self.combined_data['date'] >= start_date]
        if end_date:
            self.combined_data = self.combined_data[self.combined_data['date'] <= end_date]
        
        # Select feature columns
        if feature_columns is None:
            # Default feature columns for FinRL
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'dma_50', 'dma_200', 'rsi_14',
                'volatility_20d', 'momentum_20d'
            ]
        
        # Ensure required columns exist
        required_columns = ['date', 'tic', 'open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in required_columns if col in self.combined_data.columns]
        
        # Add available feature columns
        for col in feature_columns:
            if col in self.combined_data.columns and col not in available_columns:
                available_columns.append(col)
        
        # Create FinRL-formatted data
        finrl_data = self.combined_data[available_columns].copy()
        
        # Handle missing values
        finrl_data = finrl_data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows with any remaining NaN values
        finrl_data = finrl_data.dropna()
        
        print(f"FinRL data prepared: {finrl_data.shape}")
        print(f"Date range: {finrl_data['date'].min()} to {finrl_data['date'].max()}")
        print(f"Stocks: {finrl_data['tic'].nunique()}")
        
        return finrl_data
    
    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional technical indicators"""
        df = df.copy()
        
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['open']
        df['price_range'] = df['high'] - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        
        # Volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['price_volume'] = df['close'] * df['volume']
        
        # Volatility indicators
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=60).mean()
        
        # Momentum indicators
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Moving averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        
        # Moving average crossovers
        df['ma_cross_5_20'] = (df['ma_5'] > df['ma_20']).astype(int)
        df['ma_cross_10_20'] = (df['ma_10'] > df['ma_20']).astype(int)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market-wide features"""
        df = df.copy()
        
        # Market cap weighted features (simplified)
        df['market_cap'] = df['close'] * df.get('metric_TotalCommonSharesOutstanding', 1000000)
        
        # Sector/industry features (placeholder - would need actual sector data)
        df['sector_tech'] = (df['tic'].str.contains('TECH|INFY|TCS|HCL', case=False)).astype(int)
        df['sector_bank'] = (df['tic'].str.contains('BANK|HDFC|ICICI|SBIN', case=False)).astype(int)
        df['sector_energy'] = (df['tic'].str.contains('RELIANCE|ONGC|BPCL', case=False)).astype(int)
        
        # Market regime indicators
        df['high_vol_regime'] = (df['volatility'] > df['volatility'].rolling(window=60).quantile(0.8)).astype(int)
        df['low_vol_regime'] = (df['volatility'] < df['volatility'].rolling(window=60).quantile(0.2)).astype(int)
        
        return df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            test_ratio: Ratio for test data
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Sort by date
        df = df.sort_values('date')
        
        # Get unique dates
        dates = df['date'].unique()
        n_dates = len(dates)
        
        # Calculate split indices
        train_end = int(n_dates * train_ratio)
        val_end = int(n_dates * (train_ratio + val_ratio))
        
        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        test_dates = dates[val_end:]
        
        # Split data
        train_df = df[df['date'].isin(train_dates)]
        val_df = df[df['date'].isin(val_dates)]
        test_df = df[df['date'].isin(test_dates)]
        
        print(f"Data split:")
        print(f"  Train: {len(train_df)} records ({len(train_dates)} dates)")
        print(f"  Validation: {len(val_df)} records ({len(val_dates)} dates)")
        print(f"  Test: {len(test_df)} records ({len(test_dates)} dates)")
        
        return train_df, val_df, test_df
    
    def get_stock_list(self) -> List[str]:
        """Get list of available stock symbols"""
        return list(self.stock_data.keys())
    
    def get_data_summary(self) -> pd.DataFrame:
        """Get summary statistics of loaded data"""
        if not self.stock_data:
            return pd.DataFrame()
        
        summary_data = []
        for symbol, df in self.stock_data.items():
            summary_data.append({
                'Symbol': symbol,
                'Records': len(df),
                'Start_Date': df.index.min() if hasattr(df.index, 'min') else df['date'].min(),
                'End_Date': df.index.max() if hasattr(df.index, 'max') else df['date'].max(),
                'Columns': len(df.columns),
                'Missing_Values': df.isnull().sum().sum()
            })
        
        return pd.DataFrame(summary_data)


def create_sample_data(n_stocks: int = 5, n_days: int = 1000) -> pd.DataFrame:
    """
    Create sample data for testing purposes
    
    Args:
        n_stocks: Number of stocks to generate
        n_days: Number of days of data
    
    Returns:
        Sample DataFrame in FinRL format
    """
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate stock symbols
    stock_symbols = [f'STOCK_{i:02d}' for i in range(n_stocks)]
    
    data = []
    for symbol in stock_symbols:
        # Generate price data with random walk
        initial_price = np.random.uniform(100, 1000)
        returns = np.random.normal(0.001, 0.02, n_days)  # 0.1% daily return, 2% volatility
        prices = [initial_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            volatility = np.random.uniform(0.01, 0.03)
            high = close * (1 + np.random.uniform(0, volatility))
            low = close * (1 - np.random.uniform(0, volatility))
            open_price = close * (1 + np.random.uniform(-volatility/2, volatility/2))
            volume = np.random.uniform(100000, 1000000)
            
            data.append({
                'date': date,
                'tic': symbol,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'dma_50': close * np.random.uniform(0.95, 1.05),
                'dma_200': close * np.random.uniform(0.9, 1.1),
                'rsi_14': np.random.uniform(20, 80),
                'volatility_20d': np.random.uniform(0.01, 0.05),
                'momentum_20d': np.random.uniform(-0.1, 0.1)
            })
    
    df = pd.DataFrame(data)
    return df


def load_feature_categories(file_path: str = "feature_categories.json") -> Dict[str, List[str]]:
    """
    Load feature categories from JSON file
    
    Args:
        file_path: Path to feature categories JSON file
    
    Returns:
        Dictionary of feature categories
    """
    import json
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('feature_categories', {})
    except FileNotFoundError:
        print(f"Feature categories file not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error parsing JSON file: {file_path}")
        return {}


def select_features_by_category(
    df: pd.DataFrame, 
    categories: List[str],
    feature_categories: Dict[str, List[str]] = None
) -> List[str]:
    """
    Select features by category
    
    Args:
        df: Input DataFrame
        categories: List of categories to include
        feature_categories: Feature categories dictionary
    
    Returns:
        List of selected feature columns
    """
    if feature_categories is None:
        feature_categories = load_feature_categories()
    
    selected_features = []
    for category in categories:
        if category in feature_categories:
            features = feature_categories[category]['features']
            # Only include features that exist in the DataFrame
            available_features = [f for f in features if f in df.columns]
            selected_features.extend(available_features)
    
    # Remove duplicates while preserving order
    selected_features = list(dict.fromkeys(selected_features))
    
    return selected_features
