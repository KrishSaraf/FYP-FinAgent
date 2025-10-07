"""
Data loading utilities for the 45-stock portfolio.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_processed_data(data_path: str = "../../processed_data") -> Tuple[pd.DataFrame, List[str]]:
    """
    Load processed data from the processed_data folder.
    
    Args:
        data_path: Path to the processed_data folder
        
    Returns:
        Tuple of (combined_data, tickers)
    """
    print(f"📊 Loading data from {data_path}...")
    
    # Get list of available CSV files
    csv_files = [f for f in os.listdir(data_path) if f.endswith('_aligned.csv')]
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_path}")
    
    print(f"Found {len(csv_files)} data files")
    
    # Load and combine all data
    all_data = []
    tickers = []
    
    for csv_file in csv_files:
        ticker = csv_file.replace('_aligned.csv', '')
        file_path = os.path.join(data_path, csv_file)
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            # Convert first column to datetime (assuming it's the date column)
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df.set_index(df.columns[0], inplace=True)
            
            # Add ticker column
            df['ticker'] = ticker
            
            # Reset index to make date a column
            df.reset_index(inplace=True)
            df.rename(columns={df.columns[0]: 'date'}, inplace=True)
            
            all_data.append(df)
            tickers.append(ticker)
            
            print(f"  ✅ Loaded {ticker}: {df.shape}")
            
        except Exception as e:
            print(f"  ❌ Failed to load {ticker}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data files could be loaded successfully")
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Sort by date and ticker
    combined_data = combined_data.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    print(f"📈 Combined data shape: {combined_data.shape}")
    print(f"📅 Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
    print(f"🏢 Tickers: {sorted(tickers)}")
    
    return combined_data, sorted(tickers)


def select_top_stocks(data: pd.DataFrame, n_stocks: int = 45) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select top N stocks based on data quality and availability.
    
    Args:
        data: Combined data DataFrame
        n_stocks: Number of stocks to select
        
    Returns:
        Tuple of (filtered_data, selected_tickers)
    """
    print(f"🎯 Selecting top {n_stocks} stocks...")
    
    # Calculate data quality metrics for each ticker
    ticker_stats = []
    
    for ticker in data['ticker'].unique():
        ticker_data = data[data['ticker'] == ticker]
        
        # Calculate metrics
        n_days = len(ticker_data)
        date_range = (ticker_data['date'].max() - ticker_data['date'].min()).days
        completeness = n_days / date_range if date_range > 0 else 0
        
        # Check for missing values in key columns
        key_columns = ['close', 'volume', 'open', 'high', 'low']
        missing_ratio = ticker_data[key_columns].isnull().sum().sum() / (len(ticker_data) * len(key_columns))
        
        # Calculate average volume (liquidity proxy)
        avg_volume = ticker_data['volume'].mean() if 'volume' in ticker_data.columns else 0
        
        ticker_stats.append({
            'ticker': ticker,
            'n_days': n_days,
            'completeness': completeness,
            'missing_ratio': missing_ratio,
            'avg_volume': avg_volume,
            'score': completeness * (1 - missing_ratio) * np.log(1 + avg_volume)
        })
    
    # Sort by score and select top N
    ticker_stats_df = pd.DataFrame(ticker_stats)
    ticker_stats_df = ticker_stats_df.sort_values('score', ascending=False)
    
    selected_tickers = ticker_stats_df.head(n_stocks)['ticker'].tolist()
    
    # Filter data to selected tickers
    filtered_data = data[data['ticker'].isin(selected_tickers)].copy()
    
    print(f"✅ Selected {len(selected_tickers)} stocks:")
    for i, ticker in enumerate(selected_tickers[:10]):  # Show first 10
        stats = ticker_stats_df[ticker_stats_df['ticker'] == ticker].iloc[0]
        print(f"  {i+1:2d}. {ticker}: {stats['n_days']} days, completeness={stats['completeness']:.3f}")
    
    if len(selected_tickers) > 10:
        print(f"  ... and {len(selected_tickers) - 10} more")
    
    return filtered_data, selected_tickers


def validate_data_schema(data: pd.DataFrame) -> bool:
    """
    Validate that data has the required schema.
    
    Args:
        data: DataFrame to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_columns = [
        'date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'vwap', 
        'value_traded', 'total_trades', 'dma_50', 'dma_200'
    ]
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return False
    
    # Check for required fundamental columns (subset)
    fundamental_columns = [
        'metric_Cash', 'metric_TotalAssets', 'metric_TotalEquity', 
        'metric_Revenue', 'metric_NetIncome'
    ]
    
    missing_fundamental = [col for col in fundamental_columns if col not in data.columns]
    if missing_fundamental:
        print(f"⚠️  Missing some fundamental columns: {missing_fundamental}")
        print("   This is okay, but some features may not be available")
    
    print("✅ Data schema validation passed")
    return True


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data for training.
    
    Args:
        data: Raw data DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    print("🔧 Preprocessing data...")
    
    processed_data = data.copy()
    
    # Handle missing values
    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
    processed_data[numeric_columns] = processed_data[numeric_columns].fillna(method='ffill').fillna(0)
    
    # Remove rows with missing essential data
    essential_columns = ['close', 'volume', 'open', 'high', 'low']
    processed_data = processed_data.dropna(subset=essential_columns)
    
    # Ensure positive prices and volumes
    for col in ['close', 'open', 'high', 'low']:
        processed_data = processed_data[processed_data[col] > 0]
    
    processed_data = processed_data[processed_data['volume'] >= 0]
    
    # Sort by date and ticker
    processed_data = processed_data.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    print(f"✅ Preprocessing completed. Final shape: {processed_data.shape}")
    
    return processed_data


def get_feature_columns(data: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns (excluding static columns).
    
    Args:
        data: DataFrame
        
    Returns:
        List of feature column names
    """
    static_columns = ['date', 'ticker', 'period_end_date']
    feature_columns = [col for col in data.columns if col not in static_columns]
    
    print(f"📊 Found {len(feature_columns)} feature columns")
    
    return feature_columns


# For backward compatibility
def generate_synthetic_data(n_stocks: int = 45, n_days: int = 252, start_date: str = '2023-01-01') -> Tuple[pd.DataFrame, List[str]]:
    """
    Load real data instead of generating synthetic data.
    
    Args:
        n_stocks: Number of stocks to select
        n_days: Ignored (uses actual data range)
        start_date: Ignored (uses actual data range)
        
    Returns:
        Tuple of (data, tickers)
    """
    print("🔄 Loading real data instead of synthetic data...")
    
    # Load processed data
    data, all_tickers = load_processed_data()
    
    # Select top stocks
    data, selected_tickers = select_top_stocks(data, n_stocks)
    
    # Validate schema
    validate_data_schema(data)
    
    # Preprocess data
    data = preprocess_data(data)
    
    return data, selected_tickers