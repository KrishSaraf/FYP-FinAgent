"""
Data Loading and Preprocessing Module for FinRL
Handles loading and preprocessing of the rich financial dataset
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class FinancialDataLoader:
    """
    Loads and preprocesses financial data for FinRL training
    """
    
    def __init__(self, data_path: str = "processed_data"):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
        # Define core features for FinRL state space
        self.core_features = [
            'open', 'high', 'low', 'close', 'volume', 'vwap'
        ]
        
        # Technical indicators
        self.technical_features = [
            'dma_50', 'dma_200', 'rsi_14', 'dma_cross', 'dma_distance'
        ]
        
        # Key fundamental metrics (top performing ones)
        self.fundamental_features = [
            'metric_pPerEExcludingExtraordinaryItemsMostRecentFiscalYear',
            'metric_priceToBookMostRecentFiscalYear', 
            'metric_returnOnAverageEquityTrailing12Month',
            'metric_returnOnInvestmentTrailing12Month',
            'metric_operatingMarginTrailing12Month',
            'metric_grossMarginTrailing12Month',
            'metric_netProfitMarginPercentTrailing12Month',
            'metric_currentRatioMostRecentFiscalYear',
            'metric_quickRatioMostRecentFiscalYear',
            'metric_totalDebtPerTotalEquityMostRecentFiscalYear',
            'metric_freeCashFlowtrailing12Month',
            'metric_revenueGrowthRate5Year',
            'metric_ePSGrowthRate5Year',
            'metric_marketCap',
            'metric_beta'
        ]
        
        # Sentiment features
        self.sentiment_features = [
            'reddit_title_sentiments_mean',
            'reddit_body_sentiments', 
            'news_sentiment_mean',
            'reddit_posts_count',
            'news_articles_count'
        ]
        
        # Selected lag features (most important ones)
        self.lag_features = [
            'close_lag_1', 'close_lag_5', 'close_lag_10',
            'volume_lag_1', 'volume_lag_5',
            'dma_50_lag_1', 'dma_50_lag_5'
        ]
        
        # Rolling statistics
        self.rolling_features = [
            'close_rolling_mean_5', 'close_rolling_mean_20',
            'close_rolling_std_20', 'close_momentum_5',
            'volume_rolling_mean_5', 'volume_rolling_mean_20'
        ]
        
        # Combine all features
        self.all_features = (self.core_features + self.technical_features + 
                           self.fundamental_features + self.sentiment_features + 
                           self.lag_features + self.rolling_features)
    
    def load_stock_data(self, stock_symbol: str) -> pd.DataFrame:
        """
        Load data for a specific stock
        
        Args:
            stock_symbol: Stock symbol (e.g., 'RELIANCE')
            
        Returns:
            DataFrame with stock data
        """
        file_path = os.path.join(self.data_path, f"{stock_symbol}_aligned.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Convert date column to datetime
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
        
        return df
    
    def get_available_stocks(self) -> List[str]:
        """
        Get list of available stock symbols
        
        Returns:
            List of stock symbols
        """
        files = [f for f in os.listdir(self.data_path) if f.endswith('_aligned.csv')]
        return [f.replace('_aligned.csv', '') for f in files]
    
    def preprocess_data(self, df: pd.DataFrame, 
                       features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Preprocess the data for FinRL training
        
        Args:
            df: Raw stock data
            features: List of features to use (if None, uses all_features)
            
        Returns:
            Preprocessed DataFrame
        """
        if features is None:
            features = self.all_features
        
        # Select only available features
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
        
        # Create processed dataframe
        processed_df = df[available_features].copy()
        
        # Handle missing values
        processed_df = processed_df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any remaining NaN values
        processed_df = processed_df.dropna()
        
        # Add derived features
        processed_df = self._add_derived_features(processed_df)
        
        return processed_df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for better model performance
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()
        
        # Price-based features
        if 'close' in df.columns:
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['price_change'] = df['close'] - df['close'].shift(1)
            df['price_change_pct'] = df['price_change'] / df['close'].shift(1)
        
        # Volume-based features
        if 'volume' in df.columns:
            df['volume_change'] = df['volume'] - df['volume'].shift(1)
            df['volume_change_pct'] = df['volume_change'] / df['volume'].shift(1)
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Technical indicators
        if 'close' in df.columns and 'dma_50' in df.columns:
            df['price_vs_dma50'] = (df['close'] - df['dma_50']) / df['dma_50']
        
        if 'close' in df.columns and 'dma_200' in df.columns:
            df['price_vs_dma200'] = (df['close'] - df['dma_200']) / df['dma_200']
        
        # RSI-based features
        if 'rsi_14' in df.columns:
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        
        # Sentiment-based features
        if 'reddit_title_sentiments_mean' in df.columns:
            df['sentiment_positive'] = (df['reddit_title_sentiments_mean'] > 0.1).astype(int)
            df['sentiment_negative'] = (df['reddit_title_sentiments_mean'] < -0.1).astype(int)
        
        return df
    
    def create_train_test_split(self, df: pd.DataFrame, 
                               train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets
        
        Args:
            df: Input DataFrame
            train_ratio: Ratio of data for training
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        return train_df, test_df
    
    def normalize_features(self, train_df: pd.DataFrame, 
                          test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize features using training data statistics
        
        Args:
            train_df: Training data
            test_df: Testing data
            
        Returns:
            Tuple of normalized (train_df, test_df)
        """
        # Get numeric columns only
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        
        # Fit scaler on training data
        train_normalized = train_df.copy()
        test_normalized = test_df.copy()
        
        train_normalized[numeric_cols] = self.feature_scaler.fit_transform(train_df[numeric_cols])
        test_normalized[numeric_cols] = self.feature_scaler.transform(test_df[numeric_cols])
        
        return train_normalized, test_normalized
    
    def get_feature_importance_ranking(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Get feature importance ranking based on correlation with returns
        
        Args:
            df: Input DataFrame with features and returns
            
        Returns:
            Dictionary of feature importance scores
        """
        if 'returns' not in df.columns:
            df = self._add_derived_features(df)
        
        # Calculate correlation with returns
        correlations = df.corr()['returns'].abs().sort_values(ascending=False)
        
        # Remove returns itself and NaN values
        correlations = correlations.drop('returns', errors='ignore').dropna()
        
        return correlations.to_dict()

# Example usage
if __name__ == "__main__":
    # Initialize data loader
    loader = FinancialDataLoader()
    
    # Load RELIANCE data
    print("Loading RELIANCE data...")
    reliance_data = loader.load_stock_data("RELIANCE")
    print(f"Loaded data shape: {reliance_data.shape}")
    
    # Preprocess data
    print("Preprocessing data...")
    processed_data = loader.preprocess_data(reliance_data)
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Available features: {len(processed_data.columns)}")
    
    # Get feature importance
    feature_importance = loader.get_feature_importance_ranking(processed_data)
    print("\nTop 10 most important features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Create train/test split
    train_data, test_data = loader.create_train_test_split(processed_data)
    print(f"\nTrain data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Normalize features
    train_norm, test_norm = loader.normalize_features(train_data, test_data)
    print(f"Normalized train data shape: {train_norm.shape}")
    print(f"Normalized test data shape: {test_norm.shape}")
