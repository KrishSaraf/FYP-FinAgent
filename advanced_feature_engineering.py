"""
Advanced Feature Engineering for FinRL
Implements sophisticated feature selection and engineering for >10% returns
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for maximum trading performance
    """
    
    def __init__(self, data_path: str = "processed_data"):
        self.data_path = data_path
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_importance_scores = {}
        self.selected_features = []
        
        # Define feature categories for systematic engineering
        self.price_features = ['open', 'high', 'low', 'close', 'vwap']
        self.volume_features = ['volume', 'value_traded', 'total_trades']
        self.technical_features = ['dma_50', 'dma_200', 'rsi_14', 'dma_cross', 'dma_distance']
        
        # Advanced technical indicators
        self.advanced_technical = [
            'bollinger_upper', 'bollinger_lower', 'bollinger_width',
            'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d', 'williams_r',
            'cci', 'atr', 'adx', 'obv'
        ]
        
        # Momentum features
        self.momentum_features = [
            'close_momentum_5', 'close_momentum_10', 'close_momentum_20',
            'volume_momentum_5', 'volume_momentum_10',
            'price_acceleration', 'volume_acceleration'
        ]
        
        # Volatility features
        self.volatility_features = [
            'close_rolling_std_5', 'close_rolling_std_10', 'close_rolling_std_20',
            'volume_rolling_std_5', 'volume_rolling_std_10',
            'volatility_ratio', 'price_volatility_ratio'
        ]
        
        # Mean reversion features
        self.mean_reversion_features = [
            'price_vs_dma50', 'price_vs_dma200', 'price_vs_vwap',
            'z_score_5', 'z_score_10', 'z_score_20',
            'bollinger_position', 'mean_reversion_signal'
        ]
        
        # Cross-asset features
        self.cross_asset_features = [
            'relative_strength', 'sector_momentum', 'market_correlation',
            'beta_adjusted_return', 'alpha_signal'
        ]
        
        # Sentiment features
        self.sentiment_features = [
            'reddit_title_sentiments_mean', 'reddit_body_sentiments',
            'news_sentiment_mean', 'sentiment_momentum',
            'sentiment_divergence', 'social_volume'
        ]
        
        # Fundamental features (top performing ones)
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
    
    def load_stock_data(self, stock_symbol: str) -> pd.DataFrame:
        """Load data for a specific stock"""
        file_path = f"{self.data_path}/{stock_symbol}_aligned.csv"
        df = pd.read_csv(file_path)
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index(df.columns[0], inplace=True)
        return df
    
    def create_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical indicators"""
        df = df.copy()
        
        # Bollinger Bands
        df['bollinger_upper'] = df['close'].rolling(20).mean() + (df['close'].rolling(20).std() * 2)
        df['bollinger_lower'] = df['close'].rolling(20).mean() - (df['close'].rolling(20).std() * 2)
        df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['close']
        df['bollinger_position'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Stochastic Oscillator
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Williams %R
        df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
        
        # Commodity Channel Index (CCI)
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (tp - sma_tp) / (0.015 * mad)
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(14).mean()
        
        # On-Balance Volume (OBV)
        df['obv'] = (df['volume'] * np.sign(df['close'].diff())).cumsum()
        
        return df
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-based features"""
        df = df.copy()
        
        # Price momentum
        df['close_momentum_5'] = df['close'].pct_change(5)
        df['close_momentum_10'] = df['close'].pct_change(10)
        df['close_momentum_20'] = df['close'].pct_change(20)
        
        # Volume momentum
        df['volume_momentum_5'] = df['volume'].pct_change(5)
        df['volume_momentum_10'] = df['volume'].pct_change(10)
        
        # Price acceleration (second derivative)
        df['price_acceleration'] = df['close'].pct_change().diff()
        
        # Volume acceleration
        df['volume_acceleration'] = df['volume'].pct_change().diff()
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features"""
        df = df.copy()
        
        # Rolling standard deviations
        df['close_rolling_std_5'] = df['close'].rolling(5).std()
        df['close_rolling_std_10'] = df['close'].rolling(10).std()
        df['close_rolling_std_20'] = df['close'].rolling(20).std()
        
        df['volume_rolling_std_5'] = df['volume'].rolling(5).std()
        df['volume_rolling_std_10'] = df['volume'].rolling(10).std()
        
        # Volatility ratios
        df['volatility_ratio'] = df['close_rolling_std_5'] / df['close_rolling_std_20']
        df['price_volatility_ratio'] = df['close_rolling_std_10'] / df['close'].rolling(10).mean()
        
        return df
    
    def create_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create mean reversion features"""
        df = df.copy()
        
        # Price vs moving averages
        df['price_vs_dma50'] = (df['close'] - df['dma_50']) / df['dma_50']
        df['price_vs_dma200'] = (df['close'] - df['dma_200']) / df['dma_200']
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Z-scores
        df['z_score_5'] = (df['close'] - df['close'].rolling(5).mean()) / df['close'].rolling(5).std()
        df['z_score_10'] = (df['close'] - df['close'].rolling(10).mean()) / df['close'].rolling(10).std()
        df['z_score_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        # Mean reversion signal
        df['mean_reversion_signal'] = np.where(
            (df['z_score_5'] < -1) & (df['z_score_10'] < -0.5), 1,
            np.where((df['z_score_5'] > 1) & (df['z_score_10'] > 0.5), -1, 0)
        )
        
        return df
    
    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment-based features"""
        df = df.copy()
        
        # Sentiment momentum
        df['sentiment_momentum'] = df['reddit_title_sentiments_mean'].rolling(5).mean()
        
        # Sentiment divergence
        df['sentiment_divergence'] = df['reddit_title_sentiments_mean'] - df['news_sentiment_mean']
        
        # Social volume
        df['social_volume'] = df['reddit_posts_count'] + df['news_articles_count']
        
        return df
    
    def calculate_feature_importance(self, df: pd.DataFrame, target_col: str = 'returns') -> Dict[str, float]:
        """Calculate feature importance using multiple methods"""
        if target_col not in df.columns:
            df[target_col] = df['close'].pct_change()
        
        # Remove NaN values
        df_clean = df.dropna()
        
        # Select numeric features
        numeric_features = df_clean.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != target_col]
        
        X = df_clean[numeric_features]
        y = df_clean[target_col]
        
        # Method 1: F-statistic
        f_scores = f_regression(X, y)[0]
        f_importance = dict(zip(numeric_features, f_scores))
        
        # Method 2: Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_importance = dict(zip(numeric_features, mi_scores))
        
        # Method 3: Correlation
        corr_importance = {}
        for feature in numeric_features:
            corr = abs(df_clean[feature].corr(df_clean[target_col]))
            corr_importance[feature] = corr if not np.isnan(corr) else 0
        
        # Combine scores (weighted average)
        combined_scores = {}
        for feature in numeric_features:
            f_score = f_importance.get(feature, 0)
            mi_score = mi_importance.get(feature, 0)
            corr_score = corr_importance.get(feature, 0)
            
            # Normalize scores
            f_norm = f_score / max(f_scores) if max(f_scores) > 0 else 0
            mi_norm = mi_score / max(mi_scores) if max(mi_scores) > 0 else 0
            
            # Weighted combination
            combined_scores[feature] = 0.4 * f_norm + 0.4 * mi_norm + 0.2 * corr_score
        
        return combined_scores
    
    def select_best_features(self, df: pd.DataFrame, n_features: int = 50) -> List[str]:
        """Select the best features using multiple criteria"""
        # Calculate feature importance
        importance_scores = self.calculate_feature_importance(df)
        
        # Sort by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top features
        selected_features = [feature for feature, score in sorted_features[:n_features]]
        
        # Ensure we have key features
        essential_features = ['close', 'volume', 'rsi_14', 'dma_50', 'dma_distance']
        for feature in essential_features:
            if feature in df.columns and feature not in selected_features:
                selected_features.append(feature)
        
        return selected_features[:n_features]
    
    def engineer_features_for_stock(self, stock_symbol: str) -> pd.DataFrame:
        """Engineer features for a specific stock"""
        print(f"🔧 Engineering features for {stock_symbol}...")
        
        # Load data
        df = self.load_stock_data(stock_symbol)
        
        # Create all feature categories
        df = self.create_advanced_technical_indicators(df)
        df = self.create_momentum_features(df)
        df = self.create_volatility_features(df)
        df = self.create_mean_reversion_features(df)
        df = self.create_sentiment_features(df)
        
        # Add returns for feature importance calculation
        df['returns'] = df['close'].pct_change()
        
        # Select best features
        selected_features = self.select_best_features(df, n_features=50)
        
        # Create final dataset
        final_df = df[selected_features + ['returns']].copy()
        
        # Handle missing values
        final_df = final_df.fillna(method='ffill').fillna(method='bfill')
        final_df = final_df.dropna()
        
        print(f"✅ {stock_symbol}: {len(selected_features)} features selected")
        return final_df
    
    def engineer_features_for_all_stocks(self, stock_list: List[str]) -> Dict[str, pd.DataFrame]:
        """Engineer features for all stocks"""
        print("🚀 Starting advanced feature engineering for all stocks...")
        
        engineered_data = {}
        for stock in stock_list:
            try:
                engineered_data[stock] = self.engineer_features_for_stock(stock)
            except Exception as e:
                print(f"❌ Error processing {stock}: {e}")
        
        print(f"✅ Feature engineering completed for {len(engineered_data)} stocks")
        return engineered_data
    
    def get_feature_summary(self, engineered_data: Dict[str, pd.DataFrame]) -> Dict:
        """Get summary of engineered features"""
        all_features = set()
        for df in engineered_data.values():
            all_features.update(df.columns)
        
        # Remove 'returns' from feature list
        all_features.discard('returns')
        
        return {
            'total_features': len(all_features),
            'feature_list': sorted(list(all_features)),
            'stocks_processed': len(engineered_data),
            'avg_features_per_stock': np.mean([len(df.columns) - 1 for df in engineered_data.values()])  # -1 for returns
        }

# Example usage
if __name__ == "__main__":
    # Initialize feature engineer
    engineer = AdvancedFeatureEngineer()
    
    # Test with a few stocks
    test_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    
    # Engineer features
    engineered_data = engineer.engineer_features_for_all_stocks(test_stocks)
    
    # Get summary
    summary = engineer.get_feature_summary(engineered_data)
    
    print(f"\n📊 Feature Engineering Summary:")
    print(f"   Total unique features: {summary['total_features']}")
    print(f"   Stocks processed: {summary['stocks_processed']}")
    print(f"   Average features per stock: {summary['avg_features_per_stock']:.1f}")
    
    print(f"\n🔝 Top 20 Features:")
    for i, feature in enumerate(summary['feature_list'][:20]):
        print(f"   {i+1:2d}. {feature}")
