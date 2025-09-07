"""
Advanced Feature Engineering Utilities for FinAgent
Focuses on creating high-alpha features for better returns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CrossSectionalFeatureEngineer:
    """Create features that use information across multiple stocks"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        
    def load_all_stock_data(self, date_range: Optional[Tuple[str, str]] = None) -> Dict[str, pd.DataFrame]:
        """Load all aligned CSV files"""
        stock_data = {}
        
        for file_path in self.data_root.glob("*_aligned.csv"):
            stock_name = file_path.stem.replace("_aligned", "")
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            if date_range:
                start_date, end_date = date_range
                df = df.loc[start_date:end_date]
                
            stock_data[stock_name] = df
            
        return stock_data
    
    def create_market_features(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create market-wide features for cross-sectional analysis"""
        
        # Get common date index
        common_dates = None
        for stock, df in stock_data.items():
            if common_dates is None:
                common_dates = df.index
            else:
                common_dates = common_dates.intersection(df.index)
        
        market_features = pd.DataFrame(index=common_dates)
        
        # === MARKET MOMENTUM & DISPERSION ===
        closes = pd.DataFrame({stock: data.loc[common_dates, 'close'] for stock, data in stock_data.items()})
        returns_1d = closes.pct_change()
        returns_5d = closes.pct_change(periods=5)
        returns_20d = closes.pct_change(periods=20)
        
        # Market breadth indicators
        market_features['market_up_ratio_1d'] = (returns_1d > 0).mean(axis=1)
        market_features['market_up_ratio_5d'] = (returns_5d > 0).mean(axis=1)
        market_features['market_up_ratio_20d'] = (returns_20d > 0).mean(axis=1)
        
        # Cross-sectional momentum dispersion
        market_features['momentum_dispersion_1d'] = returns_1d.std(axis=1)
        market_features['momentum_dispersion_5d'] = returns_5d.std(axis=1)
        market_features['momentum_dispersion_20d'] = returns_20d.std(axis=1)
        
        # Market momentum (equal-weighted)
        market_features['market_momentum_1d'] = returns_1d.mean(axis=1)
        market_features['market_momentum_5d'] = returns_5d.mean(axis=1)
        market_features['market_momentum_20d'] = returns_20d.mean(axis=1)
        
        # === VOLATILITY REGIME ===
        market_vol = returns_1d.std(axis=1).rolling(20).mean()
        market_features['market_vol_regime'] = (market_vol > market_vol.rolling(60).mean()).astype(float)
        
        # === CORRELATION STRUCTURE ===
        # Rolling correlation with market
        market_return = returns_1d.mean(axis=1)
        
        for stock in list(stock_data.keys())[:10]:  # Limit to avoid too many features
            stock_return = returns_1d[stock]
            # Calculate rolling correlation properly
            rolling_corr = stock_return.rolling(30).corr(market_return)
            market_features[f'{stock}_market_correlation'] = rolling_corr
        
        # === SECTOR/FACTOR MOMENTUM ===
        # Approximate sector momentum using stock clusters
        financial_stocks = [s for s in stock_data.keys() if any(bank in s for bank in ['BANK', 'HDFC', 'ICICI', 'AXIS', 'KOTAK', 'SBIN'])]
        tech_stocks = [s for s in stock_data.keys() if any(tech in s for tech in ['INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM'])]
        energy_stocks = [s for s in stock_data.keys() if any(energy in s for energy in ['RELIANCE', 'ONGC', 'IOC', 'BPCL'])]
        
        for sector_name, sector_stocks in [('financial', financial_stocks), ('tech', tech_stocks), ('energy', energy_stocks)]:
            if len(sector_stocks) > 2:
                sector_returns = returns_1d[sector_stocks].mean(axis=1)
                market_features[f'{sector_name}_momentum_1d'] = sector_returns
                market_features[f'{sector_name}_momentum_5d'] = sector_returns.rolling(5).mean()
                market_features[f'{sector_name}_vs_market'] = sector_returns - market_return
        
        return market_features.fillna(0.0)
    
    def create_ranking_features(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Create cross-sectional ranking features for each stock"""
        
        # Get common dates
        common_dates = None
        for stock, df in stock_data.items():
            if common_dates is None:
                common_dates = df.index
            else:
                common_dates = common_dates.intersection(df.index)
        
        stock_ranking_features = {}
        
        # Prepare data matrices for ranking
        closes = pd.DataFrame({stock: data.loc[common_dates, 'close'] for stock, data in stock_data.items()})
        volumes = pd.DataFrame({stock: data.loc[common_dates, 'volume'] for stock, data in stock_data.items() if 'volume' in data.columns})
        
        returns_1d = closes.pct_change()
        returns_5d = closes.pct_change(periods=5)
        returns_20d = closes.pct_change(periods=20)
        
        volatility_20d = returns_1d.rolling(20).std()
        
        for stock in stock_data.keys():
            stock_features = pd.DataFrame(index=common_dates)
            
            # === MOMENTUM RANKINGS ===
            stock_features['momentum_rank_1d'] = returns_1d[stock].rolling(60).rank(pct=True)
            stock_features['momentum_rank_5d'] = returns_5d[stock].rolling(60).rank(pct=True)
            stock_features['momentum_rank_20d'] = returns_20d[stock].rolling(60).rank(pct=True)
            
            # === VOLATILITY RANKINGS ===
            stock_features['vol_rank_20d'] = volatility_20d[stock].rolling(60).rank(pct=True)
            
            # === VOLUME RANKINGS ===
            if stock in volumes.columns:
                volume_ratio = volumes[stock] / volumes[stock].rolling(20).mean()
                stock_features['volume_rank'] = volume_ratio.rolling(60).rank(pct=True)
            
            # === RELATIVE STRENGTH ===
            # Stock vs market performance
            market_return = returns_1d.mean(axis=1)
            relative_strength = (returns_1d[stock] - market_return).rolling(20).mean()
            stock_features['relative_strength_rank'] = relative_strength.rolling(60).rank(pct=True)
            
            # === CONSISTENCY RANKINGS ===
            # How consistently does this stock outperform/underperform
            outperformance_days = (returns_1d[stock] > market_return).rolling(20).mean()
            stock_features['consistency_rank'] = outperformance_days.rolling(60).rank(pct=True)
            
            stock_ranking_features[stock] = stock_features.fillna(0.5)  # Fill with median rank
        
        return stock_ranking_features


class FactorFeatureEngineer:
    """Create factor-based features inspired by quantitative finance"""
    
    @staticmethod
    def create_quality_factors(df: pd.DataFrame) -> pd.DataFrame:
        """Create quality factors from fundamental data"""
        quality_features = df.copy()
        
        # ROE trend and stability
        roe_cols = [col for col in df.columns if 'returnOnEquity' in col]
        if roe_cols:
            roe_col = roe_cols[0]
            quality_features['roe_trend'] = df[roe_col] - df[roe_col].shift(252)  # YoY change
            quality_features['roe_stability'] = df[roe_col].rolling(252).std()
        
        # Profit margin trends
        margin_cols = [col for col in df.columns if 'Margin' in col and 'net' in col.lower()]
        if margin_cols:
            margin_col = margin_cols[0]
            quality_features['margin_trend'] = df[margin_col] - df[margin_col].shift(63)  # QoQ change
            quality_features['margin_volatility'] = df[margin_col].rolling(252).std()
        
        # Debt ratios
        debt_cols = [col for col in df.columns if 'debt' in col.lower() and 'equity' in col.lower()]
        if debt_cols:
            debt_col = debt_cols[0]
            quality_features['debt_trend'] = df[debt_col].pct_change(periods=252)
            quality_features['debt_level'] = df[debt_col]
        
        return quality_features.fillna(0.0)
    
    @staticmethod
    def create_momentum_factors(df: pd.DataFrame) -> pd.DataFrame:
        """Create sophisticated momentum factors"""
        momentum_features = df.copy()
        
        if 'close' not in df.columns:
            return momentum_features.fillna(0.0)
        
        returns = df['close'].pct_change()
        
        # === MOMENTUM PERSISTENCE ===
        # How consistent is the momentum over different horizons
        mom_1m = df['close'].pct_change(periods=20)
        mom_3m = df['close'].pct_change(periods=60)
        mom_6m = df['close'].pct_change(periods=120)
        mom_12m = df['close'].pct_change(periods=252)
        
        momentum_features['momentum_persistence'] = (
            (mom_1m > 0).astype(float) + 
            (mom_3m > 0).astype(float) + 
            (mom_6m > 0).astype(float) + 
            (mom_12m > 0).astype(float)
        ) / 4.0
        
        # === MOMENTUM ACCELERATION ===
        momentum_features['momentum_acceleration'] = mom_1m - mom_3m
        
        # === REVERSAL INDICATORS ===
        # Short-term reversal after strong momentum
        strong_momentum = mom_3m > mom_3m.rolling(60).quantile(0.8)
        short_reversal = returns.rolling(5).mean() < 0
        momentum_features['momentum_reversal_signal'] = (strong_momentum & short_reversal).astype(float)
        
        # === RISK-ADJUSTED MOMENTUM ===
        vol_20d = returns.rolling(20).std()
        momentum_features['risk_adj_momentum_1m'] = mom_1m / (vol_20d * np.sqrt(20) + 1e-8)
        momentum_features['risk_adj_momentum_3m'] = mom_3m / (vol_20d * np.sqrt(60) + 1e-8)
        
        return momentum_features.fillna(0.0)
    
    @staticmethod
    def create_mean_reversion_factors(df: pd.DataFrame) -> pd.DataFrame:
        """Create mean reversion factors"""
        reversion_features = df.copy()
        
        if 'close' not in df.columns:
            return reversion_features.fillna(0.0)
        
        returns = df['close'].pct_change()
        
        # === OVERBOUGHT/OVERSOLD CONDITIONS ===
        # Price distance from various moving averages
        for ma_period in [20, 50, 200]:
            ma = df['close'].rolling(ma_period).mean()
            reversion_features[f'price_vs_ma{ma_period}'] = (df['close'] - ma) / ma
            
            # How long has it been away from MA
            above_ma = df['close'] > ma
            days_above = above_ma.groupby((above_ma != above_ma.shift()).cumsum()).cumcount() + 1
            reversion_features[f'days_above_ma{ma_period}'] = days_above * above_ma
        
        # === SHORT-TERM REVERSAL ===
        # Recent extreme moves that might revert
        returns_5d = returns.rolling(5).sum()
        extreme_positive = returns_5d > returns_5d.rolling(60).quantile(0.95)
        extreme_negative = returns_5d < returns_5d.rolling(60).quantile(0.05)
        
        reversion_features['extreme_positive_reversal'] = extreme_positive.astype(float)
        reversion_features['extreme_negative_reversal'] = extreme_negative.astype(float)
        
        # === BOLLINGER BAND POSITION ===
        if 'close' in df.columns:
            bb_middle = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std
            
            reversion_features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)
            reversion_features['bb_squeeze'] = bb_std / bb_middle  # Volatility contraction
        
        return reversion_features.fillna(0.0)


def enhance_training_data(data_root: str, output_path: Optional[str] = None) -> str:
    """
    Main function to enhance training data with advanced features
    
    Args:
        data_root: Path to processed_data folder
        output_path: Where to save enhanced data (optional)
        
    Returns:
        Path to enhanced data
    """
    print("🚀 Starting advanced feature engineering...")
    
    # Initialize feature engineers
    cross_sectional = CrossSectionalFeatureEngineer(data_root)
    factor_engineer = FactorFeatureEngineer()
    
    # Load all stock data
    print("📊 Loading stock data...")
    stock_data = cross_sectional.load_all_stock_data()
    print(f"Loaded {len(stock_data)} stocks")
    
    # Create market-wide features
    print("🌐 Creating market features...")
    market_features = cross_sectional.create_market_features(stock_data)
    
    # Create ranking features for each stock
    print("📈 Creating ranking features...")
    ranking_features = cross_sectional.create_ranking_features(stock_data)
    
    # Enhance each stock's data
    enhanced_data = {}
    
    for stock_name, df in stock_data.items():
        print(f"⚡ Enhancing {stock_name}...")
        
        # Add market features
        enhanced_df = df.copy()
        for col in market_features.columns:
            if col in enhanced_df.index:
                enhanced_df = enhanced_df.join(market_features[col], rsuffix='_market')
        
        # Add ranking features
        if stock_name in ranking_features:
            enhanced_df = enhanced_df.join(ranking_features[stock_name], rsuffix='_rank')
        
        # Add factor features
        enhanced_df = factor_engineer.create_quality_factors(enhanced_df)
        enhanced_df = factor_engineer.create_momentum_factors(enhanced_df)
        enhanced_df = factor_engineer.create_mean_reversion_factors(enhanced_df)
        
        enhanced_data[stock_name] = enhanced_df
    
    # Save enhanced data
    if output_path is None:
        output_path = str(Path(data_root) / "enhanced")
    
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True)
    
    for stock_name, enhanced_df in enhanced_data.items():
        output_file = output_dir / f"{stock_name}_enhanced.csv"
        enhanced_df.to_csv(output_file)
    
    # Save market features separately
    market_features.to_csv(output_dir / "market_features_enhanced.csv")
    
    print(f"✅ Enhanced feature engineering complete! Data saved to {output_path}")
    print(f"📊 Added {len(market_features.columns)} market features")
    print(f"🎯 Enhanced {len(enhanced_data)} stock datasets")
    
    return str(output_path)


if __name__ == "__main__":
    # Example usage
    data_root = "FYP-FinAgent/processed_data"
    enhanced_path = enhance_training_data(data_root)
    print(f"Enhanced data available at: {enhanced_path}")