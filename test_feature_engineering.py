#!/usr/bin/env python3
"""
Test script for the enhanced feature engineering
Run this to verify that the feature engineering works correctly
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add the current directory to Python path
sys.path.append('.')

def test_basic_feature_engineering():
    """Test the basic feature engineering in portfolio_env.py"""
    print("🧪 Testing basic feature engineering...")
    
    try:
        from finagent.environment.portfolio_env import JAXPortfolioDataLoader
        
        # Create a sample dataframe to test feature engineering
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'close': [1000 + i + (i % 10) * 5 for i in range(100)],
            'open': [995 + i + (i % 8) * 4 for i in range(100)],
            'high': [1005 + i + (i % 12) * 6 for i in range(100)],
            'low': [990 + i + (i % 6) * 3 for i in range(100)],
            'volume': [10000 + i * 100 + (i % 20) * 500 for i in range(100)],
            'dma_50': [998 + i for i in range(100)],
            'dma_200': [995 + i * 0.8 for i in range(100)],
            'rsi_14': [50 + (i % 30) - 15 for i in range(100)]
        }, index=dates)
        
        # Initialize the data loader and test feature engineering
        loader = JAXPortfolioDataLoader(
            data_root="processed_data",
            stocks=["TEST"],
        )
        
        # Test feature engineering
        engineered_df = loader.engineer_features(sample_data, "TEST")
        
        print(f"✅ Original features: {len(sample_data.columns)}")
        print(f"✅ Engineered features: {len(engineered_df.columns)}")
        print(f"✅ Added {len(engineered_df.columns) - len(sample_data.columns)} new features")
        
        # Print some of the new features
        new_features = [col for col in engineered_df.columns if col not in sample_data.columns]
        print(f"📊 New features (first 10): {new_features[:10]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in basic feature engineering: {e}")
        return False


def test_advanced_feature_engineering():
    """Test the advanced cross-sectional feature engineering"""
    print("\n🔬 Testing advanced feature engineering...")
    
    try:
        from feature_engineering_utils import CrossSectionalFeatureEngineer, FactorFeatureEngineer
        
        # Create sample data for multiple stocks
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        
        sample_stocks = {}
        for stock in ['STOCK_A', 'STOCK_B', 'STOCK_C']:
            sample_stocks[stock] = pd.DataFrame({
                'close': [1000 + i + hash(stock) % 100 + (i % 15) * 3 for i in range(50)],
                'volume': [10000 + i * 200 + hash(stock) % 5000 for i in range(50)],
                'returns_1d': [0.01 * ((i + hash(stock)) % 20 - 10) for i in range(50)]
            }, index=dates)
        
        # Test cross-sectional feature engineering
        cross_eng = CrossSectionalFeatureEngineer("dummy_path")
        
        # Test market features
        market_features = cross_eng.create_market_features(sample_stocks)
        print(f"✅ Market features created: {len(market_features.columns)}")
        print(f"📊 Market features: {list(market_features.columns[:5])}")
        
        # Test ranking features
        ranking_features = cross_eng.create_ranking_features(sample_stocks)
        print(f"✅ Ranking features created for {len(ranking_features)} stocks")
        
        # Test factor features
        factor_eng = FactorFeatureEngineer()
        
        # Add some fundamental-like data
        sample_stocks['STOCK_A']['metric_returnOnEquity'] = [0.15 + 0.01 * (i % 10) for i in range(50)]
        factor_features = factor_eng.create_momentum_factors(sample_stocks['STOCK_A'])
        print(f"✅ Factor features added: {len(factor_features.columns) - len(sample_stocks['STOCK_A'].columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in advanced feature engineering: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test if we can load actual data from the processed_data folder"""
    print("\n📁 Testing data loading...")
    
    data_path = Path("processed_data")
    if not data_path.exists():
        print("⚠️  processed_data folder not found, skipping data loading test")
        return True
    
    try:
        # Find aligned CSV files
        aligned_files = list(data_path.glob("*_aligned.csv"))
        if not aligned_files:
            print("⚠️  No aligned CSV files found, skipping data loading test")
            return True
        
        # Test loading one file
        test_file = aligned_files[0]
        df = pd.read_csv(test_file, index_col=0, parse_dates=True)
        print(f"✅ Successfully loaded {test_file.name}")
        print(f"📊 Shape: {df.shape}")
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")
        print(f"🏷️  Columns: {len(df.columns)}")
        
        # Test basic columns are present
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing required columns: {missing_cols}")
        else:
            print("✅ All required OHLCV columns present")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def main():
    """Run all feature engineering tests"""
    print("🚀 Starting Feature Engineering Tests\n")
    
    results = []
    
    # Run tests
    results.append(("Basic Feature Engineering", test_basic_feature_engineering()))
    results.append(("Advanced Feature Engineering", test_advanced_feature_engineering()))
    results.append(("Data Loading", test_data_loading()))
    
    # Print results summary
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Feature engineering is ready.")
        print("\n💡 Next steps:")
        print("   1. Run: python feature_engineering_utils.py")
        print("   2. Or run: python train_ppo_enhanced.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)