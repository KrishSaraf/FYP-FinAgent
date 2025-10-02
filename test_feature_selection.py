"""
Quick test script to verify feature selection works correctly
"""
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the classes
from train_ppo_feature_combinations import CustomDataLoader, CustomPortfolioEnv, FeatureSelector

def test_feature_selection():
    """Test that CustomDataLoader and CustomPortfolioEnv respect feature selection"""

    logger.info("=" * 80)
    logger.info("FEATURE SELECTION TEST")
    logger.info("=" * 80)

    # Test 1: Select minimal features
    logger.info("\n>>> TEST 1: Minimal OHLCV features")
    selector = FeatureSelector()
    minimal_features = ['close', 'open', 'high', 'low', 'volume', 'returns_1d']

    logger.info(f"Selected {len(minimal_features)} features: {minimal_features}")

    # Test CustomDataLoader
    try:
        logger.info("\n--- Testing CustomDataLoader ---")
        loader = CustomDataLoader(
            selected_features=minimal_features,
            data_root='processed_data/',
            stocks=['HDFCBANK', 'RELIANCE'],  # Just 2 stocks for fast test
            use_all_features=False
        )

        data, day_idx, valid_dates, n_features = loader.load_and_preprocess_data(
            start_date='2024-06-06',
            end_date='2024-07-06',  # Just 1 month
            preload_to_gpu=False,
            force_reload=True
        )

        logger.info(f"✓ CustomDataLoader returned:")
        logger.info(f"  Data shape: {data.shape}")
        logger.info(f"  n_features: {n_features}")
        logger.info(f"  loader.features count: {len(loader.features)}")
        logger.info(f"  loader.features: {loader.features}")

        if n_features != len(minimal_features):
            logger.error(f"✗ FAILED: Expected {len(minimal_features)} features, got {n_features}")
            return False

        if data.shape[2] != len(minimal_features):
            logger.error(f"✗ FAILED: Data shape[2]={data.shape[2]}, expected {len(minimal_features)}")
            return False

        logger.info("✓ CustomDataLoader test PASSED")

    except Exception as e:
        logger.error(f"✗ CustomDataLoader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test CustomPortfolioEnv
    try:
        logger.info("\n--- Testing CustomPortfolioEnv ---")
        env = CustomPortfolioEnv(
            selected_features=minimal_features,
            data_root='processed_data/',
            stocks=['HDFCBANK', 'RELIANCE'],
            window_size=10,
            start_date='2024-06-06',
            end_date='2024-07-06'
        )

        logger.info(f"✓ CustomPortfolioEnv created:")
        logger.info(f"  Data shape: {env.data.shape}")
        logger.info(f"  n_features: {env.n_features}")
        logger.info(f"  env.features count: {len(env.features)}")
        logger.info(f"  env.features: {env.features}")
        logger.info(f"  obs_dim: {env.obs_dim}")

        if env.n_features != len(minimal_features):
            logger.error(f"✗ FAILED: Expected {len(minimal_features)} features, got {env.n_features}")
            return False

        if env.data.shape[2] != len(minimal_features):
            logger.error(f"✗ FAILED: Data shape[2]={env.data.shape[2]}, expected {len(minimal_features)}")
            return False

        if env.features != minimal_features:
            logger.error(f"✗ FAILED: Feature list mismatch")
            logger.error(f"  Expected: {minimal_features}")
            logger.error(f"  Got: {env.features}")
            return False

        logger.info("✓ CustomPortfolioEnv test PASSED")

    except Exception as e:
        logger.error(f"✗ CustomPortfolioEnv test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: OHLCV + Technical combination
    logger.info("\n>>> TEST 2: OHLCV + Technical combination")
    try:
        combined_features = selector.get_features_for_combination('ohlcv+technical')
        logger.info(f"Selected {len(combined_features)} features for 'ohlcv+technical'")

        env2 = CustomPortfolioEnv(
            selected_features=combined_features,
            data_root='processed_data/',
            stocks=['HDFCBANK'],  # Just 1 stock
            window_size=10,
            start_date='2024-06-06',
            end_date='2024-07-06'
        )

        if env2.n_features != len(combined_features):
            logger.error(f"✗ FAILED: Expected {len(combined_features)} features, got {env2.n_features}")
            return False

        if env2.data.shape[2] != len(combined_features):
            logger.error(f"✗ FAILED: Data shape[2]={env2.data.shape[2]}, expected {len(combined_features)}")
            return False

        logger.info(f"✓ Combined features test PASSED")
        logger.info(f"  Data shape: {env2.data.shape}")
        logger.info(f"  n_features: {env2.n_features}")

    except Exception as e:
        logger.error(f"✗ Combined features test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL TESTS PASSED!")
    logger.info("=" * 80)
    return True

if __name__ == "__main__":
    success = test_feature_selection()
    sys.exit(0 if success else 1)
