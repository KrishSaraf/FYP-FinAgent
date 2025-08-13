import logging
from pathlib import Path
from finagent.processor.market_data_consolidator import MarketDataConsolidator
from finagent.processor.temporal_aligner import TemporalDataAligner

def process_stocks(data_root: str, output_dir: str, market: str = "NSE"):
    """
    Process stock data by consolidating, aligning, and saving the results.

    Args:
        data_root (str): Root directory containing market data.
        output_dir (str): Directory to save processed data.
        market (str): Market name (default: "NSE").
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize consolidator and aligner
    consolidator = MarketDataConsolidator(data_root=data_root)
    aligner = TemporalDataAligner(market=market)

    # Load stock list
    stocks = consolidator._load_stock_list()
    if not stocks:
        logger.error("No stocks found in stocks.txt. Exiting.")
        return

    logger.info(f"Processing {len(stocks)} stocks...")

    # Consolidate and align data for each stock
    consolidated_data = {}
    aligned_data = {}

    for stock in stocks:
        try:
            # Consolidate data
            logger.info(f"Consolidating data for {stock}...")
            consolidated_df = consolidator.consolidate_stock_data(stock)
            if consolidated_df.empty:
                logger.warning(f"No consolidated data for {stock}. Skipping.")
                continue

            # Align data to trading days
            logger.info(f"Aligning data for {stock}...")
            aligned_df = aligner.align_to_trading_days(consolidated_df)
            if aligned_df.empty:
                logger.warning(f"No aligned data for {stock}. Skipping.")
                continue

            # Create lagged features
            logger.info(f"Creating lagged features for {stock}...")
            aligned_df = aligner.create_lagged_features(aligned_df)

            # Store processed data
            consolidated_data[stock] = consolidated_df
            aligned_data[stock] = aligned_df

        except Exception as e:
            logger.error(f"Error processing {stock}: {e}")
            continue

    # Create market-wide features
    logger.info("Creating market-wide features...")
    market_features = aligner.create_market_features(aligned_data)

    # Save processed data
    logger.info("Saving processed data...")
    consolidator.save_consolidated_data(consolidated_data, output_dir=output_dir)
    aligner.save_aligned_data(aligned_data, market_features, output_dir=output_dir)

    logger.info("Processing complete.")

if __name__ == "__main__":
    # Define paths
    DATA_ROOT = "market_data"
    OUTPUT_DIR = "processed_data"

    # Process stocks
    process_stocks(data_root=DATA_ROOT, output_dir=OUTPUT_DIR)