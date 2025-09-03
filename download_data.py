import os
from dotenv import load_dotenv
from finagent.downloader.prices import IndianMarketDownloader
import time
import pandas as pd
from nsepython import nse

# Load environment variables
load_dotenv()

# Initialize downloader
downloader = IndianMarketDownloader(
    root="/Users/puneetgarg/Documents/Pranav Material/FYP/Indian_FinAgent",
    token=os.getenv("INDIAN_API_KEY"),
    stocks_path="finagent/stocks.txt",
    workdir="market_data",
    tag="indian_market"
)

# Download data
downloader.download()

# New OHLC data from nsepython
# Read symbols from stocks.txt
STOCKS_FILE = "finagent/stocks.txt"
with open(STOCKS_FILE, "r") as file:
    SYMBOLS = [line.strip() for line in file.readlines()]

# Define date range
start_date = "06-06-2024"
end_date = "06-06-2025"

# Define output directory
output_dir = "market_data/new_OHCL"
os.makedirs(output_dir, exist_ok=True)

# Loop through all symbols
for symbol in SYMBOLS:
    try:
        print(f"Fetching data for {symbol}...")
        
        # Get historical data
        data = nse.equity_history(symbol, "EQ", start_date, end_date)
        
        if data is not None and not data.empty:
            # Create a clean OHLC dataframe
            ohlc_data = pd.DataFrame({
                'Date': pd.to_datetime(data['mTIMESTAMP'], format='%d-%b-%Y'),
                'Open': data['CH_OPENING_PRICE'],
                'High': data['CH_TRADE_HIGH_PRICE'], 
                'Low': data['CH_TRADE_LOW_PRICE'],
                'Close': data['CH_CLOSING_PRICE'],
                'Volume': data['CH_TOT_TRADED_QTY'],
                'Value_Traded': data['CH_TOT_TRADED_VAL'],
                'VWAP': data['VWAP'],
                'Total_Trades': data['CH_TOTAL_TRADES']
            })
            
            # Sort by date
            ohlc_data = ohlc_data.sort_values('Date').reset_index(drop=True)
            
            # Format for CSV
            csv_data = ohlc_data.copy()
            csv_data['Date'] = csv_data['Date'].dt.strftime('%d-%b-%Y')
            
            # Round numerical values
            csv_data['Open'] = csv_data['Open'].round(2)
            csv_data['High'] = csv_data['High'].round(2) 
            csv_data['Low'] = csv_data['Low'].round(2)
            csv_data['Close'] = csv_data['Close'].round(2)
            csv_data['VWAP'] = csv_data['VWAP'].round(2)
            csv_data['Value_Traded'] = csv_data['Value_Traded'].round(2)
            
            # Save to CSV in the output directory
            output_file = os.path.join(output_dir, f"{symbol}_OHLC_Data_Jun06_2024_Jun06_2025.csv")
            csv_data.to_csv(output_file, index=False)
            print(f"Data saved to CSV file: {output_file}")

        else:
            print(f"No data returned for {symbol}")
            
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
    
    # Small delay to avoid overwhelming the API
    time.sleep(0.5)