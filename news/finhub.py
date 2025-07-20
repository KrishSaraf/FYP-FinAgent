import os
import time
import datetime
import pandas as pd
import finnhub

# Set your Finnhub API key
API_KEY = os.getenv('FINNHUB_API_KEY', 'cum1tjpr01qovv718260cum1tjpr01qovv71826g')
client = finnhub.Client(api_key=API_KEY)

# List of stock symbols (NSE suffix)
stocks = [
    "RELIANCE.NSE", "TCS.NSE", "HDFCBANK.NSE", "INFY.NSE", "ICICIBANK.NSE", "HINDUNILVR.NSE", "HDFC.NSE", "SBIN.NSE",
    "BHARTIARTL.NSE", "KOTAKBANK.NSE", "LT.NSE", "ASIANPAINT.NSE", "AXISBANK.NSE", "MARUTI.NSE", "ULTRACEMCO.NSE",
    "TITAN.NSE", "BAJFINANCE.NSE", "NESTLEIND.NSE", "ONGC.NSE", "POWERGRID.NSE", "NTPC.NSE", "ITC.NSE", "BAJAJFINSV.NSE",
    "SUNPHARMA.NSE", "TECHM.NSE", "WIPRO.NSE", "HCLTECH.NSE", "JSWSTEEL.NSE", "TATAMOTORS.NSE", "ADANIPORTS.NSE",
    "BRITANNIA.NSE", "SHREECEM.NSE", "TATACONSUM.NSE", "BPCL.NSE", "INDUSINDBK.NSE", "GRASIM.NSE", "EICHERMOT.NSE",
    "DRREDDY.NSE", "COALINDIA.NSE", "HEROMOTOCO.NSE", "IOC.NSE", "UPL.NSE", "CIPLA.NSE", "TATASTEEL.NSE",
    "HDFCLIFE.NSE", "BAJAJ-AUTO.NSE"
]

# Date range
start_date = datetime.datetime(2021, 7, 12)
end_date = datetime.datetime(2025, 7, 12)

def fetch_stock_data(symbol, start, end):
    # Finnhub free tier: max 1 year per call
    data_frames = []
    current_start = start
    one_year = datetime.timedelta(days=365)
    while current_start < end:
        current_end = min(current_start + one_year, end)
        from_ts = int(current_start.timestamp())
        to_ts = int(current_end.timestamp())
        res = client.stock_candles(symbol, 'D', from_ts, to_ts)
        if res and res.get('s') == 'ok':
            df = pd.DataFrame({
                'date': pd.to_datetime(res['t'], unit='s'),
                'open': res['o'],
                'high': res['h'],
                'low': res['l'],
                'close': res['c'],
                'volume': res['v']
            })
            data_frames.append(df)
        else:
            print(f"Failed for {symbol} from {current_start.date()} to {current_end.date()}")
        current_start = current_end + datetime.timedelta(days=1)
        time.sleep(1)  # Respect API rate limit
    if data_frames:
        return pd.concat(data_frames).reset_index(drop=True)
    return pd.DataFrame()

# Download data for each stock
for stock in stocks:
    print(f"Fetching: {stock}")
    df = fetch_stock_data(stock, start_date, end_date)
    if not df.empty:
        df.to_csv(f"{stock}_2021-2025.csv", index=False)
        print(f"Saved {stock}_2021-2025.csv")
    else:
        print(f"No data for {stock}")

print("Done.")
