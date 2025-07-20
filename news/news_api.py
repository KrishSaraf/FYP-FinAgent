# import requests
# import os
# import time
# import csv
# from dotenv import load_dotenv

# # Load API key
# load_dotenv()
# API_KEY = os.getenv("NEWS_API_KEY")

# # Set your full output path
# BASE_SAVE_PATH = "/Users/aravbehl/ntu_college/FYP/main_code/FYP-FinAgent/news/stocks_news_data"
# os.makedirs(BASE_SAVE_PATH, exist_ok=True)

# # Stock list
# stocks = [
#     "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "HDFC", "SBIN", "BHARTIARTL", "KOTAKBANK",
#     "LT", "ASIANPAINT", "AXISBANK", "MARUTI", "ULTRACEMCO", "TITAN", "BAJFINANCE", "NESTLEIND", "ONGC", "POWERGRID",
#     "NTPC", "ITC", "BAJAJFINSV", "SUNPHARMA", "TECHM", "WIPRO", "HCLTECH", "JSWSTEEL", "TATAMOTORS", "ADANIPORTS",
#     "BRITANNIA", "SHREECEM", "TATACONSUM", "BPCL", "INDUSINDBK", "GRASIM", "EICHERMOT", "DRREDDY", "COALINDIA",
#     "HEROMOTOCO", "IOC", "UPL", "CIPLA", "TATASTEEL", "HDFCLIFE", "BAJAJ-AUTO"
# ]

# FROM_DATE = "2021-07-12"
# TO_DATE = "2025-07-12"
# BASE_URL = "https://newsdata.io/api/1/archive"

# # Loop through each stock
# for stock in stocks:
#     print(f"🔍 Fetching news for {stock}...")
#     page = 1
#     articles = []

#     while True:
#         params = {
#             "apikey": API_KEY,
#             "q": stock,
#             "language": "en",
#             "from_date": FROM_DATE,
#             "to_date": TO_DATE,
#             "page": page
#         }

#         try:
#             response = requests.get(BASE_URL, params=params)
#             if response.status_code != 200:
#                 print(f"❌ Failed to fetch {stock}, status code: {response.status_code}")
#                 break

#             data = response.json()
#             results = data.get("results", [])
#             if not results:
#                 break

#             articles.extend(results)
#             page += 1
#             time.sleep(1)  # Rate limiting

#         except Exception as e:
#             print(f"⚠️ Error fetching {stock} page {page}: {e}")
#             break

#     # Write articles to CSV (if any found)
#     if articles:
#         output_path = os.path.join(BASE_SAVE_PATH, f"{stock}.csv")
#         fieldnames = set()

#         # Dynamically collect all possible fields across all articles
#         for article in articles:
#             fieldnames.update(article.keys())

#         fieldnames = sorted(fieldnames)  # consistent column order

#         with open(output_path, mode="w", newline="", encoding="utf-8") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             for article in articles:
#                 writer.writerow(article)

#         print(f"✅ Saved {len(articles)} articles to {output_path}")
#     else:
#         print(f"⚠️ No articles found for {stock}")


import requests
import os
import time
import csv
import random
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

# Set the full output directory
BASE_SAVE_PATH = "/Users/aravbehl/ntu_college/FYP/main_code/FYP-FinAgent/news/stocks_news_data"
os.makedirs(BASE_SAVE_PATH, exist_ok=True)

# Define stock list (use a shorter one for testing)
stocks = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "HDFC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "ASIANPAINT", "AXISBANK", "MARUTI", "ULTRACEMCO", "TITAN", "BAJFINANCE", "NESTLEIND", "ONGC", "POWERGRID",
    "NTPC", "ITC", "BAJAJFINSV", "SUNPHARMA", "TECHM", "WIPRO", "HCLTECH", "JSWSTEEL", "TATAMOTORS", "ADANIPORTS",
    "BRITANNIA", "SHREECEM", "TATACONSUM", "BPCL", "INDUSINDBK", "GRASIM", "EICHERMOT", "DRREDDY", "COALINDIA",
    "HEROMOTOCO", "IOC", "UPL", "CIPLA", "TATASTEEL", "HDFCLIFE", "BAJAJ-AUTO"
]

FROM_DATE = "2021-07-13"
TO_DATE = "2025-07-13"
BASE_URL = "https://newsdata.io/api/1/archive"  # requires paid plan
# BASE_URL = "https://newsdata.io/api/1/news"   # use this for free plan (latest news only)

MAX_RETRIES = 5
DELAY_BETWEEN_REQUESTS = 2.5  # seconds
MAX_REQUESTS_PER_DAY = 190  # stay below 200 API daily limit
total_requests = 0


def fetch_with_retry(url, params, delay=2):
    global total_requests
    retries = 0

    while retries < MAX_RETRIES:
        if total_requests >= MAX_REQUESTS_PER_DAY:
            print("🛑 Daily API request limit reached. Stopping.")
            return None

        response = requests.get(url, params=params)
        total_requests += 1

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            wait = delay + random.uniform(1, 3)
            print(f"⏳ Hit rate limit. Waiting {wait:.1f}s before retrying...")
            time.sleep(wait)
            retries += 1
        else:
            print(f"❌ Failed with status {response.status_code}")
            return None

    print("⚠️ Max retries exceeded. Skipping this request.")
    return None


# Loop through each stock
for stock in stocks:
    print(f"\n🔍 Fetching news for {stock}...")
    page = 1
    articles = []

    while True:
        params = {
            "apikey": API_KEY,
            "q": stock,
            "language": "en",
            "from_date": FROM_DATE,
            "to_date": TO_DATE,
            "page": page
        }

        data = fetch_with_retry(BASE_URL, params, delay=DELAY_BETWEEN_REQUESTS)
        if not data:
            break

        results = data.get("results", [])
        if not results:
            break

        articles.extend(results)
        page += 1
        time.sleep(DELAY_BETWEEN_REQUESTS + random.uniform(0.5, 1.5))  # jitter between page pulls

    # Write to CSV if articles were found
    if articles:
        output_path = os.path.join(BASE_SAVE_PATH, f"{stock}.csv")

        # Dynamically collect all keys from articles
        fieldnames = set()
        for article in articles:
            fieldnames.update(article.keys())
        fieldnames = sorted(fieldnames)

        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for article in articles:
                writer.writerow(article)

        print(f"✅ Saved {len(articles)} articles for {stock} to {output_path}")
    else:
        print(f"⚠️ No articles found for {stock} or request was skipped.")
