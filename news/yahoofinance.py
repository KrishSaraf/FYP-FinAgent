# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import time

# # Tickers to scrape
# tickers = ['RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'HDFCBANK.NS']

# # Year range filter
# start_year = 2022
# end_year = 2025

# # Helper function to get page content
# def get_page(url):
#     response = requests.get(url)
#     if not response.ok:
#         print(f"❌ Failed to fetch {url}")
#         return None
#     return BeautifulSoup(response.text, 'html.parser')

# # Scrape news for each ticker
# for ticker in tickers:
#     print(f"\n🔍 Fetching news for {ticker}...\n")
#     url = f"https://finance.yahoo.com/quote/{ticker}/news"
#     doc = get_page(url)
    
#     if not doc:
#         continue

#     headline_tags = doc.find_all('h3')
#     news_list = []

#     for tag in headline_tags:
#         headline = tag.text.strip()
#         link_tag = tag.find('a')
#         link = f"https://finance.yahoo.com{link_tag['href']}" if link_tag else "N/A"
        
#         # Filter: keep if any year in title falls within range
#         if any(str(year) in headline for year in range(start_year, end_year + 1)):
#             news_list.append({
#                 "ticker": ticker,
#                 "headline": headline,
#                 "link": link
#             })

#     if news_list:
#         df = pd.DataFrame(news_list)
#         csv_name = f"{ticker.replace('.NS','')}_news_{start_year}_{end_year}.csv"
#         df.to_csv(csv_name, index=False)
#         print(f"✅ Saved {len(df)} headlines to {csv_name}")
#     else:
#         print(f"⚠️ No matching headlines for {ticker} in year range {start_year}–{end_year}")

#     time.sleep(2)  # Be polite to Yahoo's server

# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import time
# from datetime import datetime

# # List of Indian stock tickers (Yahoo Finance format)
# tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']

# # Function to scrape Yahoo Finance news for a single ticker
# def scrape_yahoo_news(ticker):
#     url = f"https://finance.yahoo.com/quote/{ticker}/news"
#     headers = {"User-Agent": "Mozilla/5.0"}  # Prevent blocking
#     response = requests.get(url, headers=headers)

#     if not response.ok:
#         print(f"❌ Failed to fetch news for {ticker}")
#         return []

#     soup = BeautifulSoup(response.text, 'html.parser')
#     headline_tags = soup.find_all('h3')
#     news_list = []

#     for tag in headline_tags:
#         title = tag.text.strip()
#         link_tag = tag.find('a')
#         link = f"https://finance.yahoo.com{link_tag['href']}" if link_tag else "N/A"

#         news_list.append({
#             "ticker": ticker,
#             "headline": title,
#             "link": link,
#             "scraped_on": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         })

#     return news_list

# # Scrape for all tickers and save each to CSV
# for ticker in tickers:
#     print(f"\n🔍 Scraping news for {ticker}...")
#     news = scrape_yahoo_news(ticker)

#     if news:
#         df = pd.DataFrame(news)
#         filename = f"{ticker.replace('.NS','')}_news.csv"
#         df.to_csv(filename, index=False)
#         print(f"✅ Saved {len(df)} headlines to {filename}")
#     else:
#         print(f"⚠️ No news found for {ticker}")

#     time.sleep(1)  # polite delay
from requests_html import HTMLSession
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime

tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']

def scrape_yahoo_news(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    session = HTMLSession()
    try:
        response = session.get(url)
        response.html.render(timeout=20)  # render JavaScript
    except Exception as e:
        print(f"❌ Rendering failed for {ticker}: {e}")
        return []

    soup = BeautifulSoup(response.html.html, 'html.parser')
    headline_tags = soup.find_all('h3')
    news_list = []

    for tag in headline_tags:
        title = tag.text.strip()
        link_tag = tag.find('a')
        link = f"https://finance.yahoo.com{link_tag['href']}" if link_tag else "N/A"

        news_list.append({
            "ticker": ticker,
            "headline": title,
            "link": link,
            "scraped_on": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    return news_list

# Run the scraper
for ticker in tickers:
    print(f"\n🔍 Scraping news for {ticker}...")
    news = scrape_yahoo_news(ticker)

    if news:
        df = pd.DataFrame(news)
        filename = f"{ticker.replace('.NS','')}_news.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Saved {len(df)} headlines to {filename}")
    else:
        print(f"⚠️ No news found for {ticker}")

    time.sleep(2)  # Be polite
