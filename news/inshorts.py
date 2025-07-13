import requests

def get_news(category):
    url = f"https://inshorts.deta.dev/news?category={category}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        for article in data['data']:
            print(f"📰 {article['title'].strip()}")
            print(f"🧠 {article['content']}")
            print(f"📅 {article['date']} at {article['time']}")
            print(f"🔗 {article['readMoreUrl']}\n")
    else:
        print("Failed to fetch news.")

# Example: get business news
get_news("business")
