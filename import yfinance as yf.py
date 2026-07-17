import yfinance as yf
news = news_agent("AAPL")

for article in news:
    print(article["Title"])
