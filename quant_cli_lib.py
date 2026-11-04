import yfinance as yf


def prices(tickers, years):
    df = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)["Close"]
    return df.dropna(how="all").ffill().dropna()
