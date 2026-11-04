import pandas as pd
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "sp500.csv")
SP500 = pd.read_csv(CSV_PATH)
stock_cache = {}
def get_company_info(ticker):
    
    if ticker in stock_cache:
        stock = stock_cache[ticker]
    else:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        stock_cache[ticker] = stock

    return stock, stock.info


#-----Naming Values-----#
def get_financialmetrics(info):
    return{

        "Name":info.get("longName"),
        "MarketCap":info.get("marketCap"),
        "Revenue":info.get("totalRevenue"),
        "EnterpriseValue":info.get("enterpriseValue"),
        "Peratio":info.get("trailingPE"),
        "Peratiof":info.get("forwardPE"),
        "Profit":info.get("netIncomeToCommon"),
        "Revenuegrowth":info.get("revenueGrowth"),
        "ProfitMargin":info.get("profitMargins"),
        "Roe":info.get("returnOnEquity"),
        "Deratio":info.get("debtToEquity"),
        "Cash":info.get("totalCash"),
        "Debt":info.get("totalDebt"),
        "Avg":info.get("twoHundredDayAverage"),
        "Fcf":info.get("freeCashflow"),
        "Ebidta":info.get("ebitda"),
        "Pbratio":info.get("priceToBook"),
        "EvEbidtaratio":info.get("enterpriseValue") / info.get("ebitda")
        if info.get("enterpriseValue") is not None
        and info.get("ebitda") not in (None,0)
        else None,
        "Earningsgrowth":info.get("earningsGrowth"),
        "Beta":info.get("beta"),
        "Roa":info.get("returnOnAssets"),
        "Sharesoutstanding" : info.get("sharesOutstanding"),
        "Currentprice" :info.get("currentPrice"),
        "Sector":info.get("sector"),
        "Industry":info.get("industry"),
        "OperatingMargin":info.get("operatingMargins"),
        "DividendRate":info.get("dividendRate"),
        "BookValue":info.get("bookValue"),
        "GrossMargin":info.get("grossMargins"),
        "Summary":info.get("longBusinessSummary"),
        "Employees":info.get("fullTimeEmployees"),
        "Website":info.get("website"),
    }
    
#----------------------#

#-----Mapping to s&p 500-----#

def get_cik(ticker):
    row = SP500[SP500["Symbol"].str.upper() == ticker.upper()]
    return int(row.iloc[0]["CIK"]) if not row.empty else None


def peertickers(sector):
    sp500 = SP500
    sector_map = {
        "Technology": "Information Technology",
        "Financial Services": "Financials",
        "Healthcare": "Health Care",
        "Consumer Cyclical": "Consumer Discretionary",
        "Consumer Defensive": "Consumer Staples",
        "Basic Materials": "Materials"
    }
    sp_sector = sector_map.get(sector, sector)
    same_sector = sp500[
       sp500["GICS Sector"] == sp_sector
    ]
    peer_tickers = same_sector["Symbol"].tolist()
    return peer_tickers


#---------------------------#
try:
    from rapidfuzz import process
except ImportError:  # fallback keeps search working without rapidfuzz
    import difflib

    class process:  # minimal shim
        @staticmethod
        def extract(q, choices, limit=5, score_cutoff=0):
            out = [(c, difflib.SequenceMatcher(None, q.lower(), c.lower()).ratio() * 100, i)
                   for i, c in enumerate(choices)]
            out = [o for o in out if o[1] >= score_cutoff]
            return sorted(out, key=lambda x: -x[1])[:limit]

# Search both company names and tickers
SEARCH_CHOICES = (
    SP500["Security"] + " (" + SP500["Symbol"] + ")"
).tolist()


def find_company(user_input):
    user_input = user_input.strip()

    # If user entered a ticker
    ticker_match = SP500[
        SP500["Symbol"].str.upper() == user_input.upper()
    ]

    if not ticker_match.empty:
        return ticker_match.iloc[0]["Symbol"]

    # Fuzzy search
    matches = process.extract(
        user_input,
        SEARCH_CHOICES,
        limit=5,
        score_cutoff=60
    )

    if not matches:
        return None

    # If almost perfect match, auto-select
    if matches[0][1] >= 95:
        company = matches[0][0]

        ticker = company.split("(")[1].replace(")", "")

        print(f"\nSelected: {company}")

        return ticker

    print("\nDid you mean:\n")

    for i, (company, score, _) in enumerate(matches, start=1):
        print(f"{i}. {company}")

    while True:
        choice = input("\nSelect number (Enter to cancel): ")

        if choice == "":
            return None

        if choice.isdigit():
            choice = int(choice)

            if 1 <= choice <= len(matches):
                company = matches[choice - 1][0]

                ticker = company.split("(")[1].replace(")", "")

                return ticker

        print("Invalid choice.")
#------Financial statements------#
def finstatements(stock):
    balance_sheet = stock.balance_sheet
    financials = stock.financials
    cashflow = stock.cashflow   
    return balance_sheet,financials,cashflow
#----------------------------------#


#---------News------------#
def News(stock):
    news = stock.news
    if not news:
        return []
    articles = []
    for article in news:
        content = article.get("content") or {}
        if content.get("contentType") != "STORY":
            continue
        articles.append({
            "Title": content.get("title"),
            "Summary": content.get("summary"),
            "Date": content.get("pubDate"),
            "Publisher": content.get("provider", {}).get("displayName"),
            "URL": content.get("canonicalUrl", {}).get("url")
        })
    return articles
