"""Offline end-to-end test with synthetic statements (no network)."""
import pandas as pd, data, analyze

def df(rows):  # rows: {name: [cur, prev, prev2]}
    return pd.DataFrame(rows, index=["2025", "2024", "2023"]).T

bs = df({"Total Assets": [1800, 1500, 1300], "Current Assets": [800, 600, 500], "Current Liabilities": [400, 350, 300],
         "Retained Earnings": [300, 200, 120], "Total Liabilities Net Minority Interest": [1100, 950, 800],
         "Stockholders Equity": [700, 550, 500], "Accounts Receivable": [300, 180, 150], "Inventory": [250, 150, 120],
         "Net PPE": [500, 480, 450], "Long Term Debt": [600, 500, 450], "Total Debt": [600, 500, 450],
         "Cash And Cash Equivalents": [100, 90, 80], "Ordinary Shares Number": [100, 100, 100]})
inc = df({"Total Revenue": [1200, 1000, 900], "Cost Of Revenue": [720, 550, 500], "Gross Profit": [480, 450, 400],
          "Selling General And Administration": [200, 180, 170], "EBIT": [230, 200, 170], "Interest Expense": [40, 35, 30],
          "Net Income": [150, 120, 100]})
cf = df({"Operating Cash Flow": [60, 130, 110], "Capital Expenditure": [-80, -70, -60], "Depreciation And Amortization": [50, 45, 40]})

class Fake: balance_sheet, financials, cashflow = bs, inc, cf
info = dict(longName="Synthetic Co", marketCap=3000, totalRevenue=1200, enterpriseValue=3500, trailingPE=20, forwardPE=18,
            netIncomeToCommon=150, revenueGrowth=0.2, profitMargins=0.125, returnOnEquity=0.21, debtToEquity=85, totalCash=100,
            totalDebt=600, freeCashflow=-20, ebitda=280, priceToBook=4, earningsGrowth=0.25, beta=1.1, returnOnAssets=0.08,
            sharesOutstanding=100, currentPrice=30, sector="Technology", industry="Software", operatingMargins=0.19,
            dividendRate=0.5, bookValue=7)
data.get_company_info = lambda t: (Fake(), info)
data.stock_cache.clear()
r = analyze_res = analyze.analyze("SYN")
md = analyze.render(r)
assert "Beneish" in md and "Balance check" in md
assert max(abs(y["balance_check"]) for y in r["projection"]) < 1e-6, "projection must balance"
print(md[:3500]); print("\nOK")
