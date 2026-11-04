"""Structured company facts (USD) from yfinance statements, plus a synthetic generator for offline tests."""
import numpy as np
import pandas as pd

ITEMS = {  # claim wording -> (statement, yfinance row name)
    "revenue": ("financials", "Total Revenue"),
    "gross profit": ("financials", "Gross Profit"),
    "operating income": ("financials", "Operating Income"),
    "net income": ("financials", "Net Income"),
    "operating cash flow": ("cashflow", "Operating Cash Flow"),
    "free cash flow": ("cashflow", "Free Cash Flow"),
    "total assets": ("balance_sheet", "Total Assets"),
    "total debt": ("balance_sheet", "Total Debt"),
}
MARGINS = {"gross margin": ("gross profit", "revenue"), "operating margin": ("operating income", "revenue"),
           "net margin": ("net income", "revenue")}


def from_statements(ticker, name, statements):
    facts = {}
    for item, (stmt, row) in ITEMS.items():
        df = statements.get(stmt)
        if df is None or row not in df.index:
            continue
        s = df.loc[row].dropna()
        facts[item] = {int(pd.Timestamp(c).year): float(v) for c, v in s.items()}
    return dict(ticker=ticker, name=name, facts=facts)


def load_yf(ticker):
    import yfinance as yf
    t = yf.Ticker(ticker)
    name = (t.info or {}).get("longName") or ticker
    return from_statements(ticker, name, dict(financials=t.financials, balance_sheet=t.balance_sheet, cashflow=t.cashflow))


def synthetic(n=12, seed=0, years=(2021, 2022, 2023, 2024)):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        base = float(rng.uniform(2e9, 4e11))
        g = rng.uniform(-0.05, 0.25)
        gm, om, nm = rng.uniform(0.25, 0.7), rng.uniform(0.05, 0.3), rng.uniform(0.03, 0.25)
        facts = {k: {} for k in ITEMS}
        for j, y in enumerate(years):
            rev = base * (1 + g) ** j * rng.uniform(0.97, 1.03)
            facts["revenue"][y] = rev
            facts["gross profit"][y] = rev * gm * rng.uniform(0.95, 1.05)
            facts["operating income"][y] = rev * om * rng.uniform(0.9, 1.1)
            facts["net income"][y] = rev * nm * rng.uniform(0.9, 1.1)
            facts["operating cash flow"][y] = rev * nm * rng.uniform(1.0, 1.6)
            facts["free cash flow"][y] = facts["operating cash flow"][y] * rng.uniform(0.5, 0.9)
            facts["total assets"][y] = rev * rng.uniform(1.0, 3.0)
            facts["total debt"][y] = rev * rng.uniform(0.1, 0.8)
        out.append(dict(ticker=f"SYN{i:02d}", name=f"Synthetic Co {i:02d}", facts=facts))
    return out
