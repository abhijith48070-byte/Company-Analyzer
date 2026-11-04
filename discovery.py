"""Public-company screener (explicit filter DSL, optional LLM translation) and a private-company store."""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ALIASES = {"growth": "revenueGrowth", "revenue_growth": "revenueGrowth", "margin": "profitMargins",
           "profit_margin": "profitMargins", "roe": "returnOnEquity", "fcf": "freeCashflow", "pe": "trailingPE",
           "fwd_pe": "forwardPE", "debt_equity": "debtToEquity", "mcap": "marketCap", "market_cap": "marketCap",
           "ev_ebitda": "evToEbitda", "pb": "priceToBook", "earnings_growth": "earningsGrowth", "beta": "beta"}
_OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
        "!=": lambda a, b: a != b, "=": lambda a, b: a == b, "~": lambda a, b: str(b).lower() in str(a).lower()}
_TOKEN = re.compile(r"^\s*([A-Za-z_]+)\s*(>=|<=|!=|>|<|=|~)\s*(.+?)\s*$")


def parse_filters(text):
    """'growth>0.2 and fcf>0 and pe<30 and sector~Tech' -> [(field, op, value)]. No eval()."""
    out = []
    for part in re.split(r"\s+and\s+", text.strip(), flags=re.I):
        m = _TOKEN.match(part)
        if not m:
            raise ValueError(f"cannot parse filter: {part!r}")
        f, op, v = m.groups()
        f = ALIASES.get(f.lower(), f)
        v = v.strip("'\"")
        try:
            v = float(v.replace("B", "e9").replace("M", "e6").replace("T", "e12"))
        except ValueError:
            pass
        out.append((f, op, v))
    return out


class Screener:
    def __init__(self, cache="cache/universe.json"):
        self.cache, self.rows = cache, {}
        if os.path.exists(cache):
            self.rows = json.load(open(cache))

    def load(self, tickers, workers=8, refresh=False):
        import yfinance as yf
        todo = [t for t in tickers if refresh or t not in self.rows]

        def get(t):
            try:
                i = yf.Ticker(t).info
                keep = ["longName", "sector", "industry", "marketCap", "revenueGrowth", "earningsGrowth", "profitMargins",
                        "returnOnEquity", "returnOnAssets", "freeCashflow", "debtToEquity", "trailingPE", "forwardPE",
                        "priceToBook", "beta", "enterpriseValue", "ebitda", "totalRevenue"]
                d = {k: i.get(k) for k in keep}
                d["evToEbitda"] = d["enterpriseValue"] / d["ebitda"] if d.get("enterpriseValue") and d.get("ebitda") else None
                return t, d
            except Exception:
                return t, None
        with ThreadPoolExecutor(workers) as ex:
            for t, d in ex.map(get, todo):
                if d:
                    self.rows[t] = d
        os.makedirs(os.path.dirname(self.cache) or ".", exist_ok=True)
        json.dump(self.rows, open(self.cache, "w"))
        return self

    def screen(self, filters, sort="marketCap", limit=25):
        fl = parse_filters(filters) if isinstance(filters, str) else filters
        hits = []
        for t, d in self.rows.items():
            ok = True
            for f, op, v in fl:
                x = d.get(f)
                if x is None or not _OPS[op](x, v):
                    ok = False; break
            if ok:
                hits.append({"ticker": t, **d})
        df = pd.DataFrame(hits)
        return df.sort_values(sort, ascending=False).head(limit) if not df.empty and sort in df else df


def nl_to_filters(llm, request):
    """Ask an LLM to translate English into the DSL, then validate with parse_filters (never executed)."""
    p = ("Translate the request into a filter string using only these fields: growth, margin, roe, fcf, pe, fwd_pe, "
         "debt_equity, mcap, ev_ebitda, pb, earnings_growth, beta, sector, industry. Operators: > >= < <= = != ~ (contains). "
         "Join with ' and '. Ratios are decimals (20% = 0.2). Reply with the filter string only.\n\nRequest: " + request)
    txt = llm(p).strip().strip("`").splitlines()[0]
    parse_filters(txt)
    return txt


class PrivateCompanies:
    """Manually curated store. Populate from S-1 filings, press releases, funding announcements; every round carries a source."""
    def __init__(self, path="private_companies.json"):
        self.path = path
        self.data = json.load(open(path)) if os.path.exists(path) else {}

    def _save(self):
        json.dump(self.data, open(self.path, "w"), indent=1)

    def upsert(self, name, **fields):
        c = self.data.setdefault(name, {"name": name, "rounds": [], "status": "private"})
        c.update({k: v for k, v in fields.items() if v is not None}); self._save(); return c

    def add_round(self, name, date, series, amount_usd=None, valuation_usd=None, investors=(), source=""):
        c = self.upsert(name)
        c["rounds"].append({"date": date, "series": series, "amount_usd": amount_usd, "valuation_usd": valuation_usd,
                            "investors": list(investors), "source": source})
        c["rounds"].sort(key=lambda r: r["date"]); self._save()

    def summary(self, name):
        c = self.data[name]
        rs = c["rounds"]
        return {**{k: v for k, v in c.items() if k != "rounds"}, "n_rounds": len(rs),
                "total_raised_usd": sum(r["amount_usd"] or 0 for r in rs),
                "last_valuation_usd": next((r["valuation_usd"] for r in reversed(rs) if r["valuation_usd"]), None),
                "investors": sorted({i for r in rs for i in r["investors"]}), "rounds": rs}

    def search(self, text="", industry=None, status=None):
        t = text.lower()
        return [self.summary(n) for n, c in self.data.items()
                if (not t or t in json.dumps(c).lower()) and (not industry or industry.lower() in str(c.get("industry", "")).lower())
                and (not status or c.get("status") == status)]
