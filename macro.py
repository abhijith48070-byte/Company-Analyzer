"""Macro/industry shock simulation using transparent company-specific assumptions (no data feed needed).
Shock inputs are fractions (0.15 = +15%). The user supplies cost-structure shares; defaults are labelled illustrative."""
import numpy as np

ILLUSTRATIVE = {  # share of revenue
    "auto": {"inputs": {"steel": 0.10, "aluminium": 0.04, "energy": 0.03}, "pass_through": 0.3, "debt_floating_share": 0.3, "fx_foreign_rev_share": 0.5},
    "airline": {"inputs": {"jet_fuel": 0.25}, "pass_through": 0.5, "debt_floating_share": 0.4, "fx_foreign_rev_share": 0.4},
    "software": {"inputs": {"energy": 0.01}, "pass_through": 0.0, "debt_floating_share": 0.1, "fx_foreign_rev_share": 0.5},
}


def cost_shock(revenue, op_margin, input_share, shock, pass_through=0.0):
    """Margin impact of an input-price shock. input_share = input cost / revenue. Returns new margin and delta in percentage points."""
    extra_cost = revenue * input_share * shock
    recovered = extra_cost * pass_through
    new_ebit = revenue * op_margin - extra_cost + recovered
    new_margin = new_ebit / revenue
    return {"old_margin": op_margin, "new_margin": new_margin, "delta_pp": (new_margin - op_margin) * 100, "ebit_change": new_ebit - revenue * op_margin}


def rate_shock(ebit, interest_expense, floating_debt, bp):
    """Change in interest expense and interest coverage for a parallel move of `bp` basis points on floating-rate debt."""
    extra = floating_debt * bp / 1e4
    cov0 = ebit / interest_expense if interest_expense else None
    cov1 = ebit / (interest_expense + extra) if interest_expense + extra else None
    return {"extra_interest": extra, "coverage_before": cov0, "coverage_after": cov1}


def fx_shock(revenue, op_margin, foreign_share, fx_move, cost_in_foreign_share=0.0):
    """Translation effect of foreign currency moving `fx_move` vs reporting currency (negative = foreign weakens). Natural hedge via foreign-currency costs."""
    d_rev = revenue * foreign_share * fx_move
    d_cost = revenue * (1 - op_margin) * cost_in_foreign_share * fx_move
    return {"ebit_change": d_rev - d_cost, "new_margin": (revenue * op_margin + d_rev - d_cost) / (revenue + d_rev)}


def simulate(metrics, industry="auto", shocks=None, assumptions=None):
    """shocks: {'steel':0.15,'rates_bp':100,'fx':-0.10,...}. assumptions overrides ILLUSTRATIVE[industry]. Uses metrics Revenue/OperatingMargin/Debt."""
    a = {**ILLUSTRATIVE.get(industry, ILLUSTRATIVE["software"]), **(assumptions or {})}
    rev, om = metrics.get("Revenue") or metrics.get("Totalrevenue"), metrics.get("OperatingMargin")
    if not rev or om is None:
        return {"error": "need Revenue and OperatingMargin in metrics"}
    out, ebit = {"assumptions": a, "assumption_status": "ILLUSTRATIVE unless overridden: replace with company disclosures (10-K supplier/commodity notes)"}, rev * om
    d_ebit = 0.0
    for k, sh in (shocks or {}).items():
        if k in a["inputs"]:
            r = cost_shock(rev, om, a["inputs"][k], sh, a["pass_through"]); out[k] = r; d_ebit += r["ebit_change"]
        elif k == "rates_bp":
            debt = (metrics.get("Debt") or 0) * a["debt_floating_share"]
            r = rate_shock(ebit, max((metrics.get("Debt") or 0) * 0.04, 1), debt, sh); out[k] = r; d_ebit -= r["extra_interest"]
        elif k == "fx":
            r = fx_shock(rev, om, a["fx_foreign_rev_share"], sh); out[k] = r; d_ebit += r["ebit_change"]
    out["total_ebit_change"] = d_ebit
    out["pct_of_ebit"] = d_ebit / ebit if ebit else None
    return out


# ------------------------------------------------------------- data feeds
FRED = {"rate_10y": "DGS10", "rate_2y": "DGS2", "fed_funds": "FEDFUNDS", "cpi": "CPIAUCSL", "gdp_real": "GDPC1", "unemployment": "UNRATE",
        "usd_eur": "DEXUSEU", "usd_jpy": "DEXJPUS", "usd_cny": "DEXCHUS", "oil_wti": "DCOILWTICO", "copper": "PCOPPUSDM",
        "steel_ppi": "WPU101", "natgas": "DHHNGSP"}
YF_FALLBACK = {"rate_10y": "^TNX", "oil_wti": "CL=F", "copper": "HG=F", "usd_eur": "EURUSD=X", "natgas": "NG=F"}
SECTOR_ETF = {"Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV", "Energy": "XLE", "Consumer Cyclical": "XLY",
              "Consumer Defensive": "XLP", "Industrials": "XLI", "Basic Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE", "Communication Services": "XLC"}


def parse_fred_csv(text):
    """FRED fredgraph.csv text -> list of (date, float), skipping missing '.' values."""
    rows = []
    for line in text.strip().splitlines()[1:]:
        d, _, v = line.partition(",")
        try:
            rows.append((d.strip(), float(v)))
        except ValueError:
            continue
    return rows


def fetch_series(name, fetch=None):
    """Latest-first [(date, value)] for a macro series. FRED public CSV (no key); yfinance fallback; None if both unreachable."""
    if fetch is None:
        import requests
        fetch = lambda url: requests.get(url, timeout=20).text
    try:
        rows = parse_fred_csv(fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED[name]}"))
        if rows:
            return rows[::-1]
    except Exception:
        pass
    try:
        import yfinance as yf
        h = yf.Ticker(YF_FALLBACK[name]).history(period="3y")["Close"].dropna()
        return [(str(i.date()), float(v)) for i, v in h.items()][::-1]
    except Exception:
        return None


def change_over(rows, days=365):
    """(latest, value ~days earlier, pct change) from a latest-first series."""
    import datetime as dt
    if not rows:
        return None
    t0 = dt.date.fromisoformat(rows[0][0]) - dt.timedelta(days=days)
    old = next((v for d, v in rows if dt.date.fromisoformat(d) <= t0), None)
    return {"latest": rows[0][1], "as_of": rows[0][0], "year_ago": old, "pct_change": (rows[0][1] / old - 1) if old else None,
            "abs_change": (rows[0][1] - old) if old is not None else None}


def snapshot(names=None, fetch=None):
    """Macro dashboard: latest value and 1y change per series. CPI/GDP/steel are index levels: pct_change is the inflation / growth / price change."""
    return {n: (change_over(fetch_series(n, fetch)) or {"error": "unreachable"}) for n in (names or FRED)}


def sector_performance(days="3mo"):
    try:
        import yfinance as yf
        px = yf.download(list(SECTOR_ETF.values()), period=days, auto_adjust=True, progress=False)["Close"]
        r = (px.iloc[-1] / px.iloc[0] - 1)
        return {s: float(r[e]) for s, e in SECTOR_ETF.items() if e in r}
    except Exception:
        return {"error": "sector data unreachable"}


INDUSTRY_DRIVERS = {"auto": {"steel": "steel_ppi", "aluminium": "copper", "energy": "natgas"}, "airline": {"jet_fuel": "oil_wti"},
                    "software": {"energy": "natgas"}}


def scenario_from_history(metrics, industry="auto", fetch=None, rates_bp=None, fx=None):
    """Replay the last 12 months of observed input-price moves as a shock on this company (what the past year's macro did to margins, holding cost shares fixed)."""
    shocks = {}
    for inp, series in INDUSTRY_DRIVERS.get(industry, {}).items():
        c = change_over(fetch_series(series, fetch))
        if c and c["pct_change"] is not None:
            shocks[inp] = c["pct_change"]
    if rates_bp is None:
        r = change_over(fetch_series("rate_10y", fetch))
        rates_bp = (r["abs_change"] * (100 if r["latest"] < 20 else 1)) if r and r["abs_change"] is not None else None
    if rates_bp is not None:
        shocks["rates_bp"] = rates_bp
    if fx is not None:
        shocks["fx"] = fx
    return {"observed_shocks": shocks, **simulate(metrics, industry, shocks)}
