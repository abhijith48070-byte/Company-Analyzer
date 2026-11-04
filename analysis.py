"""Core analysis: DCF (WACC-based), Altman Z'', Piotroski F, peer rating."""
import statistics
from concurrent.futures import ThreadPoolExecutor

from data import finstatements

peer_cache = {}


# ---------------------------------------------------------------- helpers
def get_value(df, names, column=0, default=0):
    """First matching row value from a statement (exact, then partial match)."""
    if df is None or getattr(df, "empty", True):
        return default
    for name in names:
        if name in df.index:
            try:
                v = df.loc[name].iloc[column]
                if v == v:  # not NaN
                    return v
            except Exception:
                pass
    for idx in df.index:
        for name in names:
            if name.lower() in str(idx).lower():
                try:
                    v = df.loc[idx].iloc[column]
                    if v == v:
                        return v
                except Exception:
                    pass
    return default


# ------------------------------------------------------------------- DCF
def dcfcalc(metrics, wacc=None):
    """Simple 5y FCF DCF. Uses the WACC engine when available (valuation.py)."""
    if metrics.get("Fcf") is None or metrics.get("Revenuegrowth") is None:
        return {"error": "Insufficient data for DCF"}
    if metrics["Fcf"] <= 0:
        return {"error": "FCF is negative; DCF not meaningful"}
    growth = max(min(metrics.get("Revenuegrowth") or 0.05, 0.25), -0.05)
    if wacc is None:
        try:
            from valuation import compute_wacc
            wacc = compute_wacc(metrics)["wacc"]
        except Exception:
            wacc = 0.043 + (metrics.get("Beta") or 1.0) * 0.057
    terminal_growth = 0.025
    wacc = max(wacc, terminal_growth + 0.02)

    flows = [metrics["Fcf"]]
    for i in range(5):  # linear fade of growth toward terminal growth
        g = growth + (terminal_growth - growth) * i / 5
        flows.append(flows[-1] * (1 + g))
    pv = sum(flows[i] / (1 + wacc) ** i for i in range(1, 6))
    tv = flows[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    ev = pv + tv / (1 + wacc) ** 5
    equity = ev - (metrics.get("Debt") or 0) + (metrics.get("Cash") or 0)
    shares, price = metrics.get("Sharesoutstanding"), metrics.get("Currentprice")
    if not shares:
        return {"error": "Shares Outstanding unavailable"}
    if not price:
        return {"error": "Current Price unavailable"}
    iv = equity / shares
    return {
        "current_price": price,
        "intrinsic_value": iv,
        "margin_of_safety": (iv - price) / price * 100,
        "wacc": wacc,
        "valuation": "UNDERVALUED" if iv > price else "OVERVALUED",
    }


# ------------------------------------------------------------ risk scores
def altman_z(stock=None, statements=None):
    """Altman Z'' (non-manufacturing/emerging): 3.25 + 6.56A + 3.26B + 6.72C + 1.05D.
    Zones: >5.85 safe, 4.15-5.85 grey, <4.15 distress (constant-inclusive version).
    """
    bs, fin, _ = statements or finstatements(stock)
    ta = get_value(bs, ["Total Assets"])
    tl = get_value(bs, ["Total Liabilities Net Minority Interest", "Total Liabilities"])
    if not ta or not tl:
        return {"error": "Insufficient data"}
    wc = get_value(bs, ["Current Assets", "Total Current Assets"]) - get_value(
        bs, ["Current Liabilities", "Total Current Liabilities"])
    re_ = get_value(bs, ["Retained Earnings"])
    ebit = get_value(fin, ["EBIT", "Operating Income"])
    eq = get_value(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
    z = 3.25 + 6.56 * wc / ta + 3.26 * re_ / ta + 6.72 * ebit / ta + 1.05 * eq / tl
    status = "Low risk" if z > 5.85 else "Grey zone" if z > 4.15 else "High risk"
    return {"Zscore": float(z), "Status": status}


def piotroski_f(stock=None, statements=None):
    bs, fin, cf = statements or finstatements(stock)
    g = get_value
    ni, ni0 = g(fin, ["Net Income"]), g(fin, ["Net Income"], 1)
    gp, gp0 = g(fin, ["Gross Profit"]), g(fin, ["Gross Profit"], 1)
    rev, rev0 = g(fin, ["Total Revenue", "Revenue"]), g(fin, ["Total Revenue", "Revenue"], 1)
    ocf = g(cf, ["Operating Cash Flow", "Cash Flow From Operating Activities"])
    ta, ta0 = g(bs, ["Total Assets"]), g(bs, ["Total Assets"], 1)
    d = ["Long Term Debt", "Total Debt", "Long Term Borrowings"]
    debt, debt0 = g(bs, d), g(bs, d, 1)
    ca = ["Current Assets", "Total Current Assets"]
    cl = ["Current Liabilities", "Total Current Liabilities"]
    cr = g(bs, ca) / g(bs, cl) if g(bs, cl) else 0
    cr0 = g(bs, ca, 1) / g(bs, cl, 1) if g(bs, cl, 1) else 0
    sh = ["Share Issued", "Ordinary Shares Number"]
    gm = gp / rev if rev else 0
    gm0 = gp0 / rev0 if rev0 else 0
    at = rev / ta if ta else 0
    at0 = rev0 / ta0 if ta0 else 0
    roa, roa0 = (ni / ta if ta else 0), (ni0 / ta0 if ta0 else 0)
    checks = {
        "ROA>0": ni > 0, "OCF>0": ocf > 0, "dROA>0": roa > roa0, "OCF>NI": ocf > ni,
        "debt down": debt < debt0 or (debt == 0 and debt0 == 0), "current ratio up": cr > cr0,
        "no dilution": g(bs, sh) <= g(bs, sh, 1), "gross margin up": gm > gm0,
        "asset turnover up": at > at0,
    }
    return int(sum(checks.values()))


# Backwards-compatible aliases
riskanalysis = altman_z
Riskanalysis = piotroski_f


# ------------------------------------------------------------ peer rating
def get_peer_info(ticker):
    if ticker in peer_cache:
        return peer_cache[ticker]
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        peer_cache[ticker] = info
        return info
    except Exception:
        return None


def _med(xs):
    return statistics.median(xs) if xs else None


def _score(value, median, higher_better=True):
    """0-100 score vs peer median; None when not meaningful."""
    if value is None or median is None or median <= 0:
        return None
    if higher_better:
        return None if value <= 0 else min(value / median * 100, 100)
    return 0.0 if value <= 0 else min(median / value * 100, 100)


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def rating(metrics, peer_tickers, peer_infos=None):
    if peer_infos is None:
        with ThreadPoolExecutor(max_workers=8) as ex:
            peer_infos = list(ex.map(get_peer_info, peer_tickers))
    cols = {k: [] for k in ["pm", "rg", "roe", "beta", "der", "cdr", "fcf", "roa", "eg", "pe", "fpe", "pb", "eve"]}
    for p in peer_infos:
        if not p:
            continue
        def add(k, v):
            if v is not None:
                cols[k].append(v)
        add("pm", p.get("profitMargins")); add("rg", p.get("revenueGrowth"))
        add("roe", p.get("returnOnEquity")); add("beta", p.get("beta"))
        add("der", p.get("debtToEquity")); add("fcf", p.get("freeCashflow"))
        add("roa", p.get("returnOnAssets")); add("eg", p.get("earningsGrowth"))
        add("pe", p.get("trailingPE")); add("fpe", p.get("forwardPE")); add("pb", p.get("priceToBook"))
        if p.get("totalCash") is not None and p.get("totalDebt"):
            add("cdr", p["totalCash"] / p["totalDebt"])
        if p.get("enterpriseValue") is not None and p.get("ebitda"):
            add("eve", p["enterpriseValue"] / p["ebitda"])
    m = {k: _med(v) for k, v in cols.items()}
    mt = metrics
    cd = (mt["Cash"] / mt["Debt"]) if mt.get("Debt") and mt.get("Cash") is not None else None
    growth = _avg([_score(mt.get("Revenuegrowth"), m["rg"]), _score(mt.get("Earningsgrowth"), m["eg"])])
    profit = _avg([_score(mt.get("ProfitMargin"), m["pm"]), _score(mt.get("Roe"), m["roe"]),
                   _score(mt.get("Roa"), m["roa"])])
    risk = _avg([_score(mt.get("Beta"), m["beta"], False), _score(mt.get("Deratio"), m["der"], False),
                 _score(cd, m["cdr"]) if cd is not None else 50, _score(mt.get("Fcf"), m["fcf"])])
    val = _avg([_score(mt.get("Peratio"), m["pe"], False), _score(mt.get("Peratiof"), m["fpe"], False),
                _score(mt.get("Pbratio"), m["pb"], False), _score(mt.get("EvEbidtaratio"), m["eve"], False)])
    parts = [growth, risk, profit, val]
    r = lambda x: None if x is None else round(x, 2)
    return {"overall_score": r(_avg(parts)), "profitabilityscore": r(profit), "growthscore": r(growth),
            "riskscore": r(risk), "valuationscore": r(val), "peers_used": sum(1 for p in peer_infos if p)}
