"""Competitive positioning from peer data (Yahoo `info` dicts) + target ROIC from statements."""
import statistics as st

METRICS = {"revenueGrowth": ("Revenue growth", True), "grossMargins": ("Gross margin", True),
           "operatingMargins": ("Operating margin", True), "fcfMargin": ("FCF margin", True),
           "returnOnEquity": ("ROE", True), "debtToEquity": ("Debt/Equity", False), "trailingPE": ("P/E", None),
           "evToEbitda": ("EV/EBITDA", None)}


def roic(p, tax=0.21):
    """NOPAT / invested capital (debt + equity - cash) from a normalised period."""
    ic = p["total_debt"] + p["equity"] - p["cash"]
    return p["ebit"] * (1 - tax) / ic if ic > 0 else None


def _row(info):
    r = dict(info)
    r["fcfMargin"] = info["freeCashflow"] / info["totalRevenue"] if info.get("freeCashflow") is not None and info.get("totalRevenue") else None
    r["evToEbitda"] = info["enterpriseValue"] / info["ebitda"] if info.get("enterpriseValue") and info.get("ebitda") and info["ebitda"] > 0 else None
    return r


def compare(ticker, target_info, peer_infos, k=8):
    """peer_infos: {ticker: info}. Returns table, percentile ranks, share of peer-set revenue, and threats."""
    t = _row(target_info)
    peers = {p: _row(i) for p, i in peer_infos.items() if p != ticker and i}
    near = dict(sorted(peers.items(), key=lambda kv: abs((kv[1].get("marketCap") or 0) - (t.get("marketCap") or 0)))[:k])
    table = [{"ticker": ticker, **{lbl: t.get(f) for f, (lbl, _) in METRICS.items()}}]
    table += [{"ticker": p, **{lbl: r.get(f) for f, (lbl, _) in METRICS.items()}} for p, r in near.items()]
    pct = {}
    for f, (lbl, hi) in METRICS.items():
        vals = [r[f] for r in peers.values() if r.get(f) is not None]
        if t.get(f) is not None and len(vals) >= 5 and hi is not None:
            below = sum(v < t[f] for v in vals) / len(vals) * 100
            pct[lbl] = below if hi else 100 - below
    rev = {p: r.get("totalRevenue") for p, r in {**peers, ticker: t}.items() if r.get("totalRevenue")}
    share = rev.get(ticker, 0) / sum(rev.values()) * 100 if rev and ticker in rev else None
    threats = []
    for p, r in near.items():
        if all(r.get(f) is not None and t.get(f) is not None for f in ("revenueGrowth", "operatingMargins")):
            if r["revenueGrowth"] > (t["revenueGrowth"] or 0) + 0.05 and r["operatingMargins"] >= (t["operatingMargins"] or 0):
                threats.append({"ticker": p, "evidence": f"growth {r['revenueGrowth']:.1%} vs {t['revenueGrowth']:.1%}, "
                                f"operating margin {r['operatingMargins']:.1%} vs {t['operatingMargins']:.1%}"})
    return {"table": table, "percentile_vs_sector": pct, "revenue_share_of_peer_set_pct": share, "threats": threats,
            "note": "Percentile = better than X% of S&P 500 sector peers. Share is within the compared set, not true market share."}
