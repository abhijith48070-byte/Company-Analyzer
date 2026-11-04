"""Orchestration of the optional engines (ML, competitors, events, RAG, tracker) used by analyze.py, api.py and dashboard.py."""
import os
from concurrent.futures import ThreadPoolExecutor

import competitors as comp_mod
import data
import events as ev_mod
import ml
from analysis import get_peer_info
from fin_model import normalize


def peer_info_map(ticker, sector, max_peers=25):
    pt = [t for t in data.peertickers(sector) if t != ticker][:max_peers]
    with ThreadPoolExecutor(6) as ex:
        infos = list(ex.map(get_peer_info, pt))
    return {t: i for t, i in zip(pt, infos) if i}


def fetch_periods(t):
    stock, _ = data.get_company_info(t)
    return normalize(data.finstatements(stock), periods=4)


def earnings_rows(stock):
    try:
        ed = stock.get_earnings_dates(limit=40)
        ed = ed.dropna(subset=["Reported EPS", "EPS Estimate"]).sort_index()
        return [{"date": str(i)[:10], "estimate": float(r["EPS Estimate"]), "actual": float(r["Reported EPS"])} for i, r in ed.iterrows()]
    except Exception:
        return []


def upcoming_earnings_dates(stock):
    try:
        return [str(i)[:10] for i in stock.get_earnings_dates(limit=40).index]
    except Exception:
        return []


def run_ml(ticker, stock, peer_tickers, target_periods, sector="sector"):
    """All ML models. Panels are cached on disk (cache/). Every result carries its validation numbers."""
    out = {}
    pbt = ml.panel_cached(peer_tickers, f"cache/panel_{str(sector).replace(' ', '_')}.pkl", fetch_periods)
    pbt.pop(ticker, None)
    panel = ml.build_panel(pbt, drop=0.5)
    out["panel"] = {"companies": len(pbt), "rows": len(panel)}
    if panel.empty or len(target_periods) < 2:
        return {**out, "error": "not enough peer history"}
    d = ml.fit_distress(panel)
    out["distress_validation"] = {k: v for k, v in d.items() if k != "model"}
    if "model" in d:
        out["distress_probability"] = ml.predict_distress(d, target_periods)
    out["anomaly"] = ml.anomaly_scan(panel, target_periods)
    for key, tgt in (("growth", "y_next_growth"), ("margin", "y_next_opm")):
        f = ml.fit_forecaster(panel, tgt)
        out[f"{key}_validation"] = {k: v for k, v in f.items() if k not in ("model", "cols")}
        if "model" in f:
            out[f"{key}_forecast"] = ml.forecast_next(f, target_periods)
    out["earnings_surprise"] = ml.earnings_surprise_model(earnings_rows(stock))
    return out


def run_competitors(ticker, target_info, peer_infos, target_periods=None):
    c = comp_mod.compare(ticker, target_info, peer_infos)
    if target_periods:
        c["target_roic"] = comp_mod.roic(target_periods[0])
    return c


def run_events(stock, filings=None):
    news = data.News(stock)
    dates = upcoming_earnings_dates(stock)
    tl = ev_mod.build_timeline(news=news, filings=filings or [], earnings_dates=dates)
    tl["next_earnings"] = ev_mod.next_earnings(dates)
    try:
        import ml as _ml
        texts = [f"{n.get('Title', '')}. {n.get('Summary', '') or ''}" for n in news]
        tl["sentiment"] = _ml.sentiment(texts) if texts else []
        if tl["sentiment"]:
            tl["avg_sentiment"] = sum(s["score"] for s in tl["sentiment"]) / len(tl["sentiment"])
    except Exception:
        pass
    return tl


def build_rag(ticker, cik, news=None, max_10k=2, max_10q=1):
    """Index recent filings + news. Returns FilingIndex (saved to cache/)."""
    import edgar
    from rag import FilingIndex
    idx = FilingIndex()
    for form, n in (("10-K", max_10k), ("10-Q", max_10q)):
        for f in edgar.list_filings(cik, forms=(form,), limit=n):
            idx.add_filing({"form": f["form"], "date": f["date"], "url": f["url"]}, edgar.get_filing_text(f))
    for n in news or []:
        if n.get("Summary"):
            idx.add_document(n.get("Publisher") or "News", f"{n['Title']}. {n['Summary']}", n.get("Date", ""), n.get("URL", ""))
    idx.build()
    os.makedirs("cache", exist_ok=True)
    idx.save(f"cache/rag_{ticker}.pkl")
    return idx


def collect_deals(ticker, cik, tracker, limit=10):
    """Auto-extract acquisition candidates from recent 8-Ks (Items 1.01/2.01) into the tracker, UNVERIFIED."""
    import edgar
    from tracker import extract_deals_from_8k
    found = 0
    for f in edgar.list_filings(cik, forms=("8-K",), limit=limit):
        if any(i in (f.get("items") or "") for i in ("1.01", "2.01")):
            for d in extract_deals_from_8k(edgar.get_filing_text(f), f):
                tracker.add_transaction("acquisition", ticker, d["target"], d["date"], d["value_usd"],
                                        rationale="auto-extracted; verify", source=d["source"], verified=False)
                found += 1
    return found


def render_extras(res):
    pc = lambda x: "n/a" if x is None else f"{x * 100:.1f}%"
    L = []
    ml_ = res.get("ml")
    if ml_:
        L.append("## ML intelligence (validated on later years; small samples flagged)\n")
        L.append(f"Peer panel: {ml_.get('panel', {})}")
        dv = ml_.get("distress_validation", {})
        if "error" in dv:
            L.append(f"- Distress model: {dv['error']}")
        elif dv:
            b = dv.get("best")
            L.append(f"- **Distress model** best={b}, AUC {dv['models'][b]['auc']:.2f} on {dv['n_test']} test rows (year {dv['test_year']}); "
                     f"Brier {dv['models'][b]['brier']:.3f} vs baseline {dv['baseline_brier']:.3f}. "
                     f"P(material Altman Z'' deterioration next year) = {pc(ml_.get('distress_probability'))}. "
                     f"Top drivers: {', '.join(dv.get('feature_importance', {}))}")
        an = ml_.get("anomaly", {})
        if "anomaly_percentile" in an:
            L.append(f"- **Anomaly (Isolation Forest):** more unusual than {an['anomaly_percentile']:.0f}% of {an['n_peers']} peer-years. "
                     "Most unusual: " + "; ".join(f"{k} {v['value']:.3f} (peer median {v['peer_median']:.3f})" for k, v in an["most_unusual_ratios"].items()))
        for key, label in (("growth", "Next-year revenue growth"), ("margin", "Next-year operating margin")):
            fc, val = ml_.get(f"{key}_forecast"), ml_.get(f"{key}_validation", {})
            if fc:
                L.append(f"- **{label} (Bayesian ridge):** {pc(fc['mean'])} (90% interval {pc(fc['low90'])} to {pc(fc['high90'])}); "
                         f"test MAE {val['mae_model']:.3f} vs persistence {val['mae_persistence']:.3f} -> "
                         f"{'beats' if val['beats_baseline'] else 'does NOT beat'} the naive baseline; interval coverage {pc(val['interval90_coverage'])}")
            elif val.get("error"):
                L.append(f"- {label}: {val['error']}")
        es = ml_.get("earnings_surprise") or {}
        L.append(f"- **Earnings beat model:** " + (es["error"] if "error" in es else
                 f"P(beat next) {pc(es['p_beat_next'])}; test accuracy {pc(es['accuracy'])} vs always-majority {pc(es['baseline_accuracy'])} "
                 f"({'beats' if es['beats_baseline'] else 'does NOT beat'} baseline, n_test={es['n_test']}); historical beat rate {pc(es['historical_beat_rate'])}"))
        L.append("")
    c = res.get("competitors")
    if c:
        L.append("## Competitive position\n")
        keys = [k for k in c["table"][0] if k != "ticker"]
        L.append("| Ticker | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys))
        for r in c["table"]:
            L.append(f"| {r['ticker']} | " + " | ".join("n/a" if r[k] is None else (f"{r[k]:.1%}" if abs(r[k]) < 5 and k not in ("P/E", "EV/EBITDA") else f"{r[k]:.1f}") for k in keys) + " |")
        L.append("\nPercentile vs S&P 500 sector peers: " + ", ".join(f"{k} {v:.0f}" for k, v in c["percentile_vs_sector"].items()))
        if c.get("target_roic") is not None:
            L.append(f"\nROIC (NOPAT / invested capital): {pc(c['target_roic'])}")
        for t in c["threats"]:
            L.append(f"- Emerging threat: **{t['ticker']}** ({t['evidence']})")
        L.append(f"\n_{c['note']}_\n")
    e = res.get("events")
    if e:
        L.append("## Event timeline\n")
        L.append("| Date | Type | Event | Status |\n|---|---|---|---|")
        for x in e["events"][:15]:
            L.append(f"| {x['date']} | {x['type']} | {x['title']} | {x['status']} |")
        if e.get("avg_sentiment") is not None:
            L.append(f"\nAverage headline sentiment: {e['avg_sentiment']:+.2f} ({e['sentiment'][0]['method']}).")
        L.append(f"\n_{e['caveat']}_\n")
    for qa in res.get("rag", []):
        L.append(f"## Q: {qa['question']}\n\n{qa['answer']}\n")
        for s in qa["sources"]:
            L.append(f"- [{s['n']}] {s['citation']}")
        if qa["unsupported_numbers"]:
            L.append(f"\n> Numbers in the answer not found in the sources: {qa['unsupported_numbers']}")
        L.append("")
    return "\n".join(L)
