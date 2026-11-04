"""Quant tools that need market prices (yfinance).
  python quant_cli.py portfolio AAPL MSFT NVDA JPM --years 3 [--weights 0.25,0.25,0.25,0.25]
  python quant_cli.py events NVDA [--market SPY]      (uses earnings dates from Yahoo, event study on those)
  python quant_cli.py macro NVDA --industry auto --shock steel=0.15 --shock rates_bp=100
Writes reports/portfolio.json / reports/TICKER_events.json and prints a summary."""
import argparse, json, os
import numpy as np, pandas as pd

def prices(tickers, years):
    import yfinance as yf
    df = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)["Close"]
    return df.dropna(how="all").ffill().dropna()

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("portfolio"); p.add_argument("tickers", nargs="+"); p.add_argument("--years", type=int, default=3)
    p.add_argument("--weights"); p.add_argument("--rf", type=float, default=0.04); p.add_argument("--max-weight", type=float, default=0.5)
    e = sub.add_parser("events"); e.add_argument("ticker"); e.add_argument("--market", default="SPY")
    s = sub.add_parser("snapshot"); s.add_argument("--sectors", action="store_true")
    m = sub.add_parser("macro"); m.add_argument("ticker"); m.add_argument("--live", action="store_true", help="replay last 12 months of observed input/rate moves"); m.add_argument("--industry", default="auto"); m.add_argument("--shock", action="append", default=[])
    a = ap.parse_args(); os.makedirs("reports", exist_ok=True)
    if a.cmd == "portfolio":
        import portfolio as P
        px = prices(a.tickers + ["SPY"], a.years); r = px.pct_change().dropna()
        mk, r = r["SPY"], r[a.tickers]
        cw = [float(x) for x in a.weights.split(",")] if a.weights else None
        res = P.analyze_portfolio(r, a.rf, cw, pd.DataFrame({"market": mk}), a.max_weight)
        res["walk_forward_oos"] = P.walk_forward_compare(r, max_w=a.max_weight)
        res["regime"] = P.regime_aware(r, max_w=a.max_weight)
        json.dump(res, open("reports/portfolio.json", "w"), default=float, indent=1)
        for k, v in res["portfolios"].items():
            print(f"{k:14s} ret {v['return']:+.1%} vol {v['vol']:.1%} sharpe {v['sharpe']:.2f} VaR95 {v['var_hist']:.2%} maxDD {v['max_drawdown']:.1%}  w={ {t: round(float(x), 2) for t, x in v['weights'].items()} }")
        print("out-of-sample Sharpe:", {k: round(v['oos_sharpe'], 2) for k, v in res["walk_forward_oos"].items()}); print(res["caveat"])
    elif a.cmd == "events":
        import yfinance as yf, event_study as ES
        t = yf.Ticker(a.ticker); px = yf.download([a.ticker, a.market], period="10y", auto_adjust=True, progress=False)["Close"].ffill().dropna()
        r = px.pct_change().dropna(); ed = t.get_earnings_dates(limit=40)
        ev = []
        for d, row in ed.iterrows():
            if pd.notna(row.get("Reported EPS")):
                s = row.get("Surprise(%)"); ev.append({"date": d.tz_localize(None).normalize(), "category": "earnings", "surprise": None if pd.isna(s) else float(s)})
        try:                                   # add 8-K item events (acquisitions, management change, results) and classified news
            import data, edgar, events as EV
            cik = data.get_cik(a.ticker)
            m8 = {"2.01": "acquisition", "1.01": "material_agreement", "5.02": "management_change", "2.02": "results_8k", "7.01": "reg_fd", "8.01": "other_8k"}
            for f in edgar.list_filings(cik, forms=("8-K",), limit=200):
                for it in (f.get("items") or "").split(","):
                    if it in m8 and it != "2.02":
                        ev.append({"date": pd.Timestamp(f["date"]), "category": m8[it], "surprise": None})
            import guidance
            for g_ in guidance.guidance_history(guidance.fetch_earnings_releases(cik, 16)):
                ev.append({"date": pd.Timestamp(g_["date"]), "category": g_["category"], "surprise": None})
            for n in (t.news or [])[:50]:
                c = EV.classify((n.get("content") or n).get("title", "")) if isinstance(n, dict) else "other"
                d = (n.get("content") or n).get("pubDate", "")[:10] if isinstance(n, dict) else ""
                if c != "other" and d:
                    ev.append({"date": pd.Timestamp(d), "category": f"news_{c}", "surprise": None})
        except Exception as e:
            print("8-K/news events skipped:", e)
        df = ES.run_events(r[a.ticker], r[a.market], ev)
        out = {"summary": ES.summarize(df), "surprise_validation": ES.surprise_validation(df), "events": df.to_dict("records")}
        json.dump(out, open(f"reports/{a.ticker}_events.json", "w"), default=str, indent=1); print(json.dumps({k: out[k] for k in ("summary", "surprise_validation")}, indent=1, default=str))
    elif a.cmd == "snapshot":
        import macro as M
        print(json.dumps(M.snapshot(), indent=1, default=str))
        if a.sectors: print(json.dumps(M.sector_performance(), indent=1))
    else:
        import data, macro as M
        if a.live:
            print(json.dumps(M.scenario_from_history(data.metrics_for(a.ticker), a.industry), indent=1, default=str)); return
        sh = {k: float(v) for k, v in (s.split("=") for s in a.shock)}
        print(json.dumps(M.simulate(data.metrics_for(a.ticker), a.industry, sh), indent=1, default=str))
main()
