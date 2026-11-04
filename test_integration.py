"""Offline integration: fake Yahoo/EDGAR at the boundaries, run the real pipeline code paths."""
import numpy as np, pandas as pd
import test_pipeline as tp, test_ml, data, pipeline, analyze, thesis

peers = [t for t in data.peertickers("Technology")][:40]
periods = {t: test_ml.synth_company(i) for i, t in enumerate(peers)}
pipeline.fetch_periods = lambda t: periods[t]
rng = np.random.default_rng(3)
pipeline.get_peer_info = lambda t: dict(tp.info, revenueGrowth=float(rng.uniform(0, .4)), operatingMargins=float(rng.uniform(.05, .3)),
        grossMargins=.5, freeCashflow=1e9, totalRevenue=1e10, returnOnEquity=.2, debtToEquity=60, trailingPE=22, enterpriseValue=1e11, ebitda=5e9, marketCap=float(rng.uniform(1e10, 1e12)), profitMargins=.1, returnOnAssets=.08, priceToBook=3, earningsGrowth=.1, beta=1.1, totalCash=1e9, totalDebt=2e9)
analyze.get_peer_info = pipeline.get_peer_info
import analysis; analysis.get_peer_info = pipeline.get_peer_info
data.News = lambda s: [{"Title": "Company beats estimates and raises guidance", "Summary": "Record quarter.", "Date": "2026-10-28", "URL": "u", "Publisher": "Wire"}]
tp.info["freeCashflow"] = 150; data.stock_cache.clear()
import os, shutil; shutil.rmtree("cache", ignore_errors=True)
r = analyze.analyze("SYN", peers=True, edgar=False, max_peers=40, do_ml=True, do_comp=True, do_events=True, do_thesis=True)
md = analyze.render(r)
for sec in ("## ML intelligence", "## Competitive position", "## Event timeline", "Peer scores"): assert sec in md, sec
print(md[md.index("## ML intelligence"):md.index("## Competitive position")][:1800])
t = thesis.render_thesis(r["thesis"], "Synthetic Co"); assert "[MODEL]" in t and "[FACT]" in t; print(t[-500:])
print("INTEGRATION OK")
