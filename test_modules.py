import discovery, competitors, events, thesis, os, tempfile
print(discovery.parse_filters("growth>0.2 and fcf>0 and pe<30 and sector~Tech"))
s = discovery.Screener(cache=os.path.join(tempfile.mkdtemp(), "u.json"))
s.rows = {"AAA": dict(sector="Technology", revenueGrowth=.3, freeCashflow=5e9, trailingPE=25, marketCap=1e11),
          "BBB": dict(sector="Technology", revenueGrowth=.1, freeCashflow=5e9, trailingPE=25, marketCap=2e11),
          "CCC": dict(sector="Energy", revenueGrowth=.4, freeCashflow=-1, trailingPE=10, marketCap=5e10)}
print(s.screen("growth>0.2 and fcf>0 and pe<30 and sector~Tech")[["ticker", "marketCap"]].to_dict("records"))
try: discovery.parse_filters("__import__('os')")
except ValueError as e: print("rejected:", e)
pc = discovery.PrivateCompanies(os.path.join(tempfile.mkdtemp(), "p.json"))
pc.upsert("Acme AI", industry="AI", founders=["A. Founder"]); pc.add_round("Acme AI", "2025-03-01", "Series A", 20e6, 100e6, ["Fund X"], "press release")
pc.add_round("Acme AI", "2026-02-01", "Series B", 80e6, 600e6, ["Fund Y", "Fund X"], "press release"); print({k: v for k, v in pc.summary("Acme AI").items() if k != "rounds"})
tgt = dict(totalRevenue=100e9, freeCashflow=10e9, revenueGrowth=.1, grossMargins=.5, operatingMargins=.2, returnOnEquity=.3, debtToEquity=50, trailingPE=25, enterpriseValue=1e12, ebitda=30e9, marketCap=1e12)
peers = {f"P{i}": {**tgt, "revenueGrowth": .02 * i, "operatingMargins": .1 + .02 * i, "marketCap": 1e12 * (1 + i / 10)} for i in range(1, 12)}
c = competitors.compare("TGT", tgt, peers); print(c["percentile_vs_sector"], c["revenue_share_of_peer_set_pct"], [t["ticker"] for t in c["threats"]])
tl = events.build_timeline(news=[{"Title": "Company sued over patents", "Date": "2026-10-20", "URL": "u"}, {"Title": "Analyst upgrade lifts shares", "Date": "2026-10-25", "URL": "u2"}],
                           filings=[{"form": "8-K", "date": "2026-10-10", "items": "2.02,9.01", "url": "u3"}], earnings_dates=["2026-11-20"])
print([(e["date"], e["type"], e["status"]) for e in tl["events"]], events.next_earnings(["2026-11-20"]))
res = dict(metrics=dict(Name="Tgt", Sector="Tech", Industry="Semis", Summary="Makes chips.", Revenue=100e9, GrossMargin=.5, OperatingMargin=.2,
           ProfitMargin=.15, Roe=.3, Fcf=10e9, Cash=5e9, Debt=2e9, Revenuegrowth=.3, Currentprice=100),
           valuation=dict(wacc=dict(wacc=.1, cost_of_equity=.1, beta=1.2), reverse_dcf=dict(implied_fcf_growth=.4, implied_fcf_margin=.3, current_fcf_margin=.1),
           scenarios={k: dict(growth=g, wacc=.1, terminal_growth=.025, value=v) for k, g, v in (("bear", .1, 30), ("base", .2, 50), ("bull", .3, 90))}),
           forensics=dict(warnings=[dict(signal="Weak cash conversion", severity="medium", evidence="OCF/NI 0.5")], facts=dict(cash_conversion_ocf_to_ni=.5, sloan_accrual_ratio=.12)),
           altman=dict(Zscore=7), competitors=c, events=dict(events=tl["events"], next_earnings="2026-11-20"), ml={})
md = thesis.render_thesis(thesis.build_thesis(res), "Tgt"); print(md[-900:]); assert "FACT" in md and "Buy" not in md
print("OK")
