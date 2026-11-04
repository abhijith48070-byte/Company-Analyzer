import numpy as np, pandas as pd, portfolio as P, montecarlo as MC, event_study as ES, macro as M
rng = np.random.default_rng(1)
n, T = 5, 1500
cov = np.diag([.0002, .0003, .0004, .0005, .0001]) + 0.00005
r = pd.DataFrame(rng.multivariate_normal(np.array([.0006,.0005,.0004,.0007,.0002]), cov, T), columns=list("ABCDE"),
                 index=pd.bdate_range("2018-01-01", periods=T))
a = P.analyze_portfolio(r, rf=0.02, current_w=[.2]*5, factors=pd.DataFrame({"mkt": r.mean(axis=1)}))
for k, v in a["portfolios"].items(): assert abs(sum(v["weights"].values()) - 1) < 1e-3, k
mu, c = P.stats(r)
assert P.perf(P.min_vol(mu, c), mu, c)["vol"] <= P.perf(np.full(5, .2), mu, c)["vol"] + 1e-9
rp = P.risk_parity(c); rc = rp * (c @ rp); assert rc.std() / rc.mean() < 0.05
fr = a["frontier"]; assert all(fr[i]["vol"] <= fr[i+1]["vol"] + 1e-6 for i in range(len(fr)-1))
post, pi = P.black_litterman(c, np.full(5, .2), np.array([[1, 0, 0, 0, 0]]), [0.30]); assert post[0] > pi[0]
print("regime", P.regime_aware(r)["regime"]); print("turnover cost", a["rebalance_to_max_sharpe"])
w0 = np.full(5, .2); assert np.abs(P.cost_aware_optimize(mu, c, w0, 500) - w0).sum() <= np.abs(P.cost_aware_optimize(mu, c, w0, 0) - w0).sum() + 1e-6
print("walk-forward OOS:", {k: round(v["oos_sharpe"], 2) for k, v in P.walk_forward_compare(r).items()})
m = dict(Fcf=1e9, Revenuegrowth=.08, Sharesoutstanding=1e8, Currentprice=100, Debt=2e9, Cash=1e9, Beta=1.1, Marketcap=1e10)
mc = MC.simulate(m, n=5000); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in mc.items() if k != "note"})
assert mc["price_outside_range"] is True  # price 100 vs ~550 simulated values
assert mc["p10"] < mc["median"] < mc["p90"] and abs(mc["prob_undervalued"] + mc["prob_overvalued"] - 1) < 1e-9
assert "error" in MC.simulate({**m, "Fcf": -1})
# event study: inject +5% on event days, known beta 1.2
mk = pd.Series(rng.normal(0, .01, 900), index=pd.bdate_range("2020-01-01", periods=900)); st_ = 1.2 * mk + rng.normal(0, .008, 900)
ev = []
for i, d in enumerate(mk.index[300:800:50]):
    s = (1 if i % 2 == 0 else -1) * 0.5; st_.loc[d] += 0.05 * s; ev.append({"date": d, "category": "earnings", "surprise": s})
df = ES.run_events(st_, mk, ev); sm = ES.summarize(df); print(sm); print(ES.surprise_validation(df))
assert abs(df["beta"].mean() - 1.2) < .15 and ES.surprise_validation(df)["oos_corr"] > .5
c = M.cost_shock(100, .10, .10, .15, 0.0); assert abs(c["delta_pp"] + 1.5) < 1e-9
print(M.simulate(dict(Revenue=1e11, OperatingMargin=.08, Debt=3e10), "auto", {"steel": .15, "rates_bp": 100, "fx": -.1}))
print("NEW MODULES OK")
