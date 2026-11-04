import numpy as np, pandas as pd, ml
rng = np.random.default_rng(1)
def synth_company(i, years=4):
    rev, ta = rng.uniform(500, 5000), rng.uniform(800, 6000)
    lev, mg = rng.uniform(.3, .8), rng.uniform(.0, .3)
    out = []
    for y in range(years):  # build oldest -> newest then reverse
        g = rng.normal(.08, .12); rev *= 1 + g; ta *= 1 + g * .8; mg = np.clip(mg + rng.normal(0, .04), -.2, .5)
        ebit = rev * mg; ni = ebit * .7 - ta * lev * .02
        out.append(dict(date=f"{2022 + y}-12-31", revenue=rev, gross_profit=rev * (.3 + mg), ebit=ebit, net_income=ni,
            interest_expense=ta * lev * .04, total_assets=ta, total_liabilities=ta * lev, current_assets=ta * .4,
            current_liabilities=ta * lev * .4, cash=ta * .1, ocf=ni + rng.normal(0, 30), retained_earnings=ta * .2 * (1 + mg),
            equity=ta * (1 - lev), receivables=rev * .15, inventory=rev * .1, ppe=ta * .3, depreciation=rev * .04,
            sga=rev * .15, total_debt=ta * lev * .6, capex=rev * .05, cogs=rev * .6, lt_debt=ta * lev * .5, pretax=ni, tax=0, shares=100))
    return out[::-1]
pbt = {f"T{i}": synth_company(i) for i in range(160)}
panel = ml.build_panel(pbt, drop=0.3); print("panel", panel.shape, "deteriorate rate", panel.y_deteriorate.mean().round(2))
d = ml.fit_distress(panel); print({k: v for k, v in d.items() if k != 'model'})
tgt = pbt["T0"]
print("p(deteriorate)", ml.predict_distress(d, tgt))
print(ml.anomaly_scan(panel, tgt))
f = ml.fit_forecaster(panel); print({k: v for k, v in f.items() if k not in ('model', 'cols')}); print(ml.forecast_next(f, tgt))
rows = [dict(date=i, estimate=1 + i * .02, actual=1 + i * .02 + rng.normal(.03, .05)) for i in range(24)]
print(ml.earnings_surprise_model(rows))
print(ml.sentiment(["Company beats estimates, raises guidance", "Lawsuit and probe trigger plunge"], use_finbert=False))
idx = pd.date_range("2025-01-01", periods=120); px = pd.Series(100 + rng.normal(0, 1, 120).cumsum(), index=idx)
print(ml.news_impact({str(d.date()): float(rng.normal()) for d in idx[:60]}, px))
print("OK")
