import numpy as np, ml, ml2
from test_ml import synth_company
rng = np.random.default_rng(5)
# --- change point: planted shift at index 10
x = np.r_[rng.normal(0.10, .01, 10), rng.normal(0.20, .01, 10)]
cp = ml2.detect_changepoints(x, list(range(20))); print(cp); assert cp["change_points"] and abs(cp["change_points"][0]["index"] - 10) <= 1
null = ml2.detect_changepoints(rng.normal(0, 1, 20)); print("null:", null["change_points"]); assert len(null["change_points"]) == 0
assert "need >= 8" in ml2.detect_changepoints([1, 2, 3])["note"]
# --- survival: 9-period synthetic firms with downward drift for a subset
def firm(i, yrs=9):
    ps = synth_company(i, yrs); drift = 0.12 if i % 3 == 0 else 0.0
    for k, p in enumerate(reversed(ps)):      # oldest->newest, shrink profitability for drifting firms
        f = 1 - drift * k; p["ebit"] *= f; p["retained_earnings"] *= max(f, .1); p["net_income"] *= f
    return ps
pbt = {f"S{i}": firm(i) for i in range(200)}
zs = [ml.altman_period(p) for ps in pbt.values() for p in ps]; thr = float(np.nanpercentile(zs, 20)); print("threshold", round(thr, 2))
fit = ml2.fit_survival(pbt, threshold=thr)
print({k: v for k, v in fit.items() if k not in ("model", "km", "cols")}); print("KM events", fit["km"]["events"], "median", fit["km"]["median_time"])
if "model" in fit: print(ml2.survival_predict(fit, pbt["S0"]))
assert "model" in fit and fit["n_events"] >= 8, fit
print("too-short:", ml2.fit_survival({f"T{i}": synth_company(i) for i in range(40)})["error"][:60])
# --- walk-forward + cross company on 9-period panel
panel = ml.build_panel(pbt, drop=0.3); print(panel.shape, sorted(panel.year.unique()))
wf = ml2.walk_forward_distress(panel); print(wf.get("mean_auc"), wf.get("verdict", wf.get("error")))
cc = ml2.cross_company(panel); print(cc)
assert wf["folds"] and cc["logistic"]["auc"] > 0.5
pit = ml2.point_in_time(panel, "2026-01-31", 60); assert (pit["year"] <= 2025).all() and len(pit) < len(panel)
print(ml2.structural_changes(pbt["S0"])["op_margin"])
print("ML2 OK")
