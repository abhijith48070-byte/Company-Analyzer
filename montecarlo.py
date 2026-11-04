"""Monte Carlo DCF with correlated drivers and variance decomposition.
Drivers: starting FCF growth, FCF margin shock (scales FCF level), WACC, terminal growth, capex intensity shock.
Correlations (documented assumptions, editable): growth~margin +0.4, growth~wacc -0.2, wacc~terminal_g +0.3, capex~margin -0.4."""
import numpy as np

from valuation import compute_wacc

NAMES = ["growth", "margin_shock", "wacc", "terminal_g", "capex_shock"]
CORR = np.array([[1, .4, -.2, 0, 0], [.4, 1, 0, 0, -.4], [-.2, 0, 1, .3, 0], [0, 0, .3, 1, 0], [0, -.4, 0, 0, 1]], float)


def _value(fcf0, g, wacc, tg, margin_shock, capex_shock, net_debt, shares, years=10):
    tg = np.minimum(tg, wacc - 0.02)
    fcf = fcf0 * (1 + margin_shock) * (1 - 0.5 * capex_shock)
    pv, f = 0.0, fcf
    for t in range(1, years + 1):
        gt = g + (tg - g) * (t - 1) / years
        f = f * (1 + gt); pv = pv + f / (1 + wacc) ** t
    tv = f * (1 + tg) / (wacc - tg)
    return (pv + tv / (1 + wacc) ** years - net_debt) / shares


def simulate(metrics, n=20000, seed=0, sd=None, corr=None):
    """metrics: dict from get_financialmetrics. Returns percentiles, probabilities vs current price, and sensitivity (rank-corr^2 share)."""
    if not metrics.get("Fcf") or metrics["Fcf"] <= 0:
        return {"error": "FCF is non-positive; Monte Carlo DCF not meaningful"}
    shares, price = metrics.get("Sharesoutstanding"), metrics.get("Currentprice")
    if not shares or not price:
        return {"error": "shares or price unavailable"}
    w = compute_wacc(metrics)
    g0 = float(np.clip(metrics.get("Revenuegrowth") or 0.05, -0.05, 0.30))
    mean = np.array([g0, 0.0, w["wacc"], 0.025, 0.0])
    s = np.array([0.04, 0.10, 0.008, 0.004, 0.10]) if sd is None else np.asarray(sd)
    C = CORR if corr is None else np.asarray(corr)
    cov = np.outer(s, s) * C
    X = np.random.default_rng(seed).multivariate_normal(mean, cov, size=n)
    net_debt = (metrics.get("Debt") or 0) - (metrics.get("Cash") or 0)
    X[:, 2] = np.maximum(X[:, 2], 0.045)
    v = _value(metrics["Fcf"], X[:, 0], X[:, 2], X[:, 3], X[:, 1], X[:, 4], net_debt, shares)
    ok = np.isfinite(v)
    v, X = v[ok], X[ok]
    from scipy.stats import spearmanr
    r2 = np.array([spearmanr(X[:, i], v)[0] ** 2 for i in range(5)])
    share = r2 / r2.sum() if r2.sum() else r2
    q = np.percentile(v, [10, 50, 90]); q01, q99 = np.percentile(v, [1, 99])
    return {"price": price, "p10": float(q[0]), "median": float(q[1]), "p90": float(q[2]), "mean": float(v.mean()),
            "prob_undervalued": float((v > price).mean()), "prob_overvalued": float((v <= price).mean()),
            "sensitivity_share": {k: round(float(x), 3) for k, x in sorted(zip(NAMES, share), key=lambda kv: -kv[1])},
            "p01": float(q01), "p99": float(q99), "price_outside_range": bool(price < q01 or price > q99), "n": int(ok.sum()), "wacc_source": w.get("rf_source"),
            "note": "Distribution reflects the assumed driver volatilities/correlations, not a forecast; widen sd for more uncertainty."}
