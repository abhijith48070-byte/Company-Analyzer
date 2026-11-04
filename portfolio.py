"""Portfolio intelligence: optimisation, risk, Black-Litterman, Monte Carlo, regime-aware and cost-aware rebalancing.
Inputs are a DataFrame of periodic simple returns (columns = tickers). Long-only, fully invested unless stated.
Estimates from history are noisy: covariance is Ledoit-Wolf shrunk, and results are descriptive, not advice."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PPY = 252


def _cov(r):
    try:
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(r.values).covariance_ * PPY
    except Exception:
        return r.cov().values * PPY


def stats(r):
    mu = r.mean().values * PPY
    return mu, _cov(r)


def _solve(obj, n, cons=(), bounds=None, w0=None):
    bounds = bounds or [(0, 1)] * n
    c = [{"type": "eq", "fun": lambda w: w.sum() - 1}, *cons]
    res = minimize(obj, w0 if w0 is not None else np.full(n, 1 / n), method="SLSQP", bounds=bounds, constraints=c, options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def perf(w, mu, cov, rf=0.0):
    ret, vol = float(w @ mu), float(np.sqrt(w @ cov @ w))
    return {"return": ret, "vol": vol, "sharpe": (ret - rf) / vol if vol else None}


def min_vol(mu, cov, max_w=1.0):
    n = len(mu); return _solve(lambda w: w @ cov @ w, n, bounds=[(0, max_w)] * n)


def max_sharpe(mu, cov, rf=0.0, max_w=1.0):
    n = len(mu); return _solve(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w + 1e-12), n, bounds=[(0, max_w)] * n)


def mean_variance(mu, cov, risk_aversion=3.0, max_w=1.0):
    n = len(mu); return _solve(lambda w: 0.5 * risk_aversion * w @ cov @ w - w @ mu, n, bounds=[(0, max_w)] * n)


def risk_parity(cov):
    n = len(cov)
    def obj(w):
        rc = w * (cov @ w); return float(((rc - rc.mean()) ** 2).sum())
    return _solve(obj, n, bounds=[(1e-4, 1)] * n)


def efficient_frontier(mu, cov, points=15, max_w=1.0):
    n = len(mu); lo, hi = perf(min_vol(mu, cov, max_w), mu, cov)["return"], float(mu.max())
    out = []
    for t in np.linspace(lo, hi * 0.999, points):
        w = _solve(lambda w: w @ cov @ w, n, cons=[{"type": "eq", "fun": lambda w, t=t: w @ mu - t}], bounds=[(0, max_w)] * n)
        out.append({**perf(w, mu, cov), "weights": w.round(4).tolist()})
    return out


def risk_report(port_returns, level=0.95):
    x = np.asarray(port_returns); q = np.quantile(x, 1 - level)
    cum = np.cumprod(1 + x); dd = cum / np.maximum.accumulate(cum) - 1
    return {"var_hist": float(-q), "cvar_hist": float(-x[x <= q].mean()),
            "var_parametric": float(-(x.mean() + np.sqrt(x.var()) * __import__("scipy.stats").stats.norm.ppf(1 - level))),
            "max_drawdown": float(dd.min()), "ann_vol": float(x.std() * np.sqrt(PPY)), "level": level, "horizon": "1 period"}


def factor_exposure(port_returns, factors):
    """OLS betas of portfolio returns on factor returns (DataFrame). Returns betas, alpha (annualised), R^2."""
    y = np.asarray(port_returns); F = np.asarray(factors)
    X = np.column_stack([np.ones(len(F)), F]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; r2 = 1 - res.var() / y.var()
    return {"alpha_annual": float(b[0] * PPY), "betas": dict(zip(factors.columns, b[1:].round(4))), "r2": float(r2)}


def black_litterman(cov, mkt_w, views_P, views_Q, tau=0.05, risk_aversion=2.5, omega=None):
    """Posterior expected returns. views_P: k x n, views_Q: k (absolute annual returns)."""
    pi = risk_aversion * cov @ mkt_w
    P, Q = np.atleast_2d(views_P), np.asarray(views_Q, float)
    om = np.diag(np.diag(P @ (tau * cov) @ P.T)) if omega is None else omega
    A = np.linalg.inv(np.linalg.inv(tau * cov) + P.T @ np.linalg.inv(om) @ P)
    return A @ (np.linalg.inv(tau * cov) @ pi + P.T @ np.linalg.inv(om) @ Q), pi


def monte_carlo(w, mu, cov, years=5, sims=5000, seed=0, contrib=0.0):
    """Multivariate-normal annual returns; terminal wealth distribution (start = 1)."""
    rng = np.random.default_rng(seed)
    r = rng.multivariate_normal(mu, cov, size=(sims, years)) @ w
    wealth = np.ones(sims)
    for t in range(years):
        wealth = wealth * (1 + r[:, t]) + contrib
    q = np.percentile(wealth, [5, 25, 50, 75, 95])
    return {"p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4], "prob_loss": float((wealth < 1 + contrib * years).mean()), "years": years}


def regimes(mkt_returns, window=60):
    """Rolling-volatility regime label: 'high' when trailing vol exceeds its historical median."""
    vol = pd.Series(mkt_returns).rolling(window).std()
    return (vol > vol.median()).map({True: "high_vol", False: "low_vol"}), vol


def regime_aware(r, mkt_col=None, window=60, max_w=1.0):
    """Estimate covariance only from the current regime's history and optimise min-vol; falls back to full sample if too few rows."""
    m = r[mkt_col] if mkt_col else r.mean(axis=1)
    lab, _ = regimes(m, window)
    cur = lab.iloc[-1]; sub = r[(lab == cur).values]
    if len(sub) < 60: sub, cur = r, "full_sample(fallback)"
    mu, cov = stats(sub)
    return {"regime": cur, "n_obs": len(sub), "weights": dict(zip(r.columns, min_vol(mu, cov, max_w).round(4)))}


def rebalance(current_w, target_w, tickers, cost_bps=10.0, threshold=0.02, value=1.0):
    """Trade list; trades smaller than `threshold` of the portfolio are skipped (a simple turnover/cost control)."""
    cw, tw = np.asarray(current_w, float), np.asarray(target_w, float)
    d = tw - cw; d[np.abs(d) < threshold] = 0.0
    turnover = float(np.abs(d).sum() / 2)
    return {"trades": {t: round(float(x), 4) for t, x in zip(tickers, d) if x}, "turnover": turnover,
            "est_cost": float(np.abs(d).sum() * cost_bps / 1e4 * value)}


def cost_aware_optimize(mu, cov, w_prev, cost_bps=10.0, risk_aversion=3.0, max_w=1.0):
    """Mean-variance objective with a linear turnover penalty (cost_bps per unit traded, annualised-return units)."""
    n, c = len(mu), cost_bps / 1e4
    return _solve(lambda w: 0.5 * risk_aversion * w @ cov @ w - w @ mu + c * np.abs(w - w_prev).sum(), n, bounds=[(0, max_w)] * n, w0=np.asarray(w_prev))


def analyze_portfolio(r, rf=0.0, current_w=None, factors=None, max_w=0.5):
    tick = list(r.columns); mu, cov = stats(r)
    n = len(tick); eq = np.full(n, 1 / n)
    cands = {"equal_weight": eq, "min_vol": min_vol(mu, cov, max_w), "max_sharpe": max_sharpe(mu, cov, rf, max_w),
             "mean_variance": mean_variance(mu, cov, 3, max_w), "risk_parity": risk_parity(cov)}
    out = {"tickers": tick, "n_obs": len(r), "correlation": r.corr().round(3).to_dict(), "portfolios": {}}
    for k, w in cands.items():
        pr = r.values @ w
        out["portfolios"][k] = {"weights": dict(zip(tick, w.round(4))), **perf(w, mu, cov, rf), **risk_report(pr), "mc": monte_carlo(w, mu, cov)}
    out["frontier"] = efficient_frontier(mu, cov, 10, max_w)
    if factors is not None:
        out["factor_exposure_max_sharpe"] = factor_exposure(r.values @ cands["max_sharpe"], factors.loc[r.index])
    if current_w is not None:
        out["rebalance_to_max_sharpe"] = rebalance(current_w, cands["max_sharpe"], tick)
    out["caveat"] = ("In-sample estimates with Ledoit-Wolf shrinkage; max-Sharpe weights are notoriously unstable. "
                     "Compare to equal_weight out-of-sample before trusting any optimiser (see walk_forward_compare).")
    return out


def walk_forward_compare(r, train=252, step=63, max_w=0.5):
    """Out-of-sample check: re-optimise every `step` periods on the trailing `train` window, hold, compare realised Sharpe vs equal weight."""
    res = {k: [] for k in ("equal_weight", "min_vol", "max_sharpe", "risk_parity")}
    for s in range(train, len(r) - step + 1, step):
        mu, cov = stats(r.iloc[s - train:s]); fut = r.iloc[s:s + step].values
        ws = {"equal_weight": np.full(len(mu), 1 / len(mu)), "min_vol": min_vol(mu, cov, max_w), "max_sharpe": max_sharpe(mu, cov, 0, max_w), "risk_parity": risk_parity(cov)}
        for k, w in ws.items(): res[k].extend((fut @ w).tolist())
    return {k: {"oos_sharpe": float(np.mean(v) / np.std(v) * np.sqrt(PPY)) if np.std(v) else None, "oos_ann_vol": float(np.std(v) * np.sqrt(PPY)), "n": len(v)} for k, v in res.items() if v}
