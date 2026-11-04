"""Event study: market-model abnormal returns, CAR, volatility change, post-event drift, category summary, chronological validation."""
import numpy as np
import pandas as pd
from scipy import stats as st


def market_model(stock, mkt, est_window=(-250, -30), event_idx=None):
    """OLS alpha/beta over the estimation window ending before the event. stock, mkt: aligned return Series."""
    lo, hi = event_idx + est_window[0], event_idx + est_window[1]
    if lo < 0 or hi <= lo + 30:
        return None
    y, x = stock.iloc[lo:hi].values, mkt.iloc[lo:hi].values
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    return {"alpha": a, "beta": b, "sigma": resid.std(ddof=2), "n": len(y)}


def event_window(stock, mkt, date, pre=1, post=1, drift=(2, 30), est_window=(-250, -30)):
    """Single event: AR, CAR[-pre,+post], t-stat, vol ratio (post/pre), post-event drift CAR over `drift` days after the window."""
    d = pd.Timestamp(date)
    pos = stock.index.searchsorted(d)
    if pos >= len(stock):
        return None
    mm = market_model(stock, mkt, est_window, pos)
    if not mm or pos + drift[1] >= len(stock):
        return None
    ar = stock - (mm["alpha"] + mm["beta"] * mkt)
    win = ar.iloc[pos - pre:pos + post + 1]
    car = float(win.sum())
    t = car / (mm["sigma"] * np.sqrt(len(win)))
    dr = float(ar.iloc[pos + drift[0]:pos + drift[1] + 1].sum())
    vb, va = stock.iloc[pos - 20:pos - 1].std(), stock.iloc[pos + 2:pos + 22].std()
    return {"date": str(stock.index[pos].date()), "car": car, "t_stat": float(t), "ar_day0": float(ar.iloc[pos]), "drift_car": dr,
            "vol_ratio": float(va / vb) if vb else None, "beta": mm["beta"]}


def run_events(stock, mkt, events, **kw):
    """events: list of {'date':..., 'category':... , 'surprise': optional float}. Returns per-event rows (skipped ones omitted)."""
    rows = []
    for e in events:
        r = event_window(stock, mkt, e["date"], **kw)
        if r:
            rows.append({**r, "category": e.get("category", "event"), "surprise": e.get("surprise")})
    return pd.DataFrame(rows)


def summarize(df):
    """By category: n, mean CAR, t-test of mean CAR != 0, hit rate, mean drift, mean vol ratio."""
    if df.empty:
        return {}
    out = {}
    for c, g in df.groupby("category"):
        t, p = st.ttest_1samp(g["car"], 0) if len(g) > 2 else (np.nan, np.nan)
        out[c] = {"n": len(g), "mean_car": float(g["car"].mean()), "t": float(t), "p": float(p), "pct_positive": float((g["car"] > 0).mean()),
                  "mean_drift": float(g["drift_car"].mean()), "mean_vol_ratio": float(g["vol_ratio"].mean())}
    return out


def surprise_validation(df):
    """Chronological check that surprise predicts CAR: fit slope on earliest 60%, test correlation on latest 40%. Reports negative results as-is."""
    d = df.dropna(subset=["surprise"]).sort_values("date")
    if len(d) < 10:
        return {"error": f"need >= 10 events with surprise, have {len(d)}"}
    k = int(len(d) * 0.6); tr, te = d.iloc[:k], d.iloc[k:]
    slope, icpt = np.polyfit(tr["surprise"], tr["car"], 1)
    pred = icpt + slope * te["surprise"]
    r = np.corrcoef(pred, te["car"])[0, 1] if te["car"].std() and pred.std() else np.nan
    return {"slope_train": float(slope), "oos_corr": float(r), "oos_n": len(te),
            "sign_hit_rate": float((np.sign(te["car"]) == np.sign(pred)).mean()),
            "verdict": "surprise has out-of-sample predictive value" if r == r and r > 0.3 else "no reliable out-of-sample relationship"}
