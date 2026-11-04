"""ML intelligence. Every model: point-in-time features at t, targets from t+1, chronological split,
compared with a naive baseline. Small samples are reported as such (n is always returned)."""
import numpy as np
import pandas as pd

FEATURES = ["rev_growth", "gross_margin", "op_margin", "net_margin", "roa", "leverage", "cash_ratio",
            "interest_cov", "current_ratio", "accruals", "asset_turnover"]


def _d(a, b):
    return a / b if b else 0.0


def features(cur, prev):
    ta = cur["total_assets"] or 1.0
    cov = _d(cur["ebit"], cur["interest_expense"]) if cur["interest_expense"] else 20.0
    return {
        "rev_growth": _d(cur["revenue"] - prev["revenue"], abs(prev["revenue"])),
        "gross_margin": _d(cur["gross_profit"], cur["revenue"]),
        "op_margin": _d(cur["ebit"], cur["revenue"]),
        "net_margin": _d(cur["net_income"], cur["revenue"]),
        "roa": cur["net_income"] / ta,
        "leverage": cur["total_liabilities"] / ta,
        "cash_ratio": cur["cash"] / ta,
        "interest_cov": float(np.clip(cov, -20, 20)),
        "current_ratio": float(min(_d(cur["current_assets"], cur["current_liabilities"]), 10)),
        "accruals": (cur["net_income"] - cur["ocf"]) / ta,
        "asset_turnover": cur["revenue"] / ta,
    }


def altman_period(p):
    ta, tl = p["total_assets"], p["total_liabilities"]
    if not ta or not tl:
        return float("nan")
    return (3.25 + 6.56 * (p["current_assets"] - p["current_liabilities"]) / ta + 3.26 * p["retained_earnings"] / ta
            + 6.72 * p["ebit"] / ta + 1.05 * p["equity"] / tl)


def build_panel(periods_by_ticker, drop=1.0):
    """Rows: features at period i (using i+1 as prior); targets from period i-1 (the NEXT year)."""
    rows = []
    for tk, ps in periods_by_ticker.items():
        for i in range(1, len(ps) - 1):
            cur, prev, nxt = ps[i], ps[i + 1], ps[i - 1]
            f, fn = features(cur, prev), features(nxt, cur)
            z, zn = altman_period(cur), altman_period(nxt)
            if np.isnan(z) or np.isnan(zn):
                continue
            rows.append({"ticker": tk, "date": cur["date"], **f, "z": z,
                         "y_deteriorate": int(zn < z - drop), "y_next_growth": fn["rev_growth"],
                         "y_next_opm": fn["op_margin"]})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["year"] = df["date"].str[:4].astype(int)
    return df


def chrono_split(df):
    """Train on earlier fiscal years, test on the latest year(s). Never random."""
    years = sorted(df["year"].unique())
    if len(years) < 2:
        return df.iloc[:0], df
    cut = years[-1]
    tr, te = df[df["year"] < cut], df[df["year"] >= cut]
    return tr, te


def _models():
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5)),
         "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0),
         "hist_gradient_boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, random_state=0)}
    try:
        import lightgbm as lgb
        m["lightgbm"] = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, verbose=-1)
    except Exception:
        pass
    return m


def fit_distress(panel):
    """Predict material Altman Z'' deterioration next year. Compared against the base-rate baseline."""
    from sklearn.metrics import brier_score_loss, roc_auc_score
    tr, te = chrono_split(panel)
    if len(tr) < 30 or len(te) < 10 or tr["y_deteriorate"].nunique() < 2 or tr["y_deteriorate"].sum() < 8:
        return {"error": f"not enough data or too few deterioration cases (train {len(tr)}, test {len(te)}, "
                         f"positives in train {int(tr['y_deteriorate'].sum()) if len(tr) else 0}); "
                         f"try build_panel(..., drop=0.5) or a larger peer set"}
    X, y, Xt, yt = tr[FEATURES], tr["y_deteriorate"], te[FEATURES], te["y_deteriorate"]
    base = y.mean()
    res = {"n_train": len(tr), "n_test": len(te), "test_year": int(te["year"].max()), "base_rate_train": float(base),
           "baseline_brier": float(brier_score_loss(yt, np.full(len(yt), base))), "models": {}}
    best, best_auc = None, -1
    for name, mdl in _models().items():
        mdl.fit(X, y)
        p = mdl.predict_proba(Xt)[:, 1]
        auc = float(roc_auc_score(yt, p)) if yt.nunique() == 2 else None
        res["models"][name] = {"auc": auc, "brier": float(brier_score_loss(yt, p))}
        if auc is not None and auc > best_auc:
            best, best_auc = name, auc
    res["best"] = best
    if best:
        final = _models()[best].fit(pd.concat([X, Xt]), pd.concat([y, yt]))  # refit on all history for live use
        res["model"] = final
    rf = _models()["random_forest"].fit(X, y)
    res["feature_importance"] = dict(sorted(zip(FEATURES, map(float, rf.feature_importances_)), key=lambda kv: -kv[1])[:5])
    return res


def predict_distress(fit, periods):
    if "model" not in fit or len(periods) < 2:
        return None
    x = pd.DataFrame([features(periods[0], periods[1])])[FEATURES]
    return float(fit["model"].predict_proba(x)[:, 1][0])


def anomaly_scan(panel, target_periods):
    """Isolation Forest over the peer cross-section + robust z-scores to explain which ratios stand out."""
    from sklearn.ensemble import IsolationForest
    if len(panel) < 30 or len(target_periods) < 2:
        return {"error": "not enough peer data"}
    iso = IsolationForest(n_estimators=300, contamination="auto", random_state=0).fit(panel[FEATURES])
    t = pd.DataFrame([features(target_periods[0], target_periods[1])])[FEATURES]
    s = -iso.score_samples(t)[0]
    peers = -iso.score_samples(panel[FEATURES])
    med, mad = panel[FEATURES].median(), (panel[FEATURES] - panel[FEATURES].median()).abs().median().replace(0, 1e-9)
    z = ((t.iloc[0] - med) / (1.4826 * mad)).sort_values(key=abs, ascending=False)
    return {"anomaly_percentile": float((peers < s).mean() * 100), "n_peers": len(panel),
            "most_unusual_ratios": {k: {"value": float(t.iloc[0][k]), "peer_median": float(med[k]), "robust_z": float(v)}
                                    for k, v in z.head(3).items()}}


def fit_forecaster(panel, target="y_next_growth"):
    """Bayesian ridge forecast of next-year revenue growth (or op margin) with predictive uncertainty.
    Baseline = persistence (next year equals this year)."""
    from sklearn.linear_model import BayesianRidge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr, te = chrono_split(panel)
    if len(tr) < 30 or len(te) < 10:
        return {"error": f"not enough data (train {len(tr)}, test {len(te)})"}
    cols = ["rev_growth", "gross_margin", "op_margin", "roa", "leverage", "asset_turnover"]
    clip = lambda s: s.clip(s.quantile(0.02), s.quantile(0.98))   # winsorise targets
    y = clip(tr[target])
    mdl = make_pipeline(StandardScaler(), BayesianRidge()).fit(tr[cols], y)
    pred, std = mdl.predict(te[cols], return_std=True)
    persist = te["rev_growth" if target == "y_next_growth" else "op_margin"]
    mae = float(np.mean(np.abs(pred - te[target])))
    mae_b = float(np.mean(np.abs(persist - te[target])))
    cover = float(np.mean(np.abs(te[target] - pred) <= 1.645 * std))
    mdl_all = make_pipeline(StandardScaler(), BayesianRidge()).fit(pd.concat([tr, te])[cols], clip(pd.concat([tr, te])[target]))
    return {"target": target, "n_train": len(tr), "n_test": len(te), "mae_model": mae, "mae_persistence": mae_b,
            "beats_baseline": mae < mae_b, "interval90_coverage": cover, "model": mdl_all, "cols": cols}


def forecast_next(fit, periods):
    if "model" not in fit or len(periods) < 2:
        return None
    x = pd.DataFrame([features(periods[0], periods[1])])[fit["cols"]]
    m, s = fit["model"].predict(x, return_std=True)
    return {"mean": float(m[0]), "low90": float(m[0] - 1.645 * s[0]), "high90": float(m[0] + 1.645 * s[0])}


# ----------------------------------------------------------------- earnings surprise
def earnings_surprise_model(rows):
    """rows: [{'date','estimate','actual'}] ascending. Predict beat/miss from prior surprise history."""
    from sklearn.linear_model import LogisticRegression
    rows = [r for r in rows if r.get("estimate") is not None and r.get("actual") is not None]
    if len(rows) < 14:
        return {"error": f"need >= 14 quarters of estimate/actual history, have {len(rows)}"}
    s = np.array([(r["actual"] - r["estimate"]) / max(abs(r["estimate"]), 0.01) for r in rows])
    beat = (s > 0).astype(int)
    X, y = [], []
    for i in range(4, len(s)):
        X.append([s[i - 1], s[i - 4:i].mean(), beat[i - 4:i].mean()]); y.append(beat[i])
    X, y = np.array(X), np.array(y)
    k = int(len(y) * 0.7)
    if len(set(y[:k])) < 2 or k < 6:
        return {"error": "history has only one outcome class"}
    mdl = LogisticRegression(max_iter=500).fit(X[:k], y[:k])
    acc = float(mdl.score(X[k:], y[k:]))
    base_acc = float((y[k:] == int(y[:k].mean() >= 0.5)).mean())
    nxt = np.array([[s[-1], s[-4:].mean(), beat[-4:].mean()]])
    return {"n_quarters": len(rows), "n_test": len(y) - k, "accuracy": acc, "baseline_accuracy": base_acc,
            "beats_baseline": acc > base_acc, "p_beat_next": float(mdl.predict_proba(nxt)[0, 1]),
            "historical_beat_rate": float(beat.mean())}


# ------------------------------------------------------------------------ sentiment
_POS = set("""beat beats surge surges soar soars record growth strong gain gains rally upgrade upgrades outperform profit
profits raised raises raise expands expansion approval approved win wins breakthrough boost robust exceeds exceeded""".split())
_NEG = set("""miss misses plunge plunges fall falls drop drops lawsuit probe investigation downgrade downgrades loss losses
cut cuts weak decline declines recall fraud warning warns delay delays ban fine fined layoffs shortfall slump""".split())


def sentiment(texts, use_finbert=True):
    """FinBERT if `transformers` + torch are installed, else a transparent word-list fallback.
    Returns [{'label','score','method'}] with score in [-1, 1]."""
    if use_finbert:
        try:
            from transformers import pipeline
            pipe = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)
            out = []
            for t, r in zip(texts, pipe(list(texts))):
                d = {x["label"].lower(): x["score"] for x in r}
                sc = d.get("positive", 0) - d.get("negative", 0)
                out.append({"label": max(d, key=d.get), "score": float(sc), "method": "finbert"})
            return out
        except Exception:
            pass
    out = []
    for t in texts:
        w = [x.strip(".,;:!?()\"'").lower() for x in str(t).split()]
        p, n = sum(x in _POS for x in w), sum(x in _NEG for x in w)
        sc = (p - n) / max(p + n, 1)
        out.append({"label": "positive" if sc > 0 else "negative" if sc < 0 else "neutral", "score": sc,
                    "method": "wordlist"})
    return out


def news_impact(sentiment_by_date, prices):
    """Does same-day news sentiment relate to NEXT-day return? Spearman rank correlation, with n and p-value.
    sentiment_by_date: {'YYYY-MM-DD': score}; prices: pd.Series indexed by date (close)."""
    from scipy.stats import spearmanr
    r = prices.pct_change().shift(-1)
    r.index = [str(i)[:10] for i in r.index]
    pairs = [(s, r[d]) for d, s in sentiment_by_date.items() if d in r.index and r[d] == r[d]]
    if len(pairs) < 20:
        return {"error": f"only {len(pairs)} matched days; need >= 20", "n": len(pairs)}
    rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    return {"n": len(pairs), "spearman_rho": float(rho), "p_value": float(p),
            "interpretation": "association only; not evidence that news causes returns"}


def panel_cached(sector_tickers, cache_path, fetch_periods, max_age_days=30):
    """Fetch (or load) normalised statements for peers. fetch_periods(ticker) -> list of period dicts."""
    import os, pickle, time
    if os.path.exists(cache_path) and time.time() - os.path.getmtime(cache_path) < max_age_days * 86400:
        return pickle.load(open(cache_path, "rb"))
    out = {}
    for t in sector_tickers:
        try:
            ps = fetch_periods(t)
            if len(ps) >= 3:
                out[t] = ps
        except Exception:
            continue
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    pickle.dump(out, open(cache_path, "wb"))
    return out
