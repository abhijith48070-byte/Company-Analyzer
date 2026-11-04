"""Advanced validation and time-series tools: survival analysis, change-point detection,
walk-forward validation, cross-company generalisation, point-in-time filtering.
Everything reports negative results as plainly as positive ones."""
import numpy as np
import pandas as pd

import ml
from ml import FEATURES, altman_period, features, _models, _auc_ci


# ------------------------------------------------------------ point-in-time
def point_in_time(panel, asof, lag_days=60):
    """Rows whose financials would have been PUBLIC at `asof` (fiscal year-end + filing lag <= asof). Use for any back-test."""
    avail = pd.to_datetime(panel["date"]) + pd.Timedelta(days=lag_days)
    return panel[avail <= pd.Timestamp(asof)]


# ----------------------------------------------------------------- survival
def build_survival(periods_by_ticker, threshold=4.15):
    """One row per at-risk firm-year (Z'' >= threshold at the start). event=1 if next-year Z'' < threshold. Features use data up to the start year only.
    Firms already in the distress zone at their first observation are excluded (left-censored)."""
    rows = []
    for tk, ps in periods_by_ticker.items():
        ps = ps[::-1]                       # oldest -> newest
        zs = [altman_period(p) for p in ps]
        for i in range(1, len(ps) - 1):
            if np.isnan(zs[i]) or np.isnan(zs[i + 1]) or zs[i] < threshold:
                continue
            if any(z < threshold for z in zs[:i] if not np.isnan(z)):
                continue                    # already distressed earlier: not at risk of FIRST entry
            rows.append({"ticker": tk, "age": i, "event": int(zs[i + 1] < threshold), "z": zs[i], **features(ps[i], ps[i - 1])})
    return pd.DataFrame(rows)


def kaplan_meier(periods_by_ticker, threshold=4.15):
    """Time (years from first observation) to first entry into the distress zone; censored at last observation."""
    obs = []
    for ps in periods_by_ticker.values():
        zs = [altman_period(p) for p in ps[::-1]]
        zs = [z for z in zs if not np.isnan(z)]
        if len(zs) < 2 or zs[0] < threshold:
            continue
        t = next((i for i, z in enumerate(zs) if z < threshold), None)
        obs.append((t, 1) if t is not None else (len(zs) - 1, 0))
    if not obs:
        return {"error": "no at-risk companies"}
    s, curve = 1.0, []
    for t in sorted({o[0] for o in obs if o[1]}):
        n = sum(1 for o in obs if o[0] >= t); d = sum(1 for o in obs if o[0] == t and o[1])
        s *= 1 - d / n; curve.append({"year": t, "at_risk": n, "events": d, "survival": s})
    return {"n": len(obs), "events": sum(o[1] for o in obs), "curve": curve,
            "median_time": next((c["year"] for c in curve if c["survival"] <= 0.5), None)}


def fit_survival(periods_by_ticker, threshold=4.15):
    """Discrete-time hazard model (logistic on person-period rows). Validation: grouped CV AUC by company.
    Multi-year probability assumes the one-year hazard persists: P(h years) = 1-(1-p)^h (stated assumption)."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    df = build_survival(periods_by_ticker, threshold)
    if len(df) < 60 or df["event"].sum() < 8:
        return {"error": f"too few at-risk rows/events (rows {len(df)}, events {int(df['event'].sum()) if len(df) else 0}); needs multi-year statements (see EDGAR XBRL history)",
                "km": kaplan_meier(periods_by_ticker, threshold)}
    cols = FEATURES + ["z"]
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(5).split(df, groups=df["ticker"]):
        oof[te] = _models()["logistic"].fit(df.iloc[tr][cols], df.iloc[tr]["event"]).predict_proba(df.iloc[te][cols])[:, 1]
    auc = float(roc_auc_score(df["event"], oof)) if df["event"].nunique() == 2 else None
    mdl = _models()["logistic"].fit(df[cols], df["event"])
    return {"model": mdl, "cols": cols, "n_rows": len(df), "n_events": int(df["event"].sum()), "grouped_cv_auc": auc,
            "auc_ci90": _auc_ci(df["event"], oof), "km": kaplan_meier(periods_by_ticker, threshold), "threshold": threshold}


def survival_predict(fit, periods, horizons=(1, 2, 3)):
    if "model" not in fit or len(periods) < 2 or altman_period(periods[0]) < fit["threshold"]:
        return None
    x = pd.DataFrame([{**features(periods[0], periods[1]), "z": altman_period(periods[0])}])[fit["cols"]]
    p = float(fit["model"].predict_proba(x)[0, 1])
    return {"p_1y": p, **{f"p_{h}y": 1 - (1 - p) ** h for h in horizons if h > 1}, "assumption": "constant one-year hazard"}


# ------------------------------------------------------------ change points
def detect_changepoints(values, labels=None, max_cp=3, min_seg=2, n_perm=500, seed=0, alpha=0.05):
    """Binary segmentation for mean shifts, each split accepted only if a permutation test is significant at `alpha`
    (shuffling the segment). Short series have low power: with n < 8 the function says so instead of guessing."""
    x = np.asarray(values, float); keep = ~np.isnan(x); x = x[keep]
    labels = [l for l, k in zip(labels or range(len(keep)), keep) if k]
    if len(x) < 8:
        return {"change_points": [], "note": f"only {len(x)} observations; need >= 8 for a reliable test"}
    rng = np.random.default_rng(seed)

    def best_split(seg):
        sse = lambda a: ((a - a.mean()) ** 2).sum()
        base = sse(seg)
        gains = [(base - sse(seg[:k]) - sse(seg[k:]), k) for k in range(min_seg, len(seg) - min_seg + 1)]
        return max(gains) if gains else (0, None)

    found, segs = [], [(0, len(x))]
    while segs and len(found) < max_cp:
        cand = []
        for lo, hi in segs:
            if hi - lo >= 2 * min_seg:
                g, k = best_split(x[lo:hi])
                if k is not None:
                    cand.append((g, lo, hi, lo + k))
        if not cand:
            break
        g, lo, hi, k = max(cand)
        null = [best_split(rng.permutation(x[lo:hi]))[0] for _ in range(n_perm)]
        p = (1 + sum(n >= g for n in null)) / (n_perm + 1)
        segs.remove((lo, hi))
        if p > alpha:
            continue
        before, after = x[lo:k].mean(), x[k:hi].mean()
        found.append({"index": int(k), "at": labels[k], "mean_before": float(before), "mean_after": float(after), "p_value": float(p)})
        segs += [(lo, k), (k, hi)]
    return {"change_points": sorted(found, key=lambda d: d["index"]), "n": len(x)}


def structural_changes(periods):
    """Change-point scan of margins, growth and Z'' for one company (periods newest-first)."""
    ps = periods[::-1]; labels = [p["date"] for p in ps]
    f = [features(ps[i], ps[i - 1]) for i in range(1, len(ps))]; lab = labels[1:]
    out = {"op_margin": detect_changepoints([d["op_margin"] for d in f], lab),
           "gross_margin": detect_changepoints([d["gross_margin"] for d in f], lab),
           "rev_growth": detect_changepoints([d["rev_growth"] for d in f], lab),
           "altman_z": detect_changepoints([altman_period(p) for p in ps], labels)}
    return out


# ---------------------------------------------------- walk-forward / grouped
def walk_forward_distress(panel, min_train_years=2, models=("logistic", "hist_gradient_boosting")):
    """Expanding-window: for each year Y, train on years < Y, test on Y. Compares models against a Z''-only baseline."""
    from sklearn.metrics import roc_auc_score
    years = sorted(panel["year"].unique()); folds = []
    avail = _models()
    for i in range(min_train_years, len(years)):
        tr, te = panel[panel.year < years[i]], panel[panel.year == years[i]]
        if te["y_deteriorate"].nunique() < 2 or tr["y_deteriorate"].nunique() < 2:
            continue
        row = {"test_year": int(years[i]), "n_test": len(te)}
        for name in models:
            if name in avail:
                row[name] = float(roc_auc_score(te["y_deteriorate"], avail[name].fit(tr[FEATURES], tr["y_deteriorate"]).predict_proba(te[FEATURES])[:, 1]))
        base = avail["logistic"].fit(tr[["z"]], tr["y_deteriorate"])
        row["z_only_baseline"] = float(roc_auc_score(te["y_deteriorate"], base.predict_proba(te[["z"]])[:, 1]))
        folds.append(row)
    if not folds:
        return {"error": f"need >= {min_train_years + 1} fiscal years with both classes in train and test; have {len(years)}"}
    keys = [k for k in folds[0] if k not in ("test_year", "n_test")]
    mean = {k: float(np.mean([f[k] for f in folds])) for k in keys}
    return {"folds": folds, "mean_auc": mean,
            "verdict": ("features beat the Z''-only baseline" if mean.get("logistic", 0) > mean["z_only_baseline"] + 0.02
                        else "NEGATIVE RESULT: features do not beat a Z''-only baseline out-of-sample")}


def cross_company(panel, group="ticker", n_splits=5, models=("logistic", "hist_gradient_boosting")):
    """Generalisation to companies (or sectors, group='sector') never seen in training: grouped K-fold out-of-fold AUC with bootstrap CI.
    NOTE: ignores time order, so it measures cross-sectional generalisation, not forecasting."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    if group not in panel or panel[group].nunique() < n_splits or panel["y_deteriorate"].nunique() < 2:
        return {"error": f"need >= {n_splits} distinct {group} values and both classes"}
    out, avail = {}, _models()
    for name in models:
        if name not in avail:
            continue
        oof = np.zeros(len(panel))
        for tr, te in GroupKFold(n_splits).split(panel, groups=panel[group]):
            oof[te] = avail[name].fit(panel.iloc[tr][FEATURES], panel.iloc[tr]["y_deteriorate"]).predict_proba(panel.iloc[te][FEATURES])[:, 1]
        out[name] = {"auc": float(roc_auc_score(panel["y_deteriorate"], oof)), "auc_ci90": _auc_ci(panel["y_deteriorate"], oof), "grouped_by": group}
    return out
