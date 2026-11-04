import numpy as np
from .verifiers import LABELS


def accuracy(y, p):
    return float(np.mean(np.array(y) == np.array(p)))


def macro_f1(y, p):
    y, p, f = np.array(y), np.array(p), []
    for l in LABELS:
        tp, fp, fn = np.sum((y == l) & (p == l)), np.sum((y != l) & (p == l)), np.sum((y == l) & (p != l))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def recall_by_error(rows, preds):
    out = {}
    for et in sorted({r["error_type"] for r in rows if r["label"] == "REFUTED"}):
        idx = [i for i, r in enumerate(rows) if r["error_type"] == et]
        out[et] = float(np.mean([preds[i] == "REFUTED" for i in idx]))
    return out


def cluster_bootstrap(rows, preds, fn=accuracy, n=500, seed=0):
    """90% CI resampling companies (claims within a company are not independent)."""
    rng, tick = np.random.default_rng(seed), sorted({r["ticker"] for r in rows})
    by = {t: [i for i, r in enumerate(rows) if r["ticker"] == t] for t in tick}
    vals = []
    for _ in range(n):
        idx = [i for t in rng.choice(tick, len(tick)) for i in by[t]]
        vals.append(fn([rows[i]["label"] for i in idx], [preds[i] for i in idx]))
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def cohen_kappa(a, b):
    a, b = np.array(a), np.array(b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == l) * np.mean(b == l) for l in set(a) | set(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
