"""Claim generation with gold labels. Labels: SUPPORTED / REFUTED / NOT_ENOUGH_INFO.
Error types for REFUTED: wrong_number, wrong_period, unit_scale, wrong_direction, wrong_entity.
NOT_ENOUGH_INFO: items or years that the evidence does not contain.
Claims are template-generated (so labels are exact); natural-language claims need human labelling (see run.py labelsheet)."""
import hashlib
import json
import random
from .facts import ITEMS, MARGINS

NEI_ITEMS = ["research and development expense", "number of employees", "share repurchases", "dividends per share",
             "capital expenditure on data centres"]


def money(v, unit=None):
    unit = unit or ("billion" if abs(v) >= 1e9 else "million")
    d = 1e9 if unit == "billion" else 1e6
    return f"${v / d:,.1f} billion" if unit == "billion" else f"${v / d:,.0f} million"


def evidence_text(rec):
    lines = [f"Source: annual financial statements of {rec['name']} ({rec['ticker']}), USD millions."]
    for item, series in rec["facts"].items():
        lines.append(f"{item}: " + "; ".join(f"FY{y}={v / 1e6:,.1f}" for y, v in sorted(series.items())))
    return "\n".join(lines)


def _margin(f, name, y):
    a, b = MARGINS[name]
    if y in f.get(a, {}) and y in f.get(b, {}) and f[b][y] > 0:
        return 100 * f[a][y] / f[b][y]
    return None


def _growth(f, item, y):
    s = f.get(item, {})
    if y in s and y - 1 in s and s[y] > 0 and s[y - 1] > 0:
        return 100 * (s[y] / s[y - 1] - 1)
    return None


def _split(ticker, seed):
    h = int(hashlib.md5(f"{seed}{ticker}".encode()).hexdigest(), 16) % 10
    return "train" if h < 6 else "dev" if h < 8 else "test"


def _base_claims(rec, rng):
    f, nm, out = rec["facts"], rec["name"], []
    for item, s in f.items():
        for y in s:
            out.append(dict(kind="level", item=item, year=y, value=s[y]))
            g = _growth(f, item, y)
            if g is not None and abs(g) >= 1.0:
                out.append(dict(kind="growth", item=item, year=y, value=g))
    for m in MARGINS:
        for y in sorted({y for s in f.values() for y in s}):
            v = _margin(f, m, y)
            if v is not None:
                out.append(dict(kind="margin", item=m, year=y, value=v))
    rng.shuffle(out)
    return out


def _text(rec, c, value=None, unit=None, year=None, direction=None):
    nm, y = rec["name"], year or c["year"]
    v = c["value"] if value is None else value
    if c["kind"] == "level":
        return f"{nm}'s {c['item']} in fiscal {y} was {money(v, unit)}."
    if c["kind"] == "growth":
        up = (v >= 0) if direction is None else direction
        return f"{nm}'s {c['item']} {'grew' if up else 'fell'} {abs(v):.1f}% from fiscal {y - 1} to fiscal {y}."
    return f"{nm}'s {c['item']} in fiscal {y} was {v:.1f}%."


def _refute(rec, c, others, rng):
    f, v = rec["facts"], c["value"]
    opts = ["wrong_number"]
    if c["kind"] == "level":
        opts += ["wrong_period", "unit_scale", "wrong_entity"]
    if c["kind"] == "growth":
        opts += ["wrong_direction"]
    et = rng.choice(opts)
    if et == "wrong_number":
        if c["kind"] == "level":
            nv = v * (1 + rng.choice([-1, 1]) * rng.uniform(0.08, 0.35))
        else:
            nv = v + rng.choice([-1, 1]) * rng.uniform(4, 15)
            nv = abs(nv) if c["kind"] == "growth" and v >= 0 else nv
        return et, _text(rec, c, nv)
    if et == "wrong_period":
        alt = [y for y in f[c["item"]] if y != c["year"] and abs(f[c["item"]][y] / v - 1) > 0.06]
        if not alt:
            return _refute(rec, c, others, rng) if len(opts) > 1 and False else ("wrong_number", _text(rec, c, v * 1.2))
        return et, _text(rec, c, f[c["item"]][rng.choice(alt)])
    if et == "unit_scale":
        unit = "million" if abs(v) >= 1e9 else "billion"
        return et, f"{rec['name']}'s {c['item']} in fiscal {c['year']} was ${(v / (1e9 if abs(v) >= 1e9 else 1e6)):,.1f} {unit}."
    if et == "wrong_direction":
        return et, _text(rec, c, direction=(c["value"] < 0))
    cand = [o for o in others if c["item"] in o["facts"] and c["year"] in o["facts"][c["item"]]
            and abs(o["facts"][c["item"]][c["year"]] / v - 1) > 0.1]
    if not cand:
        return "wrong_number", _text(rec, c, v * 1.25)
    return et, _text(rec, c, rng.choice(cand)["facts"][c["item"]][c["year"]])


def _nei(rec, rng):
    ys = sorted({y for s in rec["facts"].values() for y in s})
    if rng.random() < 0.5:
        item = rng.choice(NEI_ITEMS)
        y = rng.choice(ys)
        txt = (f"{rec['name']}'s {item} in fiscal {y} was ${rng.uniform(0.5, 40):.1f} billion." if "number" not in item
               else f"{rec['name']}'s {item} in fiscal {y} was {rng.randint(2000, 200000):,}.")
    else:
        y = max(ys) + rng.choice([1, 2])
        item = rng.choice(list(rec["facts"]))
        txt = f"{rec['name']}'s {item} in fiscal {y} was ${rng.uniform(1, 300):.1f} billion."
    return txt


def build(records, per_company=30, seed=0):
    rng, rows = random.Random(seed), []
    for rec in records:
        others, ev = [o for o in records if o["ticker"] != rec["ticker"]], evidence_text(rec)
        base = _base_claims(rec, rng)
        n_s, n_r, n_n = int(per_company * .4), int(per_company * .4), per_company - 2 * int(per_company * .4)
        for i, c in enumerate(base[:n_s + n_r]):
            if i < n_s:
                lab, et, txt = "SUPPORTED", "none", _text(rec, c)
            else:
                et, txt = _refute(rec, c, others, rng)
                lab = "REFUTED"
            rows.append(dict(ticker=rec["ticker"], claim_type=c["kind"], claim=txt, label=lab, error_type=et))
        for _ in range(n_n):
            rows.append(dict(ticker=rec["ticker"], claim_type="missing", claim=_nei(rec, rng),
                             label="NOT_ENOUGH_INFO", error_type="missing_data"))
    evs = {r["ticker"]: evidence_text(r) for r in records}
    for i, r in enumerate(rows):
        r.update(id=i, evidence=evs[r["ticker"]], split=_split(r["ticker"], seed))
    return rows


def save(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load(path):
    return [json.loads(l) for l in open(path)]
