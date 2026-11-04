"""Earnings releases (SEC 8-K Item 2.02, Exhibit 99.1) and management-guidance change detection.
Full earnings-call transcripts are not freely/legally downloadable in bulk; the press release carries the official outlook, and
you can add transcript text yourself with FilingIndex.add_transcript."""
import re

import edgar

METRICS = [("revenue", r"revenue|net sales|sales"), ("gross_margin", r"gross margin"), ("operating_expenses", r"operating expenses|opex"),
           ("operating_margin", r"operating margin"), ("eps", r"eps|earnings per share"), ("capex", r"capital expenditures|capex"),
           ("tax_rate", r"tax rate")]
_NUM = r"\$?\s*(\d[\d,]*\.?\d*)\s*(billion|million|%|percent)?"
_RANGE = re.compile(_NUM + r"\s*(?:to|-|–|and)\s*" + _NUM, re.I)
_POINT = re.compile(r"(?:approximately|about|expected to be|to be|of)\s+" + _NUM, re.I)
_SCALE = {"billion": 1e9, "million": 1e6}


def _val(num, unit):
    v = float(num.replace(",", ""))
    unit = (unit or "").lower()
    return v if unit in ("%", "percent") else v * _SCALE.get(unit, 1)


def extract_guidance(text):
    """Forward-looking numeric statements -> [{'metric','low','high','sentence'}]. Heuristic: needs an outlook/expect cue in the sentence."""
    out = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        if not re.search(r"outlook|expect|guidance|forecast|anticipate|project", s, re.I):
            continue
        if re.search(r"preliminary|versus|\bvs\.?\b|compared (?:to|with)|previously|prior outlook|reported|was \$|were \$|year[- ]over[- ]year", s, re.I) \
                and not re.search(r"\b(?:we|company|management)\s+(?:now\s+)?(?:expects?|anticipates?)\b|\bis expected\b|\bare expected\b|outlook for", s, re.I):
            continue                                              # reports an actual/preliminary result or a comparison, not new guidance
        metric = next((m for m, pat in METRICS if re.search(pat, s, re.I)), None)
        if not metric:
            continue
        m = _RANGE.search(s)
        if m:
            u = m.group(4) or m.group(2)                          # a single unit may follow only the 2nd number
            lo, hi = _val(m.group(1), m.group(2) or u), _val(m.group(3), m.group(4) or u)
        else:
            p = _POINT.search(s)
            if not p:
                continue
            lo = hi = _val(p.group(1), p.group(2))
        out.append({"metric": metric, "low": min(lo, hi), "high": max(lo, hi), "sentence": s.strip()[:240]})
    seen, uniq = set(), []
    for g in out:                                                 # keep the first statement per metric
        if g["metric"] not in seen:
            seen.add(g["metric"]); uniq.append(g)
    return uniq


def compare_guidance(prev, cur, tol=0.01):
    """Midpoint change per metric between two guidance lists. 'raised'/'lowered' beyond +-tol (relative); expense/tax items are labelled neutrally."""
    pm = {g["metric"]: g for g in prev}
    res = []
    for g in cur:
        p = pm.get(g["metric"])
        if not p:
            continue
        a, b = (p["low"] + p["high"]) / 2, (g["low"] + g["high"]) / 2
        ch = (b - a) / abs(a) if a else 0.0
        word = "maintained" if abs(ch) <= tol else ("raised" if ch > 0 else "lowered")
        res.append({"metric": g["metric"], "change": ch, "direction": word, "prev_mid": a, "new_mid": b})
    return res


def classify_guidance_event(changes):
    """One label for the release: guidance_raise / guidance_lower / guidance_mixed / guidance_maintain (favourable metrics only: revenue, margins, EPS)."""
    fav = [c for c in changes if c["metric"] in ("revenue", "gross_margin", "operating_margin", "eps")]
    if not fav:
        return None
    dirs = {c["direction"] for c in fav}
    if dirs == {"maintained"}: return "guidance_maintain"
    if dirs <= {"raised", "maintained"}: return "guidance_raise"
    if dirs <= {"lowered", "maintained"}: return "guidance_lower"
    return "guidance_mixed"


def fetch_earnings_releases(cik, limit=8):
    """[{'date','accession','url','text'}] for 8-K Item 2.02 filings: the Exhibit 99.x press release text. Needs SEC access."""
    out = []
    for f in edgar.list_filings(cik, forms=("8-K",), limit=200):
        if "2.02" not in (f.get("items") or ""):
            continue
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{f['accession'].replace('-', '')}/"
        try:
            idx = edgar._get(base + "index.json", True)
            docs = [d["name"] for d in idx["directory"]["item"] if re.search(r"(?i)(ex-?99|exhibit99|ex99|q\d.*release|press)", d["name"]) and d["name"].lower().endswith((".htm", ".html"))]
            if not docs:
                continue
            out.append({"date": f["date"], "accession": f["accession"], "url": base + docs[0], "text": edgar.html_to_text(edgar._get(base + docs[0]))})
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def guidance_history(releases):
    """releases newest-first -> chronological list of events with category for the event study."""
    rel = sorted(releases, key=lambda r: r["date"])
    prev, events = None, []
    for r in rel:
        g = extract_guidance(r["text"])
        if prev is not None and g:
            ch = compare_guidance(prev, g)
            cat = classify_guidance_event(ch)
            if cat:
                events.append({"date": r["date"], "category": cat, "changes": ch, "url": r["url"]})
        if g:
            prev = g
    return events
