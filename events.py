"""Company event timeline. Confirmed (filings) vs reported (news) vs model interpretation are labelled separately."""
import re
from datetime import datetime

KEYWORDS = [("earnings", r"earnings|quarterly results|eps|revenue (rose|fell|beat|miss)"),
            ("acquisition", r"acqui|merger|buyout|takeover"), ("lawsuit", r"lawsuit|sued|litigation|settle|probe|investigation"),
            ("guidance", r"guidance|outlook|forecast"), ("product", r"launch|unveil|announce[sd]? (new|a new)|release"),
            ("management", r"ceo|cfo|chief .* officer|resign|appoint|steps down"),
            ("regulatory", r"sec |fda|ftc|doj|antitrust|export (control|ban)|tariff|regulat"),
            ("funding", r"funding|raises \$|series [a-f]\b|ipo"), ("analyst", r"upgrade|downgrade|price target|rating")]
ITEMS = {"1.01": "material agreement", "2.01": "acquisition/disposition completed", "2.02": "earnings release",
         "2.05": "restructuring", "2.06": "impairment", "4.01": "auditor change", "4.02": "non-reliance on prior financials",
         "5.02": "officer/director change", "7.01": "Reg FD", "8.01": "other event"}
SIGNIFICANCE = {"earnings": "Updates the financial baseline.", "acquisition": "Possible change in strategy or leverage.",
                "lawsuit": "Possible legal cost or reputational risk.", "guidance": "Resets expectations.",
                "product": "Possible growth driver; revenue impact unproven.", "management": "Possible change in execution risk.",
                "regulatory": "Possible change in operating constraints.", "funding": "Capital-structure change.",
                "analyst": "Opinion of a third party, not a company fact.", "other": ""}


def classify(title):
    for kind, pat in KEYWORDS:
        if re.search(pat, title, re.I):
            return kind
    return "other"


def build_timeline(news=(), filings=(), earnings_dates=()):
    ev = []
    for f in filings:
        its = [i for i in (f.get("items") or "").split(",") if i]
        kinds = "; ".join(ITEMS.get(i, i) for i in its)
        ev.append({"date": f["date"], "type": "filing", "title": f"{f['form']}" + (f": {kinds}" if kinds else ""),
                   "status": "CONFIRMED (SEC filing)", "source": f.get("url", ""), "significance": ""})
    for n in news:
        k = classify(n.get("Title") or n.get("title") or "")
        ev.append({"date": str(n.get("Date") or n.get("published") or "")[:10], "type": k,
                   "title": n.get("Title") or n.get("title"), "status": "REPORTED (news; verify)",
                   "source": n.get("URL") or n.get("url") or "", "significance": SIGNIFICANCE[k]})
    for d in earnings_dates:
        ev.append({"date": str(d)[:10], "type": "earnings", "title": "Earnings date", "status": "SCHEDULED/REPORTED",
                   "source": "Yahoo Finance", "significance": SIGNIFICANCE["earnings"]})
    ev.sort(key=lambda e: e["date"], reverse=True)
    return {"events": ev, "caveat": "Significance notes are generic interpretations. Timing proximity to price moves "
                                    "is not evidence of causation."}


def next_earnings(earnings_dates):
    today = datetime.now().strftime("%Y-%m-%d")
    fut = sorted(str(d)[:10] for d in earnings_dates if str(d)[:10] >= today)
    return fut[0] if fut else None
