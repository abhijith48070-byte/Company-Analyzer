"""SEC EDGAR client: submissions, XBRL facts, filing text, section extraction, 10-K diffing, 8-K/Form 4 parsing.
SEC fair-access: descriptive User-Agent required (set SEC_USER_AGENT), <=10 requests/second."""
import difflib
import os
import re
import time

import warnings

import requests
from bs4 import XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)  # inline-XBRL filings start with an XML header; HTML parsing is fine

UA = os.getenv("SEC_USER_AGENT", "CompanyAnalyzer research (set-SEC_USER_AGENT@example.com)")
_last = [0.0]


def _get(url, as_json=False):
    wait = 0.12 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    r = requests.get(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}, timeout=30)
    r.raise_for_status()
    return r.json() if as_json else r.text


def filing_url(cik, accession, doc):
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{doc}"


def list_filings(cik, forms=("10-K", "10-Q", "8-K", "DEF 14A", "4"), limit=20):
    recent = _get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", True)["filings"]["recent"]
    out = []
    for i, f in enumerate(recent["form"]):
        if f in forms:
            out.append({"form": f, "date": recent["filingDate"][i], "accession": recent["accessionNumber"][i],
                        "doc": recent["primaryDocument"][i], "items": recent.get("items", [""] * 99)[i],
                        "url": filing_url(cik, recent["accessionNumber"][i], recent["primaryDocument"][i])})
            if len(out) >= limit:
                break
    return out


def company_facts(cik):
    return _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json", True)


def fact_series(facts, concept, unit="USD", form="10-K"):
    """Annual series for a us-gaap concept -> [(fiscal_end, value, accession)], newest first, deduped."""
    try:
        rows = facts["facts"]["us-gaap"][concept]["units"][unit]
    except KeyError:
        return []
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: r["end"], reverse=True):
        if r.get("form") == form and r["end"] not in seen and (r.get("fp") == "FY"):
            seen.add(r["end"])
            out.append((r["end"], r["val"], r["accn"]))
    return out


def html_to_text(html):
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for t in soup(["script", "style"]):
            t.decompose()
        txt = soup.get_text("\n")
    except Exception:
        txt = re.sub(r"<[^>]+>", "\n", html)
    return re.sub(r"\n\s*\n+", "\n\n", re.sub(r"[ \t\xa0]+", " ", txt))


def get_filing_text(filing):
    return html_to_text(_get(filing["url"]))


SECTIONS = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "3": "Legal Proceedings",
    "7": "MD&A", "7A": "Market Risk", "8": "Financial Statements", "9A": "Controls and Procedures",
}


def extract_sections(text):
    """Split a 10-K into {item: text}. Takes the LAST 'Item X' heading occurrence (skips the TOC)."""
    pat = re.compile(r"(?im)^\s*item\s+(1a|1b|1|2|3|4|5|6|7a|7|8|9a|9b|9|10|11|12|13|14|15)\s*[\.:\-—–]?\s*(.*)$")
    marks = [(m.group(1).upper(), m.start()) for m in pat.finditer(text)]
    best = {}
    for i, (item, pos) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        if item not in best or (end - pos) > (best[item][1] - best[item][0]):  # longest span beats TOC stub
            best[item] = (pos, end)
    return {k: text[a:b].strip() for k, (a, b) in best.items()}


def _paras(txt, min_len=80):
    return [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", txt) if len(p.strip()) >= min_len]


def diff_sections(old_text, new_text, sim_threshold=0.5):
    """Paragraph-level diff by word-set Jaccard similarity: a paragraph with no similar counterpart
    in the other filing is 'added' (new filing) or 'removed' (old filing)."""
    words = lambda p: set(re.findall(r"[a-z0-9$%.]+", p.lower()))
    old_p, new_p = _paras(old_text), _paras(new_text)
    old_w, new_w = [words(p) for p in old_p], [words(p) for p in new_p]

    def unmatched(src, src_w, other_w):
        res = []
        for p, w in zip(src, src_w):
            best = max((len(w & o) / len(w | o) for o in other_w if o), default=0.0)
            if best < sim_threshold:
                res.append(p)
        return res
    return {"added": unmatched(new_p, new_w, old_w), "removed": unmatched(old_p, old_w, new_w)}


def compare_10ks(old_filing, new_filing, old_text, new_text, items=("1A", "3", "7")):
    """What changed between consecutive 10-Ks, each claim carrying filing + section citation."""
    so, sn = extract_sections(old_text), extract_sections(new_text)
    report = []
    for it in items:
        if it not in sn or it not in so:
            continue
        d = diff_sections(so[it], sn[it])
        cite = lambda f: f"{f['form']} filed {f['date']}, Item {it} ({SECTIONS.get(it, '')}) {f['url']}"
        report.append({"item": it, "name": SECTIONS.get(it, ""),
                       "new": [{"text": p[:600], "source": cite(new_filing)} for p in d["added"][:15]],
                       "removed": [{"text": p[:600], "source": cite(old_filing)} for p in d["removed"][:15]],
                       "size_change_pct": (len(sn[it]) - len(so[it])) / max(len(so[it]), 1) * 100})
    return report


def financial_changes(facts):
    """YoY changes in key XBRL concepts with filing accession citations."""
    concepts = {"Revenues": "Revenue", "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue",
                "LongTermDebt": "Long-term debt", "Goodwill": "Goodwill", "InventoryNet": "Inventory",
                "NetCashProvidedByUsedInOperatingActivities": "Operating cash flow"}
    out, seen = [], set()
    for c, label in concepts.items():
        s = fact_series(facts, c)
        if len(s) >= 2 and label not in seen:
            seen.add(label)
            (e1, v1, a1), (e0, v0, _) = s[0], s[1]
            out.append({"metric": label, "period": e1, "value": v1, "prior": v0,
                        "change_pct": (v1 - v0) / abs(v0) * 100 if v0 else None, "accession": a1})
    return out


ITEM_8K = {"1.01": "Material agreement", "1.02": "Agreement termination", "2.01": "Acquisition/disposition completed",
           "2.02": "Earnings results", "2.05": "Restructuring costs", "2.06": "Material impairment",
           "4.01": "Auditor change", "4.02": "Non-reliance on prior financials", "5.02": "Officer/director change",
           "7.01": "Reg FD", "8.01": "Other events"}


def classify_8k(items_str):
    return [{"item": i, "meaning": ITEM_8K.get(i, "Other")} for i in items_str.split(",") if i]


def parse_form4(xml):
    """Insider transactions from a Form 4 XML document."""
    from bs4 import BeautifulSoup
    s = BeautifulSoup(xml, "xml")
    owner = s.find("rptOwnerName")
    tx = []
    for t in s.find_all("nonDerivativeTransaction"):
        g = lambda tag: (t.find(tag).find("value").text if t.find(tag) and t.find(tag).find("value") else None)
        tx.append({"date": g("transactionDate"), "code": g("transactionCode"), "shares": g("transactionShares"),
                   "price": g("transactionPricePerShare"), "acq_disp": g("transactionAcquiredDisposedCode")})
    return {"insider": owner.text if owner else None, "transactions": tx}


# ------------------------------------------------ multi-year history (XBRL)
XBRL_MAP = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "depreciation": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "Depreciation"],
    "ebit": ["OperatingIncomeLoss"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
    "pretax": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
               "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "tax": ["IncomeTaxExpenseBenefit"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "receivables": ["AccountsReceivableNetCurrent", "ReceivablesNetCurrent"],
    "inventory": ["InventoryNet"],
    "current_assets": ["AssetsCurrent"],
    "ppe": ["PropertyPlantAndEquipmentNet"],
    "total_assets": ["Assets"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "total_liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}
_FLOWS = {"revenue", "cogs", "gross_profit", "sga", "depreciation", "ebit", "interest_expense", "pretax", "tax", "net_income", "ocf", "capex"}


def _annual(facts, concepts, flow):
    """{fiscal_end: (value, filed)}. 10-K annual facts only; flows must span ~1 year; for each end the LATEST filing wins (restatements)."""
    from datetime import date
    out = {}
    for c in concepts:
        try:
            rows = facts["facts"]["us-gaap"][c]["units"]["USD"]
        except KeyError:
            continue
        for r in rows:
            if r.get("form") not in ("10-K", "10-K/A") or r.get("fp") != "FY":
                continue
            if flow:
                try:
                    d = (date.fromisoformat(r["end"]) - date.fromisoformat(r["start"])).days
                except Exception:
                    continue
                if not 330 <= d <= 380:
                    continue
            if r["end"] not in out or r.get("filed", "") > out[r["end"]][1]:
                out[r["end"]] = (float(r["val"]), r.get("filed", ""))
        if out:
            break                        # first concept that has data wins (avoids mixing definitions)
    return out


def periods_from_xbrl(facts, years=12):
    """SEC companyfacts -> list of period dicts (same fields as fin_model.normalize, newest first) with a 'filed' date for point-in-time use.
    Gives up to ~10+ fiscal years instead of Yahoo's 4. Missing items are 0 (as with the Yahoo path); a period needs assets, revenue."""
    series = {k: _annual(facts, v, k in _FLOWS) for k, v in XBRL_MAP.items()}
    ends = sorted(set(series["total_assets"]) & set(series["revenue"]), reverse=True)[:years]
    out = []
    for e in ends:
        row = {"date": e, "filed": max((series[k][e][1] for k in series if e in series[k]), default="")}
        for k in XBRL_MAP:
            row[k] = series[k].get(e, (0.0, ""))[0]
        row["shares"] = 0.0
        if not row["gross_profit"] and row["revenue"] and row["cogs"]:
            row["gross_profit"] = row["revenue"] - row["cogs"]
        if not row["total_liabilities"] and row["total_assets"]:
            row["total_liabilities"] = row["total_assets"] - row["equity"]
        row["total_debt"] = row["lt_debt"]; row["capex"] = abs(row["capex"])
        out.append(row)
    return out
