"""Financial forensics: Beneish M, Sloan accruals, cash conversion, divergences, debt sustainability.
Outputs are warning signals with evidence, never fraud labels."""


def _div(a, b):
    return a / b if b else None


def beneish_m(cur, prev):
    """8-variable Beneish M-Score from two normalised periods (cur, prev)."""
    try:
        dsri = _div(_div(cur["receivables"], cur["revenue"]), _div(prev["receivables"], prev["revenue"]))
        gmi = _div(_div(prev["gross_profit"], prev["revenue"]), _div(cur["gross_profit"], cur["revenue"]))
        aq = lambda p: 1 - (p["current_assets"] + p["ppe"]) / p["total_assets"]
        aqi = _div(aq(cur), aq(prev))
        sgi = _div(cur["revenue"], prev["revenue"])
        dep = lambda p: _div(p["depreciation"], p["depreciation"] + p["ppe"])
        depi = _div(dep(prev), dep(cur))
        sgai = _div(_div(cur["sga"], cur["revenue"]), _div(prev["sga"], prev["revenue"]))
        lvgi = _div(_div(cur["total_liabilities"], cur["total_assets"]),
                    _div(prev["total_liabilities"], prev["total_assets"]))
        tata = (cur["net_income"] - cur["ocf"]) / cur["total_assets"]
    except (ZeroDivisionError, KeyError, TypeError):
        return {"error": "insufficient data"}
    idx = dict(DSRI=dsri, GMI=gmi, AQI=aqi, SGI=sgi, DEPI=depi, SGAI=sgai, LVGI=lvgi, TATA=tata)
    # missing components (e.g. SG&A not reported) are set to the neutral value 1 (0 for TATA)
    imputed = [k for k, v in idx.items() if v is None]
    for k in imputed:
        idx[k] = 1.0
    m = (-4.84 + 0.920 * idx["DSRI"] + 0.528 * idx["GMI"] + 0.404 * idx["AQI"] + 0.892 * idx["SGI"]
         + 0.115 * idx["DEPI"] - 0.172 * idx["SGAI"] + 4.679 * idx["TATA"] - 0.327 * idx["LVGI"])
    return {"m_score": m, "components": idx, "imputed": imputed,
            "flag": m > -1.78}  # -1.78 is the commonly used (8-var) cutoff


def sloan_ratio(cur, prev=None):
    """(NI - OCF) / avg total assets. > +10% concerning, < -10% conservative."""
    ta = (cur["total_assets"] + (prev["total_assets"] if prev else cur["total_assets"])) / 2
    return (cur["net_income"] - cur["ocf"]) / ta if ta else None


def forensic_report(periods):
    """periods: normalised list, newest first (>=2). Returns score + evidence-backed warnings."""
    if len(periods) < 2:
        return {"error": "need at least two periods"}
    cur, prev = periods[0], periods[1]
    warnings, facts = [], {}

    bm = beneish_m(cur, prev)
    facts["beneish"] = bm
    if bm.get("flag"):
        worst = sorted(bm["components"].items(), key=lambda kv: -abs(kv[1] - 1 if kv[0] != "TATA" else kv[1]))[:3]
        warnings.append({"signal": "Beneish M-Score above -1.78", "severity": "high",
                         "evidence": f"M={bm['m_score']:.2f}; largest drivers: " +
                         ", ".join(f"{k}={v:.2f}" for k, v in worst) +
                         ("; NOTE: sales growth (SGI) is a top driver. Fast growers trip this score mechanically, "
                          "so treat as a prompt to check, not a finding" if bm["components"]["SGI"] > 1.4 else "")})

    sl = sloan_ratio(cur, prev)
    facts["sloan_accrual_ratio"] = sl
    if sl is not None and sl > 0.10:
        warnings.append({"signal": "High accruals (Sloan)", "severity": "medium",
                         "evidence": f"(NI-OCF)/avg assets = {sl:.1%}; earnings exceed cash generation"})

    conv = _div(cur["ocf"], cur["net_income"]) if cur["net_income"] > 0 else None
    facts["cash_conversion_ocf_to_ni"] = conv
    if conv is not None and conv < 0.8:
        warnings.append({"signal": "Weak cash conversion", "severity": "medium",
                         "evidence": f"OCF/NI = {conv:.2f} (<0.8)"})

    fcf = cur["ocf"] - cur["capex"]
    facts["fcf"] = fcf
    facts["fcf_conversion"] = _div(fcf, cur["net_income"]) if cur["net_income"] > 0 else None

    def growth(k):
        return _div(cur[k] - prev[k], abs(prev[k])) if prev[k] else None
    rg, arg_, ig = growth("revenue"), growth("receivables"), growth("inventory")
    facts.update(revenue_growth=rg, receivables_growth=arg_, inventory_growth=ig)
    if rg is not None and arg_ is not None and arg_ - rg > 0.15:
        warnings.append({"signal": "Receivables outpacing revenue", "severity": "medium",
                         "evidence": f"receivables {arg_:+.1%} vs revenue {rg:+.1%}"})
    if rg is not None and ig is not None and ig - rg > 0.15:
        warnings.append({"signal": "Inventory outpacing revenue", "severity": "medium",
                         "evidence": f"inventory {ig:+.1%} vs revenue {rg:+.1%}"})

    ebitda = cur["ebit"] + cur["depreciation"]
    cov = _div(cur["ebit"], cur["interest_expense"]) if cur["interest_expense"] else None
    lev = _div(cur["total_debt"] - cur["cash"], ebitda) if ebitda > 0 else None
    facts.update(interest_coverage=cov, net_debt_to_ebitda=lev)
    if cov is not None and cov < 3:
        warnings.append({"signal": "Thin interest coverage", "severity": "high" if cov < 1.5 else "medium",
                         "evidence": f"EBIT/interest = {cov:.1f}x"})
    if lev is not None and lev > 4:
        warnings.append({"signal": "High leverage", "severity": "medium", "evidence": f"net debt/EBITDA = {lev:.1f}x"})
    if ebitda <= 0 and cur["total_debt"] > 0:
        warnings.append({"signal": "Debt with non-positive EBITDA", "severity": "high",
                         "evidence": f"EBITDA = {ebitda:,.0f}; debt = {cur['total_debt']:,.0f}"})

    pts = {"high": 3, "medium": 1}
    score = sum(pts[w["severity"]] for w in warnings)
    level = "elevated" if score >= 4 else "moderate" if score >= 2 else "low"
    return {"risk_level": level, "score": score, "warnings": warnings, "facts": facts,
            "disclaimer": "Statistical warning signals only; not evidence of manipulation or fraud."}


def governance_flags(filing_text):
    """Text-based flags from a 10-K/proxy: related parties, auditor changes, material weakness."""
    import re
    t = filing_text.lower()
    pats = {"related-party transactions": r"related[- ]party (transactions|agreements)",
            "material weakness": r"material weakness",
            "auditor change": r"(dismiss|replac|resign)\w* .{0,40}(auditor|independent registered)",
            "restatement": r"restate(d|ment)",
            "going concern": r"substantial doubt .{0,60}going concern"}
    out = []
    for name, p in pats.items():
        for m in re.finditer(p, t):
            s = max(m.start() - 120, 0)
            out.append({"flag": name, "context": filing_text[s:m.end() + 160].replace("\n", " ")})
            break
    return out
