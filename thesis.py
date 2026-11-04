"""Investment thesis generator. Every line is tagged FACT (reported data), MODEL (computed under stated
assumptions) or INTERPRETATION (rule-based judgement). No Buy/Sell call."""


def F(t, s): return {"text": t, "tag": "FACT", "source": s}
def M(t, s): return {"text": t, "tag": "MODEL", "source": s}
def I(t, s): return {"text": t, "tag": "INTERPRETATION", "source": s}


def _p(x, d=1): return "n/a" if x is None else f"{x * 100:.{d}f}%"
def _b(x):
    if x is None: return "n/a"
    for div, s in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(x) >= div: return f"${x / div:,.2f}{s}"
    return f"${x:,.2f}"


def build_thesis(res):
    m, v, fo = res["metrics"], res.get("valuation") or {}, res.get("forensics") or {}
    ml, comp, ev, fl = res.get("ml") or {}, res.get("competitors") or {}, res.get("events") or {}, res.get("filings") or {}
    T = {}
    # ---- business
    b = []
    if m.get("Summary"): b.append(F(m["Summary"][:700] + ("..." if len(m["Summary"]) > 700 else ""), "Yahoo Finance profile"))
    b.append(F(f"{m.get('Sector')} / {m.get('Industry')}; revenue (TTM) {_b(m.get('Revenue'))}; employees {m.get('Employees') or 'n/a'}.", "Yahoo Finance"))
    T["business_overview"] = b
    # ---- financial quality
    q, facts = [], fo.get("facts", {}) if "error" not in fo else {}
    q.append(F(f"Gross margin {_p(m.get('GrossMargin'))}, operating margin {_p(m.get('OperatingMargin'))}, net margin {_p(m.get('ProfitMargin'))}, ROE {_p(m.get('Roe'))}.", "Yahoo Finance"))
    if m.get("Fcf") is not None and m.get("Revenue"):
        q.append(F(f"Free cash flow {_b(m['Fcf'])} ({_p(m['Fcf'] / m['Revenue'])} of revenue); cash {_b(m.get('Cash'))}, debt {_b(m.get('Debt'))}.", "Yahoo Finance"))
    cc = facts.get("cash_conversion_ocf_to_ni")
    if cc is not None: q.append(F(f"Operating cash flow / net income = {cc:.2f}; Sloan accrual ratio {_p(facts.get('sloan_accrual_ratio'))}.", "financial statements"))
    for w in fo.get("warnings", []): q.append(F(f"{w['signal']}: {w['evidence']}", "forensics engine (statistical signal)"))
    high = [w for w in fo.get("warnings", []) if w["severity"] == "high"]
    z = (res.get("altman") or {}).get("Zscore")
    level = ("Weak" if (high and (cc or 1) < 0.8) or (z is not None and z < 4.15) else
             "Mixed" if high or (cc is not None and cc < 0.8) else "Strong")
    q.append(I(f"Earnings-quality read: {level}. Based on {len(fo.get('warnings', []))} warning signals, cash conversion and Altman Z''.", "rule on forensics outputs"))
    T["financial_quality"] = q
    # ---- growth
    g = [F(f"Latest quarter revenue growth (YoY, Yahoo) {_p(m.get('Revenuegrowth'))}; earnings growth {_p(m.get('Earningsgrowth'))}.", "Yahoo Finance")]
    fc = ml.get("growth_forecast")
    if fc: g.append(M(f"Bayesian forecast of next-year revenue growth {_p(fc['mean'])} (90% interval {_p(fc['low90'])} to {_p(fc['high90'])}). "
                      f"Validated against persistence baseline: {'beats' if ml.get('growth_validation', {}).get('beats_baseline') else 'does not beat'} it.", "ml.fit_forecaster"))
    if comp.get("percentile_vs_sector"): 
        pr = comp["percentile_vs_sector"]
        g.append(F("Sector percentile: " + ", ".join(f"{k} {v:.0f}" for k, v in pr.items()), "peer comparison (S&P 500 sector)"))
    T["growth"] = g
    # ---- valuation
    va, rd, sc = [], v.get("reverse_dcf") or {}, v.get("scenarios") or {}
    if v.get("wacc"): va.append(M(f"WACC {_p(v['wacc']['wacc'], 2)} (cost of equity {_p(v['wacc']['cost_of_equity'], 2)}, adjusted beta {v['wacc']['beta']:.2f}).", "valuation.compute_wacc"))
    if rd.get("implied_fcf_growth") is not None:
        va.append(M(f"At {_b(m.get('Currentprice'))}/share the market implies starting FCF growth of {_p(rd['implied_fcf_growth'])} fading to terminal over 10 years, "
                    f"or a starting FCF margin of {_p(rd.get('implied_fcf_margin'))} vs {_p(rd.get('current_fcf_margin'))} today.", "valuation.reverse_dcf"))
    if "base" in sc:
        va.append(M("DCF scenarios per share: bear {} / base {} / bull {}.".format(*(f"${sc[k]['value']:.2f}" for k in ("bear", "base", "bull"))), "valuation.scenarios"))
    if v.get("residual_income"): va.append(M(f"Residual income model ${v['residual_income']:.2f}/share.", "valuation.residual_income"))
    if v.get("comps"): va.append(M("Peer-multiple implied prices: " + ", ".join(f"{k}={x:.2f}" for k, x in v["comps"].items()), "valuation.comps_valuation"))
    if sc.get("base") and m.get("Currentprice"):
        gap = sc["base"]["value"] / m["Currentprice"] - 1
        va.append(I(f"Price is {'above' if gap < 0 else 'below'} the base DCF by {abs(gap):.0%}. {'Expectations exceed what the base assumptions deliver.' if gap < -0.15 else 'Roughly in line with base assumptions.' if abs(gap) <= 0.15 else 'Base assumptions are more optimistic than the market.'}", "rule on scenario vs price"))
    T["valuation"] = va
    # ---- risks
    r = {"financial": [], "operational": [], "regulatory": [], "competitive": [], "governance": [], "valuation": []}
    if facts.get("interest_coverage") is not None: r["financial"].append(F(f"Interest coverage {facts['interest_coverage']:.1f}x; net debt/EBITDA {facts.get('net_debt_to_ebitda')}.", "financial statements"))
    pd_ = ml.get("distress_probability")
    if pd_ is not None: r["financial"].append(M(f"Model probability of material Altman Z'' deterioration next year: {pd_:.0%} (validated AUC {ml.get('distress_validation', {}).get('models', {}).get(ml.get('distress_validation', {}).get('best'), {}).get('auc')}).", "ml.fit_distress"))
    if facts.get("inventory_growth") is not None and facts.get("revenue_growth") is not None and facts["inventory_growth"] - facts["revenue_growth"] > 0.15:
        r["operational"].append(F(f"Inventory grew {_p(facts['inventory_growth'])} vs revenue {_p(facts['revenue_growth'])}.", "financial statements"))
    for sec in (fl.get("10k_changes") or []):
        if sec["item"] == "1A":
            for n in sec["new"][:4]: r["regulatory"].append(F("New risk-factor language: " + n["text"][:220] + "...", n["source"]))
    for t in comp.get("threats", []): r["competitive"].append(F(f"{t['ticker']} is growing faster with equal or better margins ({t['evidence']}).", "peer comparison"))
    for gf in fl.get("governance_flags", []): r["governance"].append(F(f"{gf['flag']}: ...{gf['context'][:180]}...", "latest 10-K text search"))
    if rd.get("implied_fcf_growth") is not None and rd["implied_fcf_growth"] > 0.25:
        r["valuation"].append(I(f"The price requires very high sustained growth ({_p(rd['implied_fcf_growth'])} starting FCF growth); small disappointments could reprice it.", "rule on reverse DCF"))
    an = ml.get("anomaly")
    if an and an.get("anomaly_percentile", 0) > 90: r["operational"].append(M(f"Financial profile more unusual than {an['anomaly_percentile']:.0f}% of sector peers (Isolation Forest).", "ml.anomaly_scan"))
    T["risks"] = r
    # ---- cases
    cases = {}
    for k in ("bear", "base", "bull"):
        if k in sc:
            s = sc[k]; cases[k] = M(f"{k.title()}: FCF growth starts at {_p(s['growth'])}, WACC {_p(s['wacc'], 2)}, terminal growth {_p(s['terminal_growth'])} -> ${s['value']:.2f}/share.", "valuation.scenarios")
    T["cases"] = cases
    # ---- catalysts and invalidators
    cat = []
    if ev.get("next_earnings"): cat.append(F(f"Next scheduled earnings date: {ev['next_earnings']}.", "Yahoo Finance"))
    for e in (ev.get("events") or [])[:5]:
        if e["type"] in ("filing", "acquisition", "guidance", "regulatory", "management"): cat.append(F(f"{e['date']} {e['title']} [{e['status']}]", e["source"]))
    T["catalysts"] = cat
    inv = []
    if rd.get("implied_fcf_margin") and rd.get("current_fcf_margin") is not None and rd["implied_fcf_margin"] > rd["current_fcf_margin"] * 1.5:
        inv.append(I(f"If FCF margin does not move from {_p(rd['current_fcf_margin'])} toward {_p(rd['implied_fcf_margin'])}, the current price is hard to justify at constant growth.", "reverse DCF"))
    if fo.get("warnings"): inv.append(I("If accrual and receivable/inventory warnings persist for another year, treat reported earnings as lower quality.", "forensics"))
    inv.append(I("If revenue growth falls below the growth path assumed in the base case, the base valuation falls with it.", "scenario structure"))
    T["thesis_invalidators"] = inv
    # ---- final assessment (no buy/sell)
    T["final_assessment"] = {
        "financial_quality": level,
        "growth_quality": "High" if (m.get("Revenuegrowth") or 0) > 0.15 and (cc or 0) >= 0.8 else "Moderate" if (m.get("Revenuegrowth") or 0) > 0.05 else "Low",
        "valuation": "Demanding" if (rd.get("implied_fcf_growth") or 0) > 0.25 else "Reasonable" if rd.get("implied_fcf_growth") is not None else "Undetermined",
        "financial_risk": "Elevated" if (z is not None and z < 4.15) or (pd_ or 0) > 0.3 else "Contained" if z is not None else "Undetermined",
        "competitive_position": "Under pressure" if comp.get("threats") else "No faster-growing, higher-margin peer found" if comp else "Not assessed",
        "note": "Rule-based summary of the sections above; not a recommendation."}
    return T


def render_thesis(T, name=""):
    L = [f"# Investment thesis: {name}", "_Tags: FACT = reported data - MODEL = computed under stated assumptions - INTERPRETATION = rule-based judgement._\n"]
    line = lambda x: f"- **[{x['tag']}]** {x['text']}  \n  _source: {x['source']}_"
    for key, title in (("business_overview", "Business overview"), ("financial_quality", "Financial quality"), ("growth", "Growth"), ("valuation", "Valuation")):
        L += [f"## {title}"] + [line(x) for x in T.get(key, [])] + [""]
    L.append("## Risks")
    for k, items in T["risks"].items():
        if items: L += [f"### {k.title()}"] + [line(x) for x in items]
    L += ["", "## Bull / base / bear"] + [line(T["cases"][k]) for k in ("bull", "base", "bear") if k in T["cases"]]
    L += ["", "## Catalysts"] + [line(x) for x in T["catalysts"]] + ["", "## What would invalidate this thesis"] + [line(x) for x in T["thesis_invalidators"]]
    fa = T["final_assessment"]
    L += ["", "## Final assessment"] + [f"- **{k.replace('_', ' ').title()}:** {v}" for k, v in fa.items() if k != "note"] + [f"\n_{fa['note']}_"]
    return "\n".join(L)
