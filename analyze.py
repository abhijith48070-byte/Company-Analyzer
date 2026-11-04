"""Company Analyzer CLI:  python analyze.py NVDA [--peers] [--edgar] [--llm ollama:qwen2.5:7b]
Prints and saves a markdown research report from computed numbers (no LLM required)."""
import argparse, json, os, sys

import data
from analysis import dcfcalc, altman_z, piotroski_f, rating, get_peer_info
from fin_model import normalize, project_three_statements
from forensics import forensic_report
from valuation import valuation_suite


def pct(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}%"


def money(x):
    if x is None:
        return "n/a"
    a = abs(x)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if a >= div:
            return f"${x / div:,.2f}{suf}"
    return f"${x:,.2f}"


def analyze(ticker, peers=False, edgar=False, max_peers=25, do_ml=False, do_comp=False, do_events=False, questions=(), llm="", do_thesis=False):
    stock, info = data.get_company_info(ticker)
    m = data.get_financialmetrics(info)
    statements = data.finstatements(stock)
    out = {"ticker": ticker, "metrics": m}
    peer_infos = None
    import pipeline
    pmap = {}
    if (peers or do_comp or do_ml) and m.get("Sector"):
        pmap = pipeline.peer_info_map(ticker, m["Sector"], max_peers)
    if peers and m.get("Sector"):
        pt, peer_infos = list(pmap), list(pmap.values())
        out["industry"] = rating(m, pt, peer_infos)
    out["dcf"] = dcfcalc(m)
    out["valuation"] = valuation_suite(m, peer_infos) if m.get("Fcf") and m.get("Sharesoutstanding") else {}
    try:
        periods = normalize(statements)
        out["periods"] = periods
        out["forensics"] = forensic_report(periods)
        out["projection"] = project_three_statements(periods[0], revenue_growth=max(min(m.get("Revenuegrowth") or 0.05, 0.25), -0.05))
    except Exception as e:
        out["forensics"] = {"error": str(e)}
    for k, f in (("altman", altman_z), ("piotroski", piotroski_f)):
        try:
            out[k] = f(statements=statements)
        except Exception as e:
            out[k] = {"error": str(e)}
    if edgar or questions:
        out["filings"] = edgar_section(ticker) if edgar else {}
    tp = out.get("periods") or []
    if do_comp and pmap:
        out["competitors"] = pipeline.run_competitors(ticker, info, pmap, tp)
    if do_ml and m.get("Sector"):
        try:
            out["ml"] = pipeline.run_ml(ticker, stock, list(pmap), tp, m["Sector"])
        except Exception as e:
            out["ml"] = {"error": str(e)}
    if do_events:
        try:
            out["events"] = pipeline.run_events(stock, (out.get("filings") or {}).get("recent_filings"))
        except Exception as e:
            out["events"] = {"events": [], "caveat": f"unavailable: {e}"}
    if questions:
        try:
            from rag import make_llm
            cik = data.get_cik(ticker)
            idx = pipeline.build_rag(ticker, cik, data.News(stock))
            out["rag"] = [{"question": q, **idx.answer(q, llm=make_llm(llm))} for q in questions]
        except Exception as e:
            out["rag"] = [{"question": q, "answer": f"RAG unavailable: {e}", "sources": [], "unsupported_numbers": []} for q in questions]
    if do_thesis:
        import thesis
        out["thesis"] = thesis.build_thesis(out)
    return out


def edgar_section(ticker):
    import edgar
    cik = data.get_cik(ticker)
    if cik is None:
        return {"error": "CIK not found (ticker not in sp500.csv)"}
    res = {}
    try:
        res["recent_filings"] = edgar.list_filings(cik, limit=8)
        res["financial_changes"] = edgar.financial_changes(edgar.company_facts(cik))
        tenks = [f for f in edgar.list_filings(cik, forms=("10-K",), limit=2)]
        if len(tenks) == 2:
            new_t, old_t = (edgar.get_filing_text(f) for f in tenks)
            res["10k_changes"] = edgar.compare_10ks(tenks[1], tenks[0], old_t, new_t)
            from forensics import governance_flags
            res["governance_flags"] = governance_flags(new_t)
    except Exception as e:
        res["error"] = f"EDGAR unavailable: {e}"
    return res


def render(r):
    m, L = r["metrics"], []
    a = L.append
    a(f"# {m.get('Name') or r['ticker']} ({r['ticker']}) - Research Report")
    a(f"*Sector: {m.get('Sector')} | Industry: {m.get('Industry')}*\n")
    a("## Snapshot\n")
    a(f"- Price {money(m.get('Currentprice'))} | Market cap {money(m.get('MarketCap'))} | Beta {m.get('Beta')} (raw, Yahoo)")
    a(f"- Revenue growth {pct(m.get('Revenuegrowth'))} | Profit margin {pct(m.get('ProfitMargin'))} | ROE {pct(m.get('Roe'))}")
    a(f"- Free cash flow {money(m.get('Fcf'))} | Cash {money(m.get('Cash'))} | Debt {money(m.get('Debt'))}\n")
    v = r.get("valuation") or {}
    if v:
        w = v["wacc"]
        a("## Valuation\n")
        a(f"**WACC {pct(w['wacc'],2)}** (cost of equity {pct(w['cost_of_equity'],2)}, beta {w['beta']:.2f}, "
          f"risk-free {pct(w['rf'])}, ERP {pct(w['erp'])}, debt weight {pct(w['weight_debt'])})\n")
        sc = v.get("scenarios", {})
        if "error" in sc:
            a(f"**DCF scenarios unavailable:** {sc['error']}\n")
        if "base" in sc:
            a("| Scenario | FCF growth | WACC | Terminal g | Value/share |\n|---|---|---|---|---|")
            for k in ("bear", "base", "bull"):
                s = sc[k]
                a(f"| {k} | {pct(s['growth'])} | {pct(s['wacc'],2)} | {pct(s['terminal_growth'])} | {money(s['value'])} |")
            fv = v.get("fair_value_range", {})
            a(f"\nFair-value range {money(fv.get('low'))} - {money(fv.get('high'))}; base case margin of safety "
              f"{fv.get('margin_of_safety_base_pct', float('nan')):+.1f}% vs price {money(m.get('Currentprice'))}.\n")
        rd = v.get("reverse_dcf", {})
        if "error" not in rd:
            a("### What the market price implies (reverse DCF)\n")
            a(f"- Implied starting FCF growth (fades linearly to terminal over 10 years): **{pct(rd.get('implied_fcf_growth'))}**" +
              ("" if rd.get('implied_fcf_growth') is not None else " (no solution between -50% and +300%)"))
            a(f"- Implied terminal growth (holding current growth): {pct(rd.get('implied_terminal_growth'),2)}")
            a(f"- Implied FCF margin (starting, at current revenue and growth path): {pct(rd.get('implied_fcf_margin'))} vs current {pct(rd.get('current_fcf_margin'))}")
            a("")
        if v.get("ddm"):
            a(f"- Dividend discount model: {money(v['ddm'])}/share")
        elif v.get("ddm_note"):
            a(f"- Dividend discount model: {v['ddm_note']}")
        if v.get("residual_income"):
            a(f"- Residual income model: {money(v['residual_income'])}/share (assumes ROE fades toward cost of equity)")
        if (m.get("Revenuegrowth") or 0) > 0.40:
            a(f"\n> **Assumption warning:** revenue growth is {pct(m['Revenuegrowth'])} but FCF growth is capped at 40% "
              f"and fades over 10 years. For hypergrowth companies the DCF values are conservative and very sensitive "
              f"to this cap and to beta (raw {w['raw_beta']:.2f}, adjusted {w['beta']:.2f}). Read the reverse DCF, not the margin of safety.")
        if v.get("comps"):
            a(f"- Comparable companies: {json.dumps({k: round(x, 2) for k, x in v['comps'].items()})}")
        sn = v.get("sensitivity", {})
        if sn.get("grid"):
            a("\n### Sensitivity: value/share (rows WACC, columns terminal growth)\n")
            a("| WACC \\ g | " + " | ".join(pct(t) for t in sn["terminal_growth"]) + " |\n|---" * 1 + "|---" * len(sn["terminal_growth"]) + "|")
            for w_, row in zip(sn["wacc"], sn["grid"]):
                a(f"| {pct(w_,1)} | " + " | ".join(money(x) for x in row) + " |")
        a("")
    f = r.get("forensics", {})
    a("## Financial quality and forensics\n")
    if "error" in f:
        a(f"Unavailable: {f['error']}\n")
    else:
        a(f"**Risk level: {f['risk_level']}** (signal score {f['score']}). {f['disclaimer']}\n")
        for wn in f["warnings"]:
            a(f"- [{wn['severity']}] **{wn['signal']}** - {wn['evidence']}")
        if not f["warnings"]:
            a("- No warning signals triggered.")
        fa = f["facts"]
        cc = fa.get('cash_conversion_ocf_to_ni')
        bm = fa.get('beneish') if isinstance(fa.get('beneish'), dict) else {}
        a(f"\nCash conversion (OCF/NI): {'n/a' if cc is None else f'{cc:.2f}'}, Sloan accruals: {pct(fa.get('sloan_accrual_ratio'))}, "
          f"Beneish M: {bm['m_score']:.2f}" if 'm_score' in bm else f"\nCash conversion (OCF/NI): {'n/a' if cc is None else f'{cc:.2f}'}")
    a(f"\nAltman Z'': {r.get('altman')} | Piotroski F: {r.get('piotroski')}/9\n")
    if r.get("industry"):
        a(f"## Peer scores (0-100, vs sector median)\n\n{r['industry']}\n")
    pj = r.get("projection")
    if pj:
        a("## Three-statement projection (base case, $)\n")
        a("| Year | Revenue | EBIT | Net income | FCF | Cash | Balance check |\n|---|---|---|---|---|---|---|")
        for y in pj:
            a(f"| {y['year']} | {money(y['revenue'])} | {money(y['ebit'])} | {money(y['net_income'])} | {money(y['fcf'])} | {money(y['cash'])} | {y['balance_check']:.2f} |")
        a("")
    fl = r.get("filings")
    if fl:
        a("## SEC filings\n")
        if "error" in fl:
            a(f"{fl['error']}\n")
        for x in fl.get("recent_filings", []):
            a(f"- {x['form']} {x['date']} {x['url']}")
        for c in fl.get("financial_changes", []):
            ch = c["change_pct"]
            a(f"- {c['metric']} ({c['period']}): {c['value']:,.0f} vs {c['prior']:,.0f} ({'n/a' if ch is None else f'{ch:+.1f}%'}) [accession {c['accession']}]")
        for sec in fl.get("10k_changes", []):
            a(f"\n### 10-K Item {sec['item']} {sec['name']}: size change {sec['size_change_pct']:+.1f}%")
            for n in sec["new"][:5]:
                a(f"- NEW: {n['text'][:300]}...  \n  _{n['source']}_")
            for n in sec["removed"][:3]:
                a(f"- REMOVED: {n['text'][:300]}...  \n  _{n['source']}_")
        for g in fl.get("governance_flags", []):
            a(f"- Governance flag **{g['flag']}**: ...{g['context'][:250]}...")
    a(__import__("pipeline").render_extras(r))
    a("\n*Educational research output; not investment advice. Model-generated figures depend on Yahoo Finance data and stated assumptions.*")
    return "\n".join(L)


def narrative(report_md, llm):
    prompt = ("You are an equity research analyst. Using ONLY the numbers in the report below, write a 250-word "
              "assessment covering quality, valuation vs market-implied expectations, key risks, and what would "
              "change the view. Do not invent figures. Label interpretations as interpretations.\n\n" + report_md)
    if llm.startswith("ollama:"):
        import requests
        r = requests.post("http://localhost:11434/api/generate", timeout=600,
                          json={"model": llm.split(":", 1)[1], "prompt": prompt, "stream": False})
        return r.json()["response"]
    from utils import client
    return client.models.generate_content(model=llm, contents=prompt).text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", nargs="?")
    ap.add_argument("--peers", action="store_true", help="fetch sector peers (slow, ~25 Yahoo calls)")
    ap.add_argument("--edgar", action="store_true", help="add SEC filings + 10-K change detection")
    ap.add_argument("--ml", action="store_true", help="distress/anomaly/forecast/earnings models (fetches peer statements, cached)")
    ap.add_argument("--competitors", action="store_true")
    ap.add_argument("--events", action="store_true", help="news + filings + earnings timeline, headline sentiment")
    ap.add_argument("--ask", action="append", default=[], help="research question answered from SEC filings with citations (repeatable)")
    ap.add_argument("--thesis", action="store_true", help="tagged investment thesis (reports/TICKER_thesis.md)")
    ap.add_argument("--all", action="store_true", help="peers + edgar + ml + competitors + events + thesis")
    ap.add_argument("--max-peers", type=int, default=25)
    ap.add_argument("--llm", default="", help="e.g. ollama:qwen2.5:7b or a Gemini model name")
    a = ap.parse_args()
    t = data.find_company(a.ticker or input("Company or ticker: "))
    if not t:
        sys.exit("Company not found")
    if a.all:
        a.peers = a.edgar = a.ml = a.competitors = a.events = a.thesis = True
    res = analyze(t, a.peers, a.edgar, a.max_peers, a.ml, a.competitors, a.events, a.ask, a.llm, a.thesis)
    md = render(res)
    if a.llm:
        md += "\n\n## AI assessment (model-generated interpretation)\n\n" + narrative(md, a.llm)
    os.makedirs("reports", exist_ok=True)
    open(f"reports/{t}.md", "w").write(md)
    json.dump(res, open(f"reports/{t}.json", "w"), default=str, indent=1)
    if res.get("thesis"):
        import thesis
        open(f"reports/{t}_thesis.md", "w").write(thesis.render_thesis(res["thesis"], res["metrics"].get("Name") or t))
    print(md)
    print(f"\nSaved reports/{t}.md and reports/{t}.json")
