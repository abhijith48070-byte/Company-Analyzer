"""Dashboard.  streamlit run dashboard.py"""
import os
import tempfile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import analyze as az
import data
import discovery
import thesis as th
from tracker import Tracker

st.set_page_config(page_title="Company Analyzer", layout="wide")
st.title("Company Analyzer")
TRACKER = Tracker("tracker.db")
PRIVATE = discovery.PrivateCompanies("private_companies.json")

with st.sidebar:
    q = st.text_input("Company or ticker", "NVDA")
    peers = st.checkbox("Sector peers (slow)", True)
    use_edgar = st.checkbox("SEC EDGAR (set SEC_USER_AGENT)", False)
    use_ml = st.checkbox("ML models", False)
    use_events = st.checkbox("News/event timeline", True)
    llm = st.text_input("LLM for answers (optional)", "", help="e.g. ollama:qwen2.5:7b")
    go = st.button("Analyze", type="primary")
    with st.expander("Screener"):
        flt = st.text_input("Filters", "growth>0.2 and fcf>0 and pe<40",
                            help="fields: growth margin roe fcf pe fwd_pe debt_equity mcap ev_ebitda pb sector industry")
        if st.button("Run screen"):
            sc = discovery.Screener().load(data.SP500["Symbol"].tolist()[:120])
            st.dataframe(sc.screen(flt))

if go:
    t = data.find_company(q) if not q.isupper() else q
    with st.spinner(f"Analyzing {t}..."):
        st.session_state["res"] = az.analyze(t, peers=peers, edgar=use_edgar, do_ml=use_ml, do_comp=peers,
                                             do_events=use_events, do_thesis=True, llm=llm)
        st.session_state["ticker"], st.session_state["llm"] = t, llm
res = st.session_state.get("res")
if not res:
    st.info("Enter a ticker in the sidebar and press Analyze.")
    st.stop()

m, v, t = res["metrics"], res.get("valuation") or {}, st.session_state["ticker"]
tabs = st.tabs(["Overview", "Financials", "Valuation", "Forensics", "Filings", "AI Research", "ML Intelligence",
                "Competitors", "Corporate Graph", "Transactions", "News", "Investment Thesis"])

with tabs[0]:
    st.subheader(f"{m.get('Name')} ({t})")
    c = st.columns(4)
    c[0].metric("Price", m.get("Currentprice")); c[1].metric("Market cap", az.money(m.get("MarketCap")))
    c[2].metric("Revenue growth (qtr YoY)", az.pct(m.get("Revenuegrowth"))); c[3].metric("FCF", az.money(m.get("Fcf")))
    st.write(m.get("Summary"))
    if res.get("industry"):
        st.json(res["industry"])

with tabs[1]:
    ps = res.get("periods") or []
    if ps:
        df = pd.DataFrame(ps).set_index("date").T
        st.dataframe(df.style.format("{:,.0f}"))
    pj = res.get("projection")
    if pj:
        st.caption("Three-statement projection (balances by construction)")
        st.dataframe(pd.DataFrame(pj).set_index("year"))
        st.line_chart(pd.DataFrame(pj).set_index("year")[["revenue", "ebit", "net_income", "fcf"]])

with tabs[2]:
    if v:
        w = v["wacc"]
        st.write(f"**WACC {w['wacc']:.2%}** (cost of equity {w['cost_of_equity']:.2%}, adjusted beta {w['beta']:.2f}, raw {w['raw_beta']:.2f})")
        sc = v.get("scenarios", {})
        if "base" in sc:
            st.dataframe(pd.DataFrame(sc).T)
        rd = v.get("reverse_dcf", {})
        st.write("**Reverse DCF**", {k: rd.get(k) for k in ("implied_fcf_growth", "implied_terminal_growth", "implied_fcf_margin", "current_fcf_margin")})
        sn = v.get("sensitivity", {})
        if sn.get("grid"):
            st.dataframe(pd.DataFrame(sn["grid"], index=[f"{x:.1%}" for x in sn["wacc"]], columns=[f"{x:.1%}" for x in sn["terminal_growth"]]))
        st.write({k: v.get(k) for k in ("ddm", "ddm_note", "residual_income", "comps", "fair_value_range")})

with tabs[3]:
    f = res.get("forensics", {})
    st.write(f"**Risk level: {f.get('risk_level')}**  ·  Altman Z'' {res.get('altman')}  ·  Piotroski {res.get('piotroski')}/9")
    for wn in f.get("warnings", []):
        st.warning(f"**{wn['signal']}**: {wn['evidence']}")
    st.caption(f.get("disclaimer", ""))

with tabs[4]:
    fl = res.get("filings")
    if not fl:
        st.info("Enable SEC EDGAR in the sidebar.")
    else:
        st.write(fl.get("error", ""))
        st.dataframe(pd.DataFrame(fl.get("recent_filings", [])))
        st.dataframe(pd.DataFrame(fl.get("financial_changes", [])))
        for sec in fl.get("10k_changes", []):
            st.markdown(f"**Item {sec['item']} {sec['name']}** ({sec['size_change_pct']:+.1f}% size)")
            for n in sec["new"][:6]:
                st.markdown(f"- NEW: {n['text'][:350]}  \n  _{n['source']}_")
            for n in sec["removed"][:3]:
                st.markdown(f"- REMOVED: {n['text'][:350]}  \n  _{n['source']}_")

with tabs[5]:
    qn = st.text_input("Ask about this company's filings", "What are the biggest risks?")
    if st.button("Answer") and data.get_cik(t):
        import pipeline
        from rag import make_llm
        with st.spinner("Indexing filings (first time only)..."):
            idx = pipeline.build_rag(t, data.get_cik(t), data.News(data.get_company_info(t)[0]))
        a = idx.answer(qn, llm=make_llm(st.session_state.get("llm")))
        st.write(a["answer"])
        for s in a["sources"]:
            st.caption(f"[{s['n']}] {s['citation']}")
        if a["unsupported_numbers"]:
            st.error(f"Numbers not found in sources: {a['unsupported_numbers']}")

with tabs[6]:
    mlr = res.get("ml")
    st.json(mlr) if mlr else st.info("Enable ML models in the sidebar.")

with tabs[7]:
    c = res.get("competitors")
    if c:
        st.dataframe(pd.DataFrame(c["table"]))
        st.write(c["percentile_vs_sector"]); [st.warning(f"{x['ticker']}: {x['evidence']}") for x in c["threats"]]
    else:
        st.info("Enable sector peers.")

with tabs[8]:
    p = os.path.join(tempfile.gettempdir(), "graph.html")
    if t in TRACKER.graph():
        TRACKER.graph_html(p, t); components.html(open(p).read(), height=600)
    else:
        st.info("No relationships recorded for this company yet. Add transactions in the next tab.")

with tabs[9]:
    st.dataframe(pd.DataFrame(TRACKER.transactions(t)))
    st.dataframe(pd.DataFrame(TRACKER.investments(t)))
    with st.form("tx"):
        k = st.selectbox("Kind", ["acquisition", "merger", "divestiture", "spin-off", "ipo", "strategic investment"])
        a_, b_ = st.text_input("Acquirer", t), st.text_input("Target")
        d_, val, src = st.text_input("Date"), st.number_input("Value (USD)", 0.0), st.text_input("Source (filing/URL)")
        if st.form_submit_button("Add") and b_:
            TRACKER.add_transaction(k, a_, b_, d_, val or None, source=src, verified=bool(src)); st.rerun()
    st.subheader("Private companies")
    st.json(PRIVATE.search(""))

with tabs[10]:
    e = res.get("events")
    if e:
        st.dataframe(pd.DataFrame(e["events"])); st.caption(e["caveat"])
    else:
        st.info("Enable the event timeline.")

with tabs[11]:
    if res.get("thesis"):
        st.markdown(th.render_thesis(res["thesis"], m.get("Name") or t))
