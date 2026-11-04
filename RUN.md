# Company Analyzer v3 - how to run

Setup:  pip install -r requirements.txt   (copy your .env with GEMINI_API_KEY only if you use Gemini)

## Command line
    python analyze.py NVDA                          valuation, reverse DCF, forensics, 3-statement projection
    python analyze.py NVDA --peers --competitors    + sector scores, comps, competitive table, ROIC, threats
    python analyze.py NVDA --edgar                  + SEC filings, XBRL changes, 10-K vs prior 10-K diff, governance flags
    python analyze.py NVDA --ml                     + distress, anomaly, growth/margin forecast, earnings-beat models
    python analyze.py NVDA --events                 + news/filing/earnings timeline and headline sentiment
    python analyze.py NVDA --ask "What are the biggest risks?" --ask "What changed in the latest 10-K?"
    python analyze.py NVDA --thesis                 + tagged thesis in reports/NVDA_thesis.md
    python analyze.py NVDA --all --llm ollama:qwen2.5:7b     everything
Set SEC_USER_AGENT="Your Name your@email" before using --edgar / --ask (SEC requires a contact).
Outputs: reports/TICKER.md, .json, _thesis.md.  First --ml run caches peer statements in cache/ (30 days).

## Dashboard and API
    streamlit run dashboard.py          12 tabs: Overview ... Investment Thesis, plus screener
    uvicorn api:app --reload            REST API, docs at /docs

## Modules
valuation.py (WACC, DCF, reverse DCF, scenarios, sensitivity, DDM, RIM, comps, SOTP) · forensics.py (Beneish, Sloan, cash
conversion, divergences, governance flags) · edgar.py (filings, XBRL, 10-K diff, 8-K, Form 4) · ml.py (distress, anomaly,
forecasts, earnings, sentiment, news impact) · rag.py (cited answers + number check) · tracker.py (M&A, investments, graph,
13F, Exhibit 21) · discovery.py (screener, private companies) · competitors.py · events.py · thesis.py

## Tests (offline, synthetic data)
    python test_pipeline.py && python test_ml.py && python test_rag_tracker.py && python test_modules.py && python test_integration.py
