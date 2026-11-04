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

## v3.1 additions
- `python analyze.py TICKER` now includes a Monte Carlo DCF section.
- `python quant_cli.py portfolio|events|macro ...` (see README).
- Distress model: train < validation < test years; reports bootstrap CI, calibration error, ablation and conformal growth intervals (needs >= 3 fiscal years of panel).
- Risk-free rate: env `COMPANY_ANALYZER_RF`, else `assumptions.json`, else live ^TNX, else flagged default.
- Tests: `python test_new.py`, `python test_rag2.py` (plus the earlier ones).

## v3.2 additions
- `ml2.py`: walk-forward, grouped (unseen-company) validation, survival, change points; shown under `--ml`.
- `edgar.periods_from_xbrl`: many-year statements from SEC; `pipeline.fetch_periods` uses it automatically. Delete `cache/panel_*.pkl` once.
- `rag.FilingIndex`: `add_statements(periods)`, `add_transcript(title, text, date)`, `explain(question, llm)`, `make_reranker()`.
- API: `POST /portfolio`, `POST /macro`, `GET /events/{ticker}`. Dashboard: new "Quant Lab" tab.
- `quant_cli.py events` also tests 8-K items (acquisitions, management change, material agreements) and keyword-classified news.
- Tests: `test_ml2.py`, `test_xbrl.py`.

## v3.3 additions
- `python quant_cli.py snapshot [--sectors]` macro dashboard; `python quant_cli.py macro TICKER --industry auto --live` replays the past year's observed moves.
- `guidance.py`: guidance changes appear in `analyze.py --all` reports and feed `quant_cli.py events` (guidance_raise / guidance_lower).
- RAG now fuses TF-IDF, BM25 and embeddings; earnings releases are indexed as kind `release`.
- Test: `python test_guidance_macro.py`.
