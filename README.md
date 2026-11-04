# Company Analyzer v3

An equity research platform built on yfinance, SEC EDGAR, scikit-learn and (optionally) a local or hosted LLM.
It reports **facts, model outputs and interpretation separately**, and it never issues a Buy/Sell call.
See `RUN.md` for setup and every command.

## What it does
| Area | Modules | Notes |
|---|---|---|
| Valuation | `valuation.py`, `assumptions.py`, `montecarlo.py` | WACC (date-aware risk-free rate, configurable ERP), 10y DCF, reverse DCF, scenarios, sensitivity, DDM, residual income, comps, Monte Carlo DCF with correlated drivers and variance decomposition |
| Forensics | `forensics.py`, `analysis.py` | Altman Z'', Piotroski F, Beneish M, Sloan accruals |
| Filings | `edgar.py` | 10-K/10-Q sections, year-over-year risk/MD&A diff, 8-K classification, Form 4 |
| ML | `ml.py`, `ml2.py` | Walk-forward and unseen-company validation, survival analysis (time to distress), change-point detection, point-in-time filtering, optional XGBoost/CatBoost/SHAP, distress model with train/validation/test split, bootstrap CIs, calibration, ablation, conformal and quantile intervals, anomaly scan, forecaster, earnings-beat model |
| Research Q&A | `rag.py` | BM25 + TF-IDF hybrid retrieval, optional cross-encoder rerank, table-aware statement rows, transcript ingestion, multi-document `explain()` (reported facts vs interpretation), source reliability, claim-level verification (value, unit, period, context), contradiction flags |
| Portfolio | `portfolio.py`, `quant_cli.py` | Mean-variance, min-vol, max-Sharpe, risk parity, efficient frontier, VaR/CVaR, drawdown, factor exposure, Black-Litterman, Monte Carlo, regime-aware, cost-aware rebalancing, walk-forward comparison |
| Event study | `event_study.py` | Market-model abnormal returns, CAR, volatility ratio, post-event drift, by category, chronological validation |
| Macro | `macro.py` | FRED/Yahoo feeds (rates, CPI, GDP, FX, oil, copper, steel PPI, sector ETFs), shock simulation, replay of the last 12 months of observed moves |
| Guidance | `guidance.py` | SEC earnings-release download, guidance extraction, raise/lower/maintain events for the event study |
| Corporate graph, screener, thesis | `tracker.py`, `discovery.py`, `competitors.py`, `events.py`, `thesis.py` | |
| Interfaces | `analyze.py` (CLI), `api.py` (FastAPI), `dashboard.py` (Streamlit) | |

## Quick start
```
pip install -r requirements.txt
python analyze.py NVDA --all --llm ollama:qwen2.5:7b
python quant_cli.py portfolio AAPL MSFT NVDA JPM --years 3
python quant_cli.py events NVDA
python quant_cli.py macro F --industry auto --shock steel=0.15 --shock rates_bp=100
```
Set `SEC_USER_AGENT="Your Name you@example.com"` for EDGAR. Override rates with `COMPANY_ANALYZER_RF` / `COMPANY_ANALYZER_ERP` or `assumptions.json`.

## Validation status (be skeptical)
- All tests (`test_*.py`) run offline on synthetic data. Valuation, forensics and projections have been run on live NVDA data; the ML, EDGAR, competitor, event, portfolio and RAG modules have **not** yet been run on live data.
- Yahoo provides about four annual statements, so the distress panel is small; with fewer than three fiscal years the model is labelled exploratory and no model selection is done.
- On synthetic data, max-Sharpe optimisation did worse out-of-sample than equal weight. Treat optimiser output as descriptive and check `walk_forward_oos`.
- Monte Carlo and macro results depend on stated volatility, correlation and cost-share assumptions, which are illustrative defaults.
- RAG claim verification catches mismatched numbers, units and periods; it cannot prove semantic truth.
- SEC XBRL history (`edgar.periods_from_xbrl`) now feeds the ML panels (up to ~10+ fiscal years instead of 4) when available. It has never been run against live SEC data. Delete `cache/panel_*.pkl` so old 4-year panels are rebuilt.
- Survival and change-point results need long histories; on short series they say so instead of guessing.
- Newly added and untested on live data: FRED/yfinance macro feeds (`quant_cli.py snapshot`, `macro TICKER --live`), SEC earnings-release download and guidance-change detection (`guidance.py`), embedding search (sentence-transformers if installed, else offline LSA).
- Guidance extraction is regex-based: it needs sentences like "expects revenue of $X to $Y billion" and will miss unusual phrasing. Always verify against the release.
- Full earnings-call transcripts are not auto-downloaded (no free bulk source); the 8-K press release outlook is used, and `add_transcript` accepts text you supply.
- Not tested here: XGBoost/CatBoost/SHAP and neural embeddings (optional installs). Cost-share assumptions in `macro.py` are illustrative until you replace them.

Educational research tool. Not investment advice.
