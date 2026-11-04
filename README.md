Company Analyzer

AI-Powered Equity Research & Financial Intelligence Platform

Company Analyzer is an integrated financial research platform that combines quantitative valuation, financial statement analysis, machine learning, regulatory filing intelligence and AI-powered financial research.

It transforms fragmented financial data into structured company analysis through fundamental modelling, predictive analytics, forensic accounting, portfolio optimization and evidence-grounded research.

The platform is designed to help users investigate companies, evaluate financial health, understand market expectations, identify potential risks and develop investment theses backed by financial evidence.

Core capabilities: Quantitative Finance · Financial ML · SEC EDGAR · Financial Forensics · RAG · Portfolio Optimization · Event Studies

⸻

Key Capabilities

1. Quantitative Valuation Engine

A multi-method valuation framework for estimating company value and analysing market expectations.

* Discounted Cash Flow (DCF) valuation.
* Reverse DCF and market-implied growth analysis.
* Weighted Average Cost of Capital (WACC).
* Capital Asset Pricing Model (CAPM).
* Dividend Discount Model (DDM).
* Residual Income Model.
* Comparable Company Analysis.
* Sum-of-the-Parts (SOTP) valuation.
* Three-statement financial projections.
* Bull, base and bear scenarios.
* Sensitivity analysis.
* Monte Carlo DCF with correlated financial assumptions.
* Valuation uncertainty and variance decomposition.

2. Financial Forensics Engine

Evaluates financial statement quality and identifies potential accounting and financial warning signals.

Model / Indicator	Purpose
Altman Z-Score	Financial distress assessment
Piotroski F-Score	Financial strength assessment
Beneish M-Score	Earnings manipulation risk indicators
Sloan Accrual Ratio	Earnings quality analysis
Cash Conversion	Earnings-to-cash-flow assessment
Financial Divergence Analysis	Identifying unusual movements across financial metrics
Governance Flags	Highlighting selected corporate governance risks

The engine provides supporting financial indicators rather than treating individual scores as proof of misconduct.

3. SEC Filing Intelligence

Integrates the SEC EDGAR ecosystem to retrieve and analyse corporate disclosures.

Supported filing categories include:

* 10-K — Annual reports.
* 10-Q — Quarterly reports.
* 8-K — Material corporate events.
* Form 4 — Insider transactions.
* DEF 14A — Proxy statements, where supported by available data.
* 13F — Institutional holdings, where available.

Capabilities:

* Filing retrieval and section extraction.
* XBRL financial statement extraction.
* Historical financial period reconstruction.
* Year-over-year MD&A and risk-factor comparisons.
* Material filing change detection.
* 8-K event classification.
* Insider transaction analysis.
* Governance-related indicators.

4. Financial Machine Learning

A collection of predictive and analytical models for investigating financial performance and corporate risk.

Financial prediction

* Financial distress classification.
* Revenue forecasting.
* Operating margin forecasting.
* Earnings surprise prediction.

Unsupervised learning

* Isolation Forest anomaly detection.
* Financial statement anomaly screening.
* Structural change-point detection.

Advanced modelling

* Logistic Regression.
* Random Forest.
* Histogram-based Gradient Boosting.
* LightGBM.
* Optional XGBoost and CatBoost.
* Bayesian Ridge Regression.
* Survival analysis for time-to-distress modelling.
* Quantile and conformal prediction intervals.

Evaluation methodology

* Chronological train/validation/test separation.
* Walk-forward evaluation.
* Unseen-company validation.
* Bootstrap confidence intervals.
* Calibration analysis.
* Feature ablation.
* Point-in-time financial data filtering.

5. Financial Research Assistant

An evidence-grounded financial research system combining document retrieval, numerical analysis and language models.

Features:

* Hybrid BM25 and TF-IDF retrieval.
* Optional embedding-based retrieval.
* Optional cross-encoder reranking.
* Financial statement table retrieval.
* Earnings release and transcript ingestion.
* Multi-document research.
* Source reliability indicators.
* Numerical claim verification.
* Entity, period and unit consistency checks.
* Contradiction flags.
* Separation of reported facts from generated interpretation.

Users can investigate questions such as:

* What changed in the company’s latest annual filing?
* Why did operating margins decline?
* What are management’s expectations for future revenue?
* What are the company’s biggest financial risks?
* How has its debt position evolved?
* What evidence supports a particular financial claim?

Gemini and local LLM integrations are optional.

6. FinClaim — Financial Claim Verification

A dedicated research component for evaluating the reliability of financial claims against supporting evidence.

Classification categories:

* SUPPORTED
* REFUTED
* NOT_ENOUGH_INFO

The verification system investigates:

* Numerical discrepancies.
* Incorrect reporting periods.
* Unit and scale mismatches.
* Incorrect direction of financial changes.
* Company and entity mismatches.
* Missing or contradictory evidence.

See FINCLAIM.md for the benchmark methodology and verification details.

7. Portfolio Optimization Engine

Quantitative portfolio construction and risk analysis.

Implemented capabilities include:

* Mean-variance optimization.
* Minimum-volatility portfolios.
* Maximum Sharpe ratio portfolios.
* Risk-parity allocation.
* Efficient frontier generation.
* Portfolio correlation analysis.
* Value at Risk (VaR).
* Conditional Value at Risk (CVaR).
* Drawdown analysis.
* Factor exposure analysis.
* Black-Litterman modelling.
* Monte Carlo portfolio simulation.
* Regime-aware allocation.
* Transaction-cost-aware rebalancing.
* Walk-forward out-of-sample comparison.

8. Earnings Event Study Engine

Analyses market reactions around corporate announcements and financial events.

Features:

* Market-model abnormal returns.
* Cumulative abnormal returns (CAR).
* Event-window analysis.
* Volatility ratio analysis.
* Post-event drift investigation.
* Event-category comparisons.
* Chronological evaluation.

The engine also incorporates classified corporate events and earnings guidance changes where available.

9. Macroeconomic Intelligence

Connects company-level research with broader economic conditions.

Supported indicators include:

* Interest rates.
* Consumer Price Index (CPI).
* GDP.
* Foreign exchange movements.
* Oil prices.
* Copper prices.
* Steel producer price indicators.
* Sector ETF performance.

Additional capabilities:

* Company and industry mapping.
* Macroeconomic shock simulation.
* Historical market-move replay.
* Financial sensitivity analysis.

Macroeconomic scenario outputs depend on the assumptions specified by the user.

10. Corporate Intelligence

Investigates relationships and developments across companies.

Company discovery

* Public company screening.
* Private company discovery.
* Financial and sector filters.

Corporate tracking

* Mergers and acquisitions.
* Strategic investments.
* Institutional holdings.
* Subsidiaries.
* Parent-company relationships.
* Corporate ownership graphs.

Competitive intelligence

* Peer comparisons.
* Financial performance comparisons.
* Sector positioning.
* Competitive risk indicators.

11. Investment Thesis Generator

Combines company-level financial research into a structured investment analysis.

The generated research includes:

* Business overview.
* Financial health.
* Revenue and profitability trends.
* Valuation analysis.
* Competitive positioning.
* Growth drivers.
* Financial and operational risks.
* Potential catalysts.
* Bull, base and bear scenarios.
* Supporting evidence and assumptions.

The platform presents analytical findings rather than issuing automatic Buy/Sell recommendations.

⸻

System Architecture

                 COMPANY ANALYZER
                         |
                  DATA INGESTION
                         |
          +--------------+--------------+
          |              |              |
       FINANCIAL       SEC EDGAR       NEWS &
         DATA           FILINGS       EVENTS
          |              |              |
          +--------------+--------------+
                         |
                  DATA PROCESSING
                         |
          +--------------+--------------+
          |              |              |
       FINANCIAL       FINANCIAL       NLP /
       MODELLING       FORENSICS      RETRIEVAL
          |              |              |
          +--------------+--------------+
                         |
                MACHINE LEARNING
                         |
          +--------------+--------------+
          |              |              |
       FORECASTING    ANOMALIES      RAG / Q&A
          |              |              |
          +--------------+--------------+
                         |
                QUANTITATIVE ENGINE
                         |
          +--------------+--------------+
          |              |              |
       VALUATION      PORTFOLIO      EVENT STUDY
          |              |              |
          +--------------+--------------+
                         |
                RESEARCH SYNTHESIS
                         |
                INVESTMENT THESIS
                         |
          +--------------+--------------+
          |              |              |
       STREAMLIT       FASTAPI       CLI REPORTS
       DASHBOARD        REST API

⸻

Technology Stack

Category	Technologies
Language	Python
Financial Data	yfinance, OpenF1-style financial data workflows, SEC EDGAR, XBRL
Data Processing	Pandas, NumPy, SciPy
Machine Learning	Scikit-learn, LightGBM
Advanced ML	XGBoost, CatBoost, SHAP (optional)
NLP / Retrieval	BM25, TF-IDF, Sentence Transformers (optional)
LLM Integration	Gemini, Ollama
Quantitative Finance	CAPM, WACC, DCF, Black-Litterman, Monte Carlo
Backend	FastAPI
Dashboard	Streamlit
Testing	Python unittest / offline synthetic-data tests

⸻

Getting Started

Requirements

* Python 3.10 or later.
* pip.
* Internet connection for external financial data.
* Optional Gemini API key for Gemini-powered research.
* Optional Ollama installation for local LLM inference.

Installation

Clone the repository:

git clone https://github.com/abhijith48070-byte/Company-Analyzer.git
cd Company-Analyzer

Create a virtual environment:

python -m venv .venv

Activate it:

macOS / Linux

source .venv/bin/activate

Windows

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Environment Configuration

For SEC EDGAR access, configure a descriptive contact:

export SEC_USER_AGENT="Your Name your@email.com"

For optional Gemini integration, create a .env file:

GEMINI_API_KEY=your_api_key

Never commit API keys or other secrets to GitHub.

⸻

Running the Application

Command-Line Analysis

Basic company analysis:

python analyze.py NVDA

Include peer and competitive analysis:

python analyze.py NVDA --peers --competitors

Include SEC filing analysis:

python analyze.py NVDA --edgar

Include machine learning analysis:

python analyze.py NVDA --ml

Include corporate events:

python analyze.py NVDA --events

Generate an investment thesis:

python analyze.py NVDA --thesis

Run the complete analysis with a local LLM:

python analyze.py NVDA --all --llm ollama:qwen2.5:7b

Portfolio Analysis

python quant_cli.py portfolio AAPL MSFT NVDA JPM --years 3

Event Analysis

python quant_cli.py events NVDA

Macroeconomic Analysis

python quant_cli.py macro F --industry auto --shock steel=0.15 --shock rates_bp=100

Launch the Dashboard

streamlit run dashboard.py

The dashboard provides 12 analytical tabs, including company analysis, financial intelligence, valuation, machine learning, research, portfolio analysis and investment thesis generation.

Launch the REST API

uvicorn api:app --reload

API documentation:

http://127.0.0.1:8000/docs

For the complete command reference, see RUN.md.

⸻

Testing

Run the offline test suites:

python test_pipeline.py
python test_ml.py
python test_ml2.py
python test_rag_tracker.py
python test_modules.py
python test_integration.py
python test_new.py
python test_rag2.py
python test_xbrl.py
python test_guidance_macro.py

These tests primarily use synthetic or controlled data to verify module behaviour.

Passing offline tests does not establish the accuracy of financial predictions or the correctness of live external integrations.

⸻

Research Integrity & Limitations

Company Analyzer is an experimental financial research platform. Its outputs should be interpreted with an understanding of the underlying data and modelling assumptions.

Current limitations include:

* Most automated tests use synthetic data rather than live market conditions.
* Some financial ML models have limited historical observations.
* Financial distress predictions are exploratory when insufficient fiscal history is available.
* Portfolio optimization can underperform simple benchmarks, including equal-weight portfolios.
* Monte Carlo valuations depend on assumptions about growth, volatility and correlations.
* Macroeconomic shock simulations use simplified exposure assumptions.
* RAG verification can identify numerical and contextual inconsistencies but cannot establish semantic truth in every case.
* SEC XBRL extraction and some newly added external integrations require further live-data validation.
* Guidance extraction uses pattern-based methods and requires verification against original disclosures.
* Full earnings-call transcripts must be supplied when no supported free source is available.

Research outputs should distinguish observed financial facts, statistical predictions, model assumptions and generated interpretations.

⸻

Project Objectives

Company Analyzer explores the intersection of:

* Quantitative finance.
* Financial machine learning.
* Probabilistic forecasting.
* Fundamental equity research.
* Financial forensics.
* Natural language processing.
* Retrieval-augmented generation.
* Portfolio optimization.
* Financial data engineering.

The broader objective is to investigate how computational methods can make financial research more systematic, reproducible and evidence-driven.

⸻

Disclaimer

Company Analyzer is an educational and experimental research project.

It is not a registered investment adviser, broker or financial institution. Its outputs are not investment recommendations. Financial data may be incomplete, delayed or inaccurate, and model-generated estimates depend on assumptions that may not hold in real markets.

Always independently verify financial information against original sources.

⸻

Repository: abhijith48070-byte/Company-Analyzer
