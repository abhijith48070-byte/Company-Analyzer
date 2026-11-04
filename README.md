> **v3:** see RUN.md for the full command list (valuation, forensics, SEC filings, ML, RAG, tracker, thesis, API, dashboard).

# Company Analyzer

An AI-powered equity research platform that automates company analysis using financial data, valuation models, AI-generated reports, and LangGraph-based workflow orchestration.

## Features

- Company search using name or ticker
- Automatic financial metric extraction
- Discounted Cash Flow (DCF) valuation
- Industry benchmarking
- Risk analysis
  - Piotroski F-Score
  - Altman Z-Score
- Latest company news retrieval
- AI-generated equity research report using Gemini
- AI-powered report verification (grounding check)
- Automatic PDF report generation
- Financial visualization charts
- Multi-agent workflow powered by LangGraph

---

## Architecture

```
                User
                  │
                  ▼
             Company Search
                  │
                  ▼
           LangGraph Workflow
                  │
      ┌───────────┼───────────┐
      │           │           │
      ▼           ▼           ▼
 Financial     Industry     Risk
 Analysis      Analysis    Analysis
      │           │           │
      └──────┬────┴────┬──────┘
             ▼
      News Collection
             ▼
      AI Report Generator
             ▼
     Grounding Evaluation
             ▼
       Graph Generation
             ▼
         PDF Report
```

---

## Technologies Used

- Python
- LangGraph
- Google Gemini API
- Yahoo Finance (yfinance)
- Pandas
- Matplotlib
- ReportLab
- NumPy

---

## Project Structure

```
Company-Analyzer/
│
├── main.py              # Entry point
├── graph.py             # LangGraph workflow
├── analysis.py          # Financial analysis algorithms
├── data.py              # Data collection
├── report.py            # AI report & PDF generation
├── utils.py             # Gemini configuration
├── requirements.txt
└── README.md
```

---

## Workflow

1. User enters a company name or ticker.
2. Company information is fetched from Yahoo Finance.
3. LangGraph executes multiple analysis agents in parallel:
   - DCF Valuation
   - Industry Analysis
   - Risk Analysis
   - News Collection
4. Gemini generates a comprehensive equity research report.
5. Gemini evaluates the generated report for factual consistency.
6. Financial charts are created.
7. A professional PDF report is generated.

---

## Example Output

The generated report includes:

- Executive Summary
- Company Overview
- Financial Performance
- DCF Valuation
- Industry Comparison
- Risk Assessment
- Recent News
- Investment Recommendation
- AI Grounding Evaluation

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/Company-Analyzer.git
```

Move into the project

```bash
cd Company-Analyzer
```

Install dependencies

```bash
pip install -r requirements.txt
```

Configure your Gemini API key inside `utils.py`.

Run the project

```bash
python main.py
```

---

## Example

```
Enter Company or Ticker:

Apple

↓

Financial Analysis
↓

DCF Valuation

↓

Industry Analysis

↓

Risk Analysis

↓

News Analysis

↓

AI Generated Equity Research Report

↓

Grounding Verification

↓

PDF Report Generated
```

---

## Future Improvements

- Monte Carlo DCF simulation
- Portfolio analysis
- SEC filing analysis
- Earnings call transcript summarization
- RAG over financial statements
- Multi-model LLM support
- Streamlit/Web dashboard
- Real-time market monitoring
- Vector database for historical reports

---

## Skills Demonstrated

- Multi-Agent AI Systems
- LangGraph Workflow Design
- Financial Modeling
- Equity Research
- LLM Prompt Engineering
- Data Engineering
- AI Report Verification
- Python Development
- API Integration
- Data Visualization

---

## License

MIT License
