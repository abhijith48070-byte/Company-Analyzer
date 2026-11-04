# FinClaim: does a financial-report grounding check actually catch errors?

Claim-level benchmark for the Company Analyzer grounding agent. Each claim about a company's financials is labelled SUPPORTED, REFUTED or NOT_ENOUGH_INFO against a structured evidence table (yfinance statements, USD millions).
REFUTED claims carry an error type (wrong_number, wrong_period, unit_scale, wrong_direction, wrong_entity). Splits are by company.

## Run (from the Company-Analyzer root; copy the `finclaim/` folder there)
```bash
python -m finclaim.tests                                   # offline self-test
python -m finclaim.run build --tickers AAPL MSFT NVDA ... --n 30     # use 40+ S&P 500 tickers; needs internet
python -m finclaim.run eval --systems majority tool                   # free, no API
python -m finclaim.run eval --systems judge closedbook --split test --limit 300   # Gemini; results cached in judge_cache.jsonl
python -m finclaim.run labelsheet --n 100                  # fill the last column by hand from the evidence only
python -m finclaim.run agreement --sheet labelsheet.csv    # Cohen's kappa of your labels vs the generated labels
```
## Read the results honestly
- `tool` recomputes from the evidence with a parser tied to the templates, so it scores ~100% by construction. It checks the labels and the pipeline; it is NOT a result.
- `closedbook` (no evidence) vs `judge` (with evidence) shows how much the judge relies on the evidence vs memory. Report accuracy with the company-bootstrap interval and REFUTED recall per error type.
- Claims are template-generated; natural-language claims written from filings must be labelled by hand (use labelsheet) before you claim anything about them.
- Not included: fine-tuning a small verifier, filing-text evidence, natural-language claims.
