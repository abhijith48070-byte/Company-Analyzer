"""Date-aware, configurable valuation inputs (risk-free rate, equity risk premium).
Priority: environment variables > assumptions.json > live 10y Treasury (Yahoo ^TNX, cached per day) > documented defaults.
Every result carries its source and as-of date so reports show where the numbers came from."""
import json
import os
from datetime import date

DEFAULTS = {"rf": 0.043, "erp": 0.05}
PATH = "assumptions.json"
CACHE = "cache/assumptions_live.json"


def _live_rf():
    today = str(date.today())
    try:
        c = json.load(open(CACHE))
        if c.get("as_of") == today:
            return c["rf"]
    except Exception:
        pass
    try:
        import yfinance as yf
        px = yf.Ticker("^TNX").history(period="5d")["Close"].dropna()
        rf = float(px.iloc[-1]) / 100.0
        if not 0.0 < rf < 0.20:
            return None
        os.makedirs("cache", exist_ok=True)
        json.dump({"rf": rf, "as_of": today}, open(CACHE, "w"))
        return rf
    except Exception:
        return None


def get_assumptions():
    """-> {'rf','erp','rf_source','erp_source','as_of'}"""
    out = {"as_of": str(date.today())}
    file_vals = {}
    if os.path.exists(PATH):
        try:
            file_vals = json.load(open(PATH))
        except Exception:
            pass
    for key, env in (("rf", "COMPANY_ANALYZER_RF"), ("erp", "COMPANY_ANALYZER_ERP")):
        if os.getenv(env):
            out[key], out[f"{key}_source"] = float(os.environ[env]), f"environment {env}"
        elif key in file_vals:
            out[key], out[f"{key}_source"] = float(file_vals[key]), f"{PATH} (as of {file_vals.get('as_of', 'unknown date')})"
        elif key == "rf" and (live := _live_rf()) is not None:
            out[key], out[f"{key}_source"] = live, "live 10y Treasury yield (Yahoo ^TNX)"
        else:
            out[key], out[f"{key}_source"] = DEFAULTS[key], "built-in default (NOT date-aware; set it in assumptions.json)"
    return out
