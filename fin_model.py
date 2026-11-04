"""Statement normalisation + integrated three-statement projection."""
from analysis import get_value

FIELDS = {
    "revenue": ["Total Revenue", "Revenue"],
    "cogs": ["Cost Of Revenue", "Cost of Revenue"],
    "gross_profit": ["Gross Profit"],
    "sga": ["Selling General And Administration", "Selling General And Administrative"],
    "depreciation": ["Depreciation And Amortization", "Depreciation Amortization Depletion",
                     "Depreciation"],
    "ebit": ["EBIT", "Operating Income"],
    "interest_expense": ["Interest Expense"],
    "pretax": ["Pretax Income"],
    "tax": ["Tax Provision"],
    "net_income": ["Net Income"],
    "receivables": ["Accounts Receivable", "Receivables"],
    "inventory": ["Inventory"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "ppe": ["Net PPE", "Property Plant Equipment"],
    "total_assets": ["Total Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "lt_debt": ["Long Term Debt"],
    "total_debt": ["Total Debt"],
    "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
    "equity": ["Stockholders Equity", "Total Equity Gross Minority Interest"],
    "cash": ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"],
    "retained_earnings": ["Retained Earnings"],
    "shares": ["Ordinary Shares Number", "Share Issued"],
    "ocf": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex": ["Capital Expenditure"],
}
_SRC = {"revenue": 1, "cogs": 1, "gross_profit": 1, "sga": 1, "ebit": 1, "interest_expense": 1,
        "pretax": 1, "tax": 1, "net_income": 1, "ocf": 2, "capex": 2, "depreciation": 2}


def normalize(statements, periods=3):
    """yfinance (balance, income, cashflow) -> list of dicts, newest first."""
    bs, inc, cf = statements
    srcs = {0: bs, 1: inc, 2: cf}
    n = min(periods, *(df.shape[1] for df in (bs, inc, cf) if df is not None and not df.empty))
    out = []
    for c in range(n):
        row = {"date": str(bs.columns[c])[:10]}
        for k, names in FIELDS.items():
            row[k] = float(get_value(srcs[_SRC.get(k, 0)], names, c, 0) or 0)
        if not row["gross_profit"] and row["revenue"]:
            row["gross_profit"] = row["revenue"] - row["cogs"]
        if not row["total_debt"]:
            row["total_debt"] = row["lt_debt"]
        if not row["total_liabilities"] and row["total_assets"]:
            row["total_liabilities"] = row["total_assets"] - row["equity"]
        row["capex"] = abs(row["capex"])
        out.append(row)
    return out


def project_three_statements(base, years=5, revenue_growth=0.08, gross_margin=None, sga_pct=None,
                             da_pct=0.04, capex_pct=0.05, tax_rate=0.21, nwc_pct=None,
                             interest_rate=0.05, payout=0.0):
    """Integrated projection. Cash is the balancing item; balance check is returned.

    base: one normalised period dict. Returns list of yearly dicts with IS, BS, CF.
    """
    rev0 = base["revenue"] or 1.0
    gm = gross_margin if gross_margin is not None else (base["gross_profit"] / rev0 or 0.4)
    sga_pct = sga_pct if sga_pct is not None else max(
        (base["gross_profit"] - base["ebit"] - base["depreciation"]) / rev0, 0.0)
    # operating NWC = receivables + inventory + other current assets ex cash - current liabilities
    other_ca = max(base["current_assets"] - base["cash"] - base["receivables"] - base["inventory"], 0)
    nwc0 = base["receivables"] + base["inventory"] + other_ca - base["current_liabilities"]
    nwc_pct = nwc_pct if nwc_pct is not None else nwc0 / rev0
    other_assets = base["total_assets"] - base["cash"] - base["receivables"] - base["inventory"] - other_ca - base["ppe"]
    debt = base["total_debt"]
    other_liab = base["total_liabilities"] - debt - base["current_liabilities"]
    cash, ppe, nwc, equity, rev = base["cash"], base["ppe"], nwc0, base["equity"], rev0
    rows = []
    for y in range(1, years + 1):
        rev *= 1 + revenue_growth
        gp = rev * gm
        da = rev * da_pct
        ebit = gp - rev * sga_pct - da
        interest = debt * interest_rate
        pretax = ebit - interest
        tax = max(pretax, 0) * tax_rate
        ni = pretax - tax
        capex = rev * capex_pct
        new_nwc = rev * nwc_pct
        d_nwc = new_nwc - nwc
        ocf = ni + da - d_nwc
        fcf = ocf - capex
        div = max(ni, 0) * payout
        cash += ocf - capex - div
        ppe += capex - da
        nwc = new_nwc
        equity += ni - div
        # Balance sheet: assets = cash + ppe + NWC-assets-net + other; liabilities = debt + other + equity
        total_assets_net = cash + ppe + nwc + other_assets   # NWC netted (assets - current liabilities)
        liab_equity_net = debt + other_liab + equity
        rows.append({
            "year": y, "revenue": rev, "gross_profit": gp, "ebit": ebit, "interest": interest,
            "net_income": ni, "da": da, "capex": capex, "d_nwc": d_nwc, "ocf": ocf, "fcf": fcf,
            "cash": cash, "ppe": ppe, "nwc": nwc, "debt": debt, "equity": equity,
            "balance_check": total_assets_net - liab_equity_net,
        })
    return rows
