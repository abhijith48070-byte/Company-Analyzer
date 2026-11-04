"""Valuation engine: CAPM/WACC, DCF, reverse DCF, scenarios, sensitivity, DDM, RIM, comps, SOTP."""
from scipy.optimize import brentq

RISK_FREE = 0.043      # override via argument; ideally the live 10y Treasury yield
ERP = 0.05             # equity risk premium
TERMINAL_G = 0.025


def compute_wacc(metrics, rf=None, erp=None, pretax_cost_of_debt=None, tax_rate=0.21):
    from assumptions import get_assumptions
    asm = get_assumptions()
    src = {"rf_source": asm["rf_source"], "erp_source": asm["erp_source"], "as_of": asm["as_of"]}
    if rf is not None: src["rf_source"] = "argument"
    if erp is not None: src["erp_source"] = "argument"
    rf = asm["rf"] if rf is None else rf
    erp = asm["erp"] if erp is None else erp
    raw_beta = metrics.get("Beta") or 1.0
    beta = min(max(0.67 * raw_beta + 0.33, 0.3), 2.5)   # Blume-adjusted: shrinks extreme betas toward 1
    ke = rf + beta * erp
    debt = metrics.get("Debt") or 0
    mcap = metrics.get("MarketCap") or 0
    kd = pretax_cost_of_debt if pretax_cost_of_debt is not None else rf + 0.015
    total = debt + mcap
    wd = debt / total if total else 0
    wacc = (1 - wd) * ke + wd * kd * (1 - tax_rate)
    return {"cost_of_equity": ke, "cost_of_debt_after_tax": kd * (1 - tax_rate), "beta": beta, "raw_beta": raw_beta,
            "weight_debt": wd, "weight_equity": 1 - wd, "rf": rf, "erp": erp, "wacc": wacc, **src}


def project_fcf(fcf0, growth, years=10, terminal_g=TERMINAL_G):
    """Growth fades linearly from `growth` to `terminal_g` over the horizon."""
    flows, f = [], fcf0
    for i in range(years):
        g = growth + (terminal_g - growth) * i / max(years - 1, 1)
        f *= 1 + g
        flows.append(f)
    return flows


def dcf_value_per_share(fcf0, growth, wacc, metrics, terminal_g=TERMINAL_G, years=10):
    wacc = max(wacc, terminal_g + 0.01)
    flows = project_fcf(fcf0, growth, years, terminal_g)
    pv = sum(f / (1 + wacc) ** (i + 1) for i, f in enumerate(flows))
    tv = flows[-1] * (1 + terminal_g) / (wacc - terminal_g)
    ev = pv + tv / (1 + wacc) ** years
    eq = ev - (metrics.get("Debt") or 0) + (metrics.get("Cash") or 0)
    return eq / metrics["Sharesoutstanding"], ev


def reverse_dcf(metrics, wacc=None, terminal_g=TERMINAL_G, years=10):
    """What does the current price imply? Solves three ways, each holding the others fixed."""
    price, shares = metrics.get("Currentprice"), metrics.get("Sharesoutstanding")
    fcf0, rev0 = metrics.get("Fcf"), metrics.get("Revenue")
    if not (price and shares and fcf0):
        return {"error": "need price, shares and FCF"}
    wacc = wacc or compute_wacc(metrics)["wacc"]
    wacc = max(wacc, terminal_g + 0.01)
    out = {"wacc": wacc, "price": price}
    if fcf0 <= 0:
        out["note"] = "FCF <= 0: growth solve is not meaningful; see implied FCF margin"

    def solve(f, lo, hi):
        try:
            return brentq(f, lo, hi, xtol=1e-6)
        except ValueError:
            return None  # price outside the solvable range

    if fcf0 > 0:
        out["implied_fcf_growth"] = solve(
            lambda g: dcf_value_per_share(fcf0, g, wacc, metrics, terminal_g, years)[0] - price, -0.5, 3.0)
        out["implied_terminal_growth"] = solve(
            lambda t: dcf_value_per_share(fcf0, metrics.get("Revenuegrowth") or 0.05, wacc, metrics,
                                          t, years)[0] - price, -0.05, wacc - 0.005)
    if rev0:
        base_g = max(min(metrics.get("Revenuegrowth") or 0.05, 0.3), -0.1)
        cur_margin = fcf0 / rev0
        # implied steady FCF margin at current growth path
        def f_margin(m):
            fcf_start = rev0 * m
            return dcf_value_per_share(fcf_start, base_g, wacc, metrics, terminal_g, years)[0] - price
        out["implied_fcf_margin"] = solve(f_margin, -0.5, 1.0)
        out["current_fcf_margin"] = cur_margin
    return out


def scenarios(metrics, wacc=None):
    wacc = wacc or compute_wacc(metrics)["wacc"]
    fcf0 = metrics.get("Fcf")
    if not fcf0 or not metrics.get("Sharesoutstanding"):
        return {"error": "insufficient data"}
    if fcf0 <= 0:
        return {"error": "FCF is negative; FCF-based DCF is not meaningful (see reverse DCF implied margin)"}
    g = metrics.get("Revenuegrowth") or 0.05
    g = max(min(g, 0.40), -0.05)
    spec = {"bear": (g * 0.5 - 0.02, wacc + 0.015, 0.015),
            "base": (g, wacc, TERMINAL_G),
            "bull": (min(g * 1.4 + 0.02, 0.50), max(wacc - 0.01, 0.05), 0.03)}
    out = {}
    for name, (gg, w, tg) in spec.items():
        v, _ = dcf_value_per_share(fcf0, gg, w, metrics, tg)
        out[name] = {"growth": gg, "wacc": w, "terminal_growth": tg, "value": v}
    return out


def sensitivity(metrics, wacc=None, growth=None, wacc_steps=(-0.02, -0.01, 0, 0.01, 0.02),
                tg_steps=(-0.01, -0.005, 0, 0.005, 0.01)):
    """Grid of value/share: rows = WACC, cols = terminal growth."""
    wacc = wacc or compute_wacc(metrics)["wacc"]
    growth = growth if growth is not None else max(min(metrics.get("Revenuegrowth") or 0.05, 0.40), -0.05)
    fcf0 = metrics["Fcf"]
    rows = []
    for dw in wacc_steps:
        rows.append([dcf_value_per_share(fcf0, growth, wacc + dw, metrics, TERMINAL_G + dt)[0]
                     for dt in tg_steps])
    return {"wacc": [wacc + d for d in wacc_steps], "terminal_growth": [TERMINAL_G + d for d in tg_steps],
            "grid": rows}


def ddm(dps, ke, g_high, years=5, g_term=0.03):
    """Two-stage dividend discount model."""
    if not dps or dps <= 0:
        return None
    ke = max(ke, g_term + 0.01)
    pv, d = 0.0, dps
    for i in range(1, years + 1):
        d *= 1 + g_high
        pv += d / (1 + ke) ** i
    tv = d * (1 + g_term) / (ke - g_term)
    return pv + tv / (1 + ke) ** years


def residual_income(book_value_ps, roe, ke, years=5, fade=0.8, g_term=0.02):
    """RIM: BV0 + sum PV of residual income, with ROE fading toward ke."""
    if book_value_ps is None or roe is None:
        return None
    bv, pv, r = book_value_ps, 0.0, roe
    for i in range(1, years + 1):
        ri = (r - ke) * bv
        pv += ri / (1 + ke) ** i
        bv *= 1 + r * 0.6            # assume 40% payout
        r = ke + (r - ke) * fade
    tv = ((r - ke) * bv) / (ke - g_term) / (1 + ke) ** years if ke > g_term else 0
    return book_value_ps + pv + tv


def comps_valuation(metrics, peer_infos):
    """Implied price from peer median P/E, EV/EBITDA."""
    import statistics as st
    pes = [p["trailingPE"] for p in peer_infos if p and p.get("trailingPE") and 0 < p["trailingPE"] < 100]
    eves = [p["enterpriseValue"] / p["ebitda"] for p in peer_infos
            if p and p.get("enterpriseValue") and p.get("ebitda") and p["ebitda"] > 0]
    out, shares = {}, metrics.get("Sharesoutstanding")
    if pes and metrics.get("Profit") and shares and metrics["Profit"] > 0:
        out["pe_implied_price"] = st.median(pes) * metrics["Profit"] / shares
        out["peer_median_pe"] = st.median(pes)
    if eves and metrics.get("Ebidta") and shares and metrics["Ebidta"] > 0:
        ev = st.median(eves) * metrics["Ebidta"]
        out["ev_ebitda_implied_price"] = (ev - (metrics.get("Debt") or 0) + (metrics.get("Cash") or 0)) / shares
        out["peer_median_ev_ebitda"] = st.median(eves)
    return out


def precedent_transactions(metrics, multiples):
    """multiples: list of EV/EBITDA from comparable deals (supplied by the M&A tracker)."""
    import statistics as st
    if not multiples or not metrics.get("Ebidta") or not metrics.get("Sharesoutstanding"):
        return None
    ev = st.median(multiples) * metrics["Ebidta"]
    return (ev - (metrics.get("Debt") or 0) + (metrics.get("Cash") or 0)) / metrics["Sharesoutstanding"]


def sum_of_parts(segments, net_debt, shares):
    """segments: [{'name','metric','multiple'}] -> value/share."""
    ev = sum(s["metric"] * s["multiple"] for s in segments)
    return (ev - net_debt) / shares


def valuation_suite(metrics, peer_infos=None, rf=None):
    w = compute_wacc(metrics, rf=rf)
    out = {"wacc": w, "reverse_dcf": reverse_dcf(metrics, w["wacc"]),
           "scenarios": scenarios(metrics, w["wacc"])}
    try:
        if (metrics.get("Fcf") or 0) <= 0:
            raise ValueError("FCF is negative; sensitivity not meaningful")
        out["sensitivity"] = sensitivity(metrics, w["wacc"])
    except Exception as e:
        out["sensitivity"] = {"error": str(e)}
    ke = w["cost_of_equity"]
    sh = metrics.get("Sharesoutstanding")
    price_ = metrics.get("Currentprice") or 0
    dy = (metrics.get("DividendRate") or 0) / price_ if price_ else 0
    out["ddm"] = ddm(metrics.get("DividendRate"), ke, 0.05) if dy >= 0.01 else None
    out["ddm_note"] = None if out["ddm"] else f"not applicable (dividend yield {dy:.2%} < 1%)"
    out["residual_income"] = residual_income(metrics.get("BookValue"), metrics.get("Roe"), ke)
    if peer_infos:
        out["comps"] = comps_valuation(metrics, peer_infos)
    vals = [v["value"] for v in out["scenarios"].values()] if "bear" in out["scenarios"] else []
    price = metrics.get("Currentprice")
    if vals and price:
        out["fair_value_range"] = {"low": min(vals), "high": max(vals), "base": out["scenarios"]["base"]["value"],
                                   "margin_of_safety_base_pct": (out["scenarios"]["base"]["value"] - price) / price * 100}
    return out
