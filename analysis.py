from data import finstatements
import yfinance as yf
import statistics
import math
from concurrent.futures import ThreadPoolExecutor
peer_cache = {}
#---------DCF-----------#
#No of years taking into account=5
#discountrate=10%
#after 5 years lets say the comapny grows 4% everyyear
def dcfcalc(metrics):
    if metrics.get("Fcf") is None or metrics.get("Revenuegrowth") is None:
        return {"error": "Insufficient data for DCF"}

    # Use Yahoo Finance's revenue growth estimate
    growth = metrics.get("Revenuegrowth") or 0.05
    growth = max(min(growth, 0.25), -0.05)

    riskfree = 0.043
    marketreturn = 0.10

    beta = metrics.get("Beta") or 1.0
    Discountrate = riskfree + beta * (marketreturn - riskfree)

    # Make sure discount rate is greater than terminal growth
    terminal_growth = 0.04
    if Discountrate <= terminal_growth:
        Discountrate = terminal_growth + 0.01

    rgrowth = [metrics["Fcf"]]

    for i in range(5):
        projected_growth = max(growth - i * 0.02, 0.03)
        rgrowth.append(rgrowth[-1] * (1 + projected_growth))

    presentvalue = []
    for i in range(1, 6):
        presentvalue.append(rgrowth[i] / ((1 + Discountrate) ** i))

    tv = (rgrowth[-1] * (1 + terminal_growth)) / (Discountrate - terminal_growth)
    pv_terminal = tv / ((1 + Discountrate) ** 5)

    EnterpriseValuenow = sum(presentvalue) + pv_terminal

    debt = metrics.get("Debt") or 0
    cash = metrics.get("Cash") or 0

    equityvalue = EnterpriseValuenow - debt + cash

    shares = metrics.get("Sharesoutstanding")
    if not shares:
        return {"error": "Shares Outstanding unavailable"}

    intrinsicvaluepershare = equityvalue / shares

    price = metrics.get("Currentprice")
    if not price:
        return {"error": "Current Price unavailable"}

    margin = ((intrinsicvaluepershare - price) / price) * 100

    return {
        "current_price": price,
        "intrinsic_value": intrinsicvaluepershare,
        "margin_of_safety": margin,
        "valuation": "UNDERVALUED" if intrinsicvaluepershare > price else "OVERVALUED"
    }

#---------------------------#

#---------Risk Agents----------#
#Altman Z-double prime-score
def riskanalysis(stock):
    balance_sheet,financials,cashflow=finstatements(stock)
    current_assets = balance_sheet.loc["Current Assets"].iloc[0]
    current_liabilities = balance_sheet.loc["Current Liabilities"].iloc[0]
    working_capital = current_assets - current_liabilities
    retained_earnings = balance_sheet.loc["Retained Earnings"].iloc[0]
    total_assets = balance_sheet.loc["Total Assets"].iloc[0]
    ebit = financials.loc["EBIT"].iloc[0]
    total_liabilities = balance_sheet.loc["Total Liabilities Net Minority Interest"].iloc[0]
    equity = balance_sheet.loc["Stockholders Equity"].iloc[0]

    #liquidity(a):
    liquidity=working_capital/total_assets
    #retained earnings(b):
    retained_earnings=retained_earnings/total_assets
    #operating profitability(c):
    op=ebit/total_assets
    #bookequity(d):
    bve=equity/total_liabilities
    #zscore=6.56(a)+3.26(b)+6.72(c)+1.05(d)
    zscore=(6.56*liquidity)+(3.26*retained_earnings)+(6.72*op)+(1.05*bve)
    if zscore>2.6:
        status="Low risk"
    elif 2.6>zscore>1.1:
        status="Grey zone"
    else:
        status="High risk"
    return {"Zscore":zscore,"Status":status}
#Piotroski F-score
def get_value(df, names, column=0, default=0):
    """
    Returns the first matching value from a financial statement.
    Tries multiple possible row names and falls back to partial matches.
    """
    # Exact matches
    for name in names:
        if name in df.index:
            try:
                return df.loc[name].iloc[column]
            except:
                pass

    # Partial matches
    for idx in df.index:
        for name in names:
            if name.lower() in str(idx).lower():
                try:
                    return df.loc[idx].iloc[column]
                except:
                    pass

    return default


def Riskanalysis(stock):
    balance, financials, cashflow = finstatements(stock)

    # Income Statement
    net_income = get_value(financials, ["Net Income"])
    prev_net_income = get_value(financials, ["Net Income"], 1)

    gross_profit = get_value(financials, ["Gross Profit"])
    prev_gross_profit = get_value(financials, ["Gross Profit"], 1)

    revenue = get_value(financials, ["Total Revenue", "Revenue"])
    prev_revenue = get_value(financials, ["Total Revenue", "Revenue"], 1)

    # Cash Flow
    ocf = get_value(
        cashflow,
        [
            "Operating Cash Flow",
            "Cash Flow From Operating Activities",
            "Total Cash From Operating Activities"
        ]
    )

    prev_ocf = get_value(
        cashflow,
        [
            "Operating Cash Flow",
            "Cash Flow From Operating Activities",
            "Total Cash From Operating Activities"
        ],
        1
    )

    # Balance Sheet
    assets = get_value(balance, ["Total Assets"])
    prev_assets = get_value(balance, ["Total Assets"], 1)

    debt = get_value(
        balance,
        [
            "Long Term Debt",
            "Total Debt",
            "Long Term Borrowings",
            "Long-Term Debt"
        ]
    )

    prev_debt = get_value(
        balance,
        [
            "Long Term Debt",
            "Total Debt",
            "Long Term Borrowings",
            "Long-Term Debt"
        ],
        1
    )

    current_assets = get_value(
        balance,
        [
            "Current Assets",
            "Total Current Assets"
        ]
    )

    prev_current_assets = get_value(
        balance,
        [
            "Current Assets",
            "Total Current Assets"
        ],
        1
    )

    current_liabilities = get_value(
        balance,
        [
            "Current Liabilities",
            "Total Current Liabilities"
        ]
    )

    prev_current_liabilities = get_value(
        balance,
        [
            "Current Liabilities",
            "Total Current Liabilities"
        ],
        1
    )

    shares = get_value(
        balance,
        [
            "Share Issued",
            "Ordinary Shares Number",
            "Common Stock Shares Outstanding"
        ]
    )

    prev_shares = get_value(
        balance,
        [
            "Share Issued",
            "Ordinary Shares Number",
            "Common Stock Shares Outstanding"
        ],
        1
    )

    # Avoid divide-by-zero
    gross_margin = gross_profit / revenue if revenue else 0
    prev_gross_margin = prev_gross_profit / prev_revenue if prev_revenue else 0

    asset_turnover = revenue / assets if assets else 0
    prev_asset_turnover = prev_revenue / prev_assets if prev_assets else 0

    score = 0

    # Profitability
    if net_income > 0:
        score += 1

    if ocf > 0:
        score += 1

    if net_income > prev_net_income:
        score += 1

    if ocf > net_income:
        score += 1

    # Leverage/Liquidity
    if debt < prev_debt:
        score += 1

    current_ratio = current_assets / current_liabilities if current_liabilities else 0
    prev_current_ratio = (
        prev_current_assets / prev_current_liabilities
        if prev_current_liabilities
        else 0
    )

    if current_ratio > prev_current_ratio:
        score += 1

    if shares <= prev_shares:
        score += 1

    # Operating Efficiency
    if gross_margin > prev_gross_margin:
        score += 1

    if asset_turnover > prev_asset_turnover:
        score += 1

    return score
#------------------#
def get_peer_info(ticker):
    if ticker in peer_cache:
        return peer_cache[ticker]

    try:
        info = yf.Ticker(ticker).info
        peer_cache[ticker] = info
        return info
    except Exception:
        return None
#------Rating campanies-------#
#to rate it we need to find the avg/median values to compare against so we take data from
#s&p 500 caompanies of same sector and then compare against them
def rating(metrics,peer_tickers): 
    
    margins=[]
    growths=[]
    roes=[]
    beta=[]
    debteqratio=[]
    cashdebtratio=[]
    fcf=[]
    roa=[]
    egs=[]
    pet=[]
    pef,pbr,evebitda=[],[],[]
    with ThreadPoolExecutor(max_workers=8) as executor:
        peer_infos = list(executor.map(get_peer_info, peer_tickers))

    for peerinfo in peer_infos:
        if peerinfo is None:
            continue

        cdr = None
        evebitdas = None

        margin = peerinfo.get("profitMargins")
        growth = peerinfo.get("revenueGrowth")
        roe = peerinfo.get("returnOnEquity")
        Beta = peerinfo.get("beta")
        der = peerinfo.get("debtToEquity")

        if peerinfo.get("totalCash") is not None and peerinfo.get("totalDebt") not in (None, 0):
            cdr = peerinfo["totalCash"] / peerinfo["totalDebt"]

        Fcf = peerinfo.get("freeCashflow")
        roas = peerinfo.get("returnOnAssets")
        eg = peerinfo.get("earningsGrowth")
        pe = peerinfo.get("trailingPE")
        forward_pe = peerinfo.get("forwardPE")
        pb = peerinfo.get("priceToBook")

        if peerinfo.get("enterpriseValue") is not None and peerinfo.get("ebitda") not in (None, 0):
            evebitdas = peerinfo["enterpriseValue"] / peerinfo["ebitda"]

        if margin is not None:
            margins.append(margin)
        if growth is not None:
            growths.append(growth)
        if roe is not None:
            roes.append(roe)
        if Beta is not None:
            beta.append(Beta)
        if der is not None:
            debteqratio.append(der)
        if cdr is not None:
            cashdebtratio.append(cdr)
        if Fcf is not None:
            fcf.append(Fcf)
        if roas is not None:
            roa.append(roas)
        if eg is not None:
            egs.append(eg)
        if pe is not None:
            pet.append(pe)
        if forward_pe is not None:
            pef.append(forward_pe)
        if pb is not None:
            pbr.append(pb)
        if evebitdas is not None:
            evebitda.append(evebitdas)

    medianmargin=statistics.median(margins)
    growthmargin=statistics.median(growths)
    roemargin=statistics.median(roes)
    betamargin=statistics.median(beta)
    dermargin=statistics.median(debteqratio)
    cdrmargin=statistics.median(cashdebtratio)
    fcfmargin=statistics.median(fcf)
    roamargin=statistics.median(roa)
    egmargin=statistics.median(egs)
    pemargin=statistics.median(pet)
    fmargin=statistics.median(pef)
    pbrmargin=statistics.median(pbr)
    evedmargin=statistics.median(evebitda)
#scoring them
    #growth
    growthscores=min((metrics["Revenuegrowth"]/growthmargin)*100,100)
    if egmargin == 0:
        earningsgrowth = 0
    elif metrics["Earningsgrowth"] is None:
        earningsgrowth=0
    else:
        earningsgrowth = min((metrics["Earningsgrowth"] / egmargin) * 100, 100)
    growthscore=(growthscores+earningsgrowth)/2
    #profitability
    marginscore=min((metrics["ProfitMargin"]/medianmargin)*100,100)
    roescore=min((metrics["Roe"]/roemargin)*100,100)
    roascore=min((metrics["Roa"]/roamargin)*100,100)
    profitabilityscore=(marginscore+roescore+roascore)/3
    #risk
    betascore=min((betamargin/metrics["Beta"])*100,100)
    derscore=min((dermargin/metrics["Deratio"])*100,100)
    if metrics["Debt"] in (None,0):
        cdrscore=50
    else:
        cdrscore=min(((metrics["Cash"]/metrics["Debt"])/cdrmargin)*100,100)
    fcfscore=min((metrics["Fcf"]/fcfmargin)*100,100)
    riskscore=(betascore+derscore+cdrscore+fcfscore)/4
    #valuation
    pescore=min((metrics["Peratio"]/pemargin)*100,100)
    fscore=min((metrics["Peratiof"]/fmargin)*100,100)
    pbrscore=min((metrics["Pbratio"]/pbrmargin)*100,100)
    evedscore=min((metrics["EvEbidtaratio"]/evedmargin)*100,100)
    valuationscore=(pescore+fscore+pbrscore+evedscore)/4
    #total score
    finalscore=(growthscore+riskscore+profitabilityscore+valuationscore)/4
    return {
    "overall_score": round(finalscore, 2),
    "profitabilityscore": round(profitabilityscore, 2),
    "growthscore": round(growthscore, 2),
    "riskscore": round(riskscore, 2),
    "valuationscore":round(valuationscore,2)
    }
