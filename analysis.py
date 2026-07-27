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
    growth = min(metrics["Revenuegrowth"], 0.20)
    Discountrate=0.1
    
    rgrowth=[metrics["Fcf"]]
    
    for i in range(1,6):
        rgrowth.append(rgrowth[i-1]*(1+growth))
    presentvalue=[]
    for i in range(1,6):
        presentvalue.append(rgrowth[i]/((1+Discountrate)**(i)))
    #finding terminal value
    tv=(rgrowth[-1]*(1+0.04))/(0.1-0.04)
    #getting tv into present value
    pv_terminal=tv/((1+Discountrate)**5)
    EnterpriseValuenow=sum(presentvalue)+pv_terminal
    #converting into equity value
    debt=metrics["Debt"] or 0
    cash=metrics["Cash"] or 0
    equityvalue=EnterpriseValuenow-debt+cash
    #intrinsic value per share
    shares=metrics["Sharesoutstanding"]
    if shares is None or shares==0:
        return {"error: Shares Outstanding are not available"}
    intrinsicvaluepershare=equityvalue/shares
    price=metrics["Currentprice"]
    if price is None or price==0:
        return {"error": "Current Price unavailable"}
    margin = ((intrinsicvaluepershare-price)/ price) * 100
    return {
    "current_price": metrics["Currentprice"],
    "intrinsic_value": intrinsicvaluepershare,
    "margin_of_safety": margin,
    "valuation": "UNDERVALUED" if intrinsicvaluepershare > metrics["Currentprice"] else "OVERVALUED"
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
def Riskanalysis(stock):
    balance,financials,cashflow=finstatements(stock)
    net_income = financials.loc["Net Income"].iloc[0]
    prev_net_income = financials.loc["Net Income"].iloc[1]
    ocf = cashflow.loc["Operating Cash Flow"].iloc[0]
    prev_ocf = cashflow.loc["Operating Cash Flow"].iloc[1]
    assets = balance.loc["Total Assets"].iloc[0]
    prev_assets = balance.loc["Total Assets"].iloc[1]
    debt = balance.loc["Long Term Debt"].iloc[0]
    prev_debt = balance.loc["Long Term Debt"].iloc[1]
    current_assets = balance.loc["Current Assets"].iloc[0]
    prev_current_assets = balance.loc["Current Assets"].iloc[1]
    current_liabilities = balance.loc["Current Liabilities"].iloc[0]
    prev_current_liabilities = balance.loc["Current Liabilities"].iloc[1]
    shares = balance.loc["Share Issued"].iloc[0]
    prev_shares = balance.loc["Share Issued"].iloc[1]
    gross_profit = financials.loc["Gross Profit"].iloc[0]
    prev_gross_profit = financials.loc["Gross Profit"].iloc[1]
    revenue = financials.loc["Total Revenue"].iloc[0]
    prev_revenue = financials.loc["Total Revenue"].iloc[1]
    gross_margin = gross_profit / revenue
    prev_gross_margin = prev_gross_profit / prev_revenue
    asset_turnover = revenue / assets
    prev_asset_turnover = prev_revenue / prev_assets
    score = 0
# 1. Positive Net Income
    if net_income > 0:
        score += 1
# 2. Positive Operating Cash Flow
    if ocf > 0:
        score += 1
# 3. ROA Improved
    if (net_income / assets) > (prev_net_income / prev_assets):
        score += 1
# 4. Operating Cash Flow > Net Income
    if ocf > net_income:
        score += 1
# 5. Long-Term Debt Decreased
    if debt < prev_debt:
        score += 1
# 6. Current Ratio Improved
    current_ratio = current_assets / current_liabilities
    prev_current_ratio = prev_current_assets / prev_current_liabilities
    if current_ratio > prev_current_ratio:
        score += 1
# 7. No New Shares Issued
    if shares <= prev_shares:
        score += 1
# 8. Gross Margin Improved
    gross_margin = gross_profit / revenue
    prev_gross_margin = prev_gross_profit / prev_revenue    
    if gross_margin > prev_gross_margin:
        score += 1
# 9. Asset Turnover Improved
    asset_turnover = revenue / assets
    prev_asset_turnover = prev_revenue / prev_assets
    if asset_turnover > prev_asset_turnover:
        score += 1

    if score >= 8:
        status = "Excellent"
    elif score >= 6:
        status = "Strong"
    elif score >= 3:
        status = "Average"
    else:
        status = "Weak"
    return {
        "PiotroskiScore": score,
        "Status": status
    }
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
