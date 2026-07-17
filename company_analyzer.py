import yfinance as yf
from google import genai
from pathlib import Path
from dotenv import load_dotenv
from dotenv import dotenv_values
import os
import matplotlib.pyplot as plt
from reportlab.platypus import Image
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
ticker=input("Enter stock:").strip().upper()
def data(ticker):
    stock=yf.Ticker(ticker)
    info=stock.info
    return info
    #import pandas as pd
    #df = pd.DataFrame(stock.info.items(),columns=["Metric", "Value"])
    #pd.set_option('display.max_rows', None)
stock=yf.Ticker(ticker)
info=data(ticker)
#-----Naming Values-----#
def get_financialmetrics(info):
    return{

        "Name":info.get("longName"),
        "MarketCap":info.get("marketCap"),
        "Revenue":info.get("totalRevenue"),
        "EnterpriseValue":info.get("enterpriseValue"),
        "Peratio":info.get("trailingPE"),
        "Peratiof":info.get("forwardPE"),
        "Profit":info.get("netIncomeToCommon"),
        "Revenuegrowth":info.get("revenueGrowth"),
        "ProfitMargin":info.get("profitMargins"),
        "Roe":info.get("returnOnEquity"),
        "Deratio":info.get("debtToEquity"),
        "Cash":info.get("totalCash"),
        "Debt":info.get("totalDebt"),
        "Avg":info.get("twoHundredDayAverage"),
        "Fcf":info.get("freeCashflow"),
        "Ebidta":info.get("ebitda"),
        "Pbratio":info.get("priceToBook"),
        "EvEbidtaratio":info.get("enterpriseValue") / info.get("ebitda")
        if info.get("enterpriseValue") is not None
        and info.get("ebitda") not in (None,0)
        else None,
        "Earningsgrowth":info.get("earningsGrowth"),
        "Beta":info.get("beta"),
        "Roa":info.get("returnOnAssets"),
        "Sharesoutstanding" : info.get("sharesOutstanding"),
        "Currentprice" :info.get("currentPrice"),
        "Sector":info.get("sector")
    }
    
#----------------------#

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

#-----Mapping to s&p 500-----#

def peertickers(sector):
    import pandas as pd
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "sp500.csv")
    sp500= pd.read_csv(csv_path)
    sector_map = {
        "Technology": "Information Technology",
        "Financial Services": "Financials",
        "Healthcare": "Health Care",
        "Consumer Cyclical": "Consumer Discretionary",
        "Consumer Defensive": "Consumer Staples",
        "Basic Materials": "Materials"
    }
    sp_sector = sector_map.get(sector, sector)
    same_sector = sp500[
       sp500["GICS Sector"] == sp_sector
    ]
    peer_tickers = same_sector["Symbol"].tolist()
    return peer_tickers


#---------------------------#

#------Financial statements------#
def finstatements(stock):
    balance_sheet = stock.balance_sheet
    financials = stock.financials
    cashflow = stock.cashflow   
    return balance_sheet,financials,cashflow
#----------------------------------#

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

#------Rating campanies-------#
#to rate it we need to find the avg/median values to compare against so we take data from
#s&p 500 caompanies of same sector and then compare against them
def rating(metrics,peer_tickers): 
    import statistics
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
    for ticker in peer_tickers:
        cdr=None
        evebitdas=None
        try:
            peerinfo=yf.Ticker(ticker).info
            margin=peerinfo.get("profitMargins")
            growth=peerinfo.get("revenueGrowth")
            roe=peerinfo.get("returnOnEquity")
            Beta=peerinfo.get("beta")
            der=peerinfo.get("debtToEquity")
            if peerinfo.get("totalCash") is not None and peerinfo.get("totalDebt") not in (None, 0):
                cdr = peerinfo["totalCash"] / peerinfo["totalDebt"]
            Fcf=peerinfo["freeCashflow"]
            roas=peerinfo["returnOnAssets"]
            eg=peerinfo.get("earningsGrowth")
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
        except Exception as e:
            print(f"{ticker}: {e}") 

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
    earningsgrowth=min((metrics["Earningsgrowth"]/egmargin)*100,100)
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

#---------News------------#
def News(stock):
    news = stock.news
    if not news:
        return []
    articles = []
    for article in news:
        content = article["content"]
        if content.get("contentType") != "STORY":
            continue
        articles.append({
            "Title": content.get("title"),
            "Summary": content.get("summary"),
            "Date": content.get("pubDate"),
            "Publisher": content.get("provider", {}).get("displayName"),
            "URL": content.get("canonicalUrl", {}).get("url")
        })
    return articles

#-------Total-------------#
def analyze_company(ticker):
    info=data(ticker)
    stock=yf.Ticker(ticker)
    metrics = get_financialmetrics(info)
    sector=metrics["Sector"]
    valuation = dcfcalc(metrics)
    peer_tickers=peertickers(sector)
    industry = rating(metrics,peer_tickers)
    Zscore=riskanalysis(stock)
    Fscore=Riskanalysis(stock)
    news=News(stock)
    return{"Metrics": metrics,
        "Valuation": valuation,
        "Industry": industry,"Altman Z'' score":Zscore,"Piotroski F-score":Fscore,
        "News": news}

#------------------------#

#-----------------Graphs-----------------#

def create_graphs(results):

    metrics = results["Metrics"]
    valuation = results["Valuation"]
    industry = results["Industry"]

    # Revenue vs Profit
    plt.figure(figsize=(6,4))

    plt.bar(
        ["Revenue","Profit"],
        [
            (metrics["Revenue"] or 0)/1e9,
            (metrics["Profit"] or 0)/1e9
        ],color=["#4F81BD", "#C0504D", "#9BBB59", "#8064A2"]
    )
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.title("Revenue vs Profit")
    plt.ylabel("USD (Billions)")
    plt.tight_layout()
    plt.savefig("revenue_profit.png")
    plt.close()

    # Cash vs Debt

    plt.figure(figsize=(6,4))

    plt.bar(
        ["Cash","Debt"],
        [
            (metrics["Cash"] or 0)/1e9,
            (metrics["Debt"] or 0)/1e9
        ],color=["#4F81BD", "#C0504D", "#9BBB59", "#8064A2"]
    )
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.title("Cash vs Debt")
    plt.ylabel("USD (Billions)")
    plt.tight_layout()

    plt.savefig("cash_debt.png")
    plt.close()

    # DCF

    plt.figure(figsize=(6,4))

    plt.bar(
        ["Current Price","Intrinsic Value"],
        [
            valuation["current_price"],
            valuation["intrinsic_value"]
        ],color=["#4F81BD", "#C0504D", "#9BBB59", "#8064A2"]
    )
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.title("DCF Valuation")
    plt.ylabel("USD")
    plt.tight_layout()

    plt.savefig("dcf.png")
    plt.close()

    # Industry Scores

    plt.figure(figsize=(7,4))

    plt.bar(
        [
            "Growth",
            "Profitability",
            "Risk",
            "Valuation"
        ],
        [
            industry["growthscore"],
            industry["profitabilityscore"],
            industry["riskscore"],
            industry["valuationscore"]
        ],color=["#4F81BD", "#C0504D", "#9BBB59", "#8064A2"]
    )
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.ylim(0,100)

    plt.title("Industry Scores")
    plt.ylabel("Score")

    plt.tight_layout()

    plt.savefig("industry.png")

    plt.close()
    
#-----Report Agent--------#

results = analyze_company(ticker)
create_graphs(results)
def report_agent(results,client):
    prompt = f"""
You are a Senior Equity Research Analyst working at a top investment firm such as Goldman Sachs or Morgan Stanley.
Your objective is to produce a professional, data-driven equity research report.
IMPORTANT RULES
1. Use ONLY the data supplied below.
2. NEVER invent financial numbers.
3. NEVER hallucinate news.
4. If information is missing, explicitly state "Data Not Available."
5. Explain WHY each metric matters instead of only describing it.
6. Be objective. Mention both strengths and weaknesses.
7. Support every conclusion using the supplied data.
8. Write professionally using markdown formatting.
9. Use tables whenever appropriate.
10. Be detailed enough that an experienced investor or financial analyst would find the report useful.
=========================================================
COMPANY METRICS
=========================================================
{results["Metrics"]}
=========================================================
DCF VALUATION
=========================================================
{results["Valuation"]}
=========================================================
INDUSTRY ANALYSIS
=========================================================
{results["Industry"]}
=========================================================
ALTMAN Z'' SCORE
=========================================================
{results["Altman Z'' score"]}
=========================================================
PIOTROSKI F-SCORE
=========================================================
{results["Piotroski F-score"]}
=========================================================
RECENT NEWS
=========================================================
{results["News"]}
=========================================================
WRITE THE REPORT IN THE FOLLOWING FORMAT
=========================================================
# COMPANY ANALYSIS REPORT
---
## Executive Dashboard
Provide a quick dashboard containing
- Company Name
- Overall Rating (/100)
- Recommendation (Strong Buy / Buy / Hold / Sell / Strong Sell)
- Confidence (/100)
- Current Price
- Intrinsic Value
- Margin of Safety
- Industry Score
- Risk Level
- Financial Health
---
## Executive Summary
Summarize the company in 5-10 concise bullet points.
Include
- Main strengths
- Main weaknesses
- Growth outlook
- Valuation outlook
- Overall opinion
---
## Business Overview
Explain
- What the company does
- Main products and services
- Revenue sources
- Competitive advantages
- Industry position
---
## Financial Performance
Analyze
Revenue
Revenue Growth
Profitability
ROE
ROA
Margins
Cash Position
Debt Position
Free Cash Flow
Comment on trends and quality of earnings.
---
## Valuation Analysis
Analyze
Current Price
Intrinsic Value
Margin of Safety
Whether the stock appears undervalued or overvalued
Explain the DCF result.
---
## Industry Comparison
Analyze
Overall Industry Score
Growth Score
Profitability Score
Risk Score
Valuation Score
Discuss how the company compares with its peers.
---
## Risk Analysis
Discuss
Altman Z'' Score
Piotroski F Score
Debt
Beta
Liquidity
Cash Flow
Mention both financial and business risks.
---
## Recent News Analysis
Summarize the supplied news.
Separate into
Positive Developments
Negative Developments
Neutral Developments
Explain how the news could affect the company.
Do NOT mention news that is not provided.
---
## Bull Case
Explain reasons the company could outperform.
---
## Bear Case
Explain reasons the company could underperform.
---
## Investment Thesis
Write a professional investment thesis combining
Financials
Valuation
Industry
Risk
Recent News
Explain whether this is attractive as a long-term investment.
---
## Final Recommendation
Give exactly one recommendation
- Strong Buy
- Buy
- Hold
- Sell
- Strong Sell
Explain why.
Assign a confidence score between 0 and 100.
After completing the report, append a JSON object enclosed in a markdown code block (```json ... ```), containing:
{{
  "overall_score": 87,
  "recommendation": "Buy",
  "confidence": 91,
  "target_price": 241.80,
  "key_strengths": [
    "...",
    "...",
    "..."
  ],
  "key_risks": [
    "...",
    "...",
    "..."
  ]
}}
---
## Appendix
Include all supplied raw metrics in neatly formatted tables.
"""
    
    import time
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            break
            
        except Exception as e:
            print(f"Attempt {attempt+1} failed:", e)
            time.sleep(5)
    else:
        raise Exception("Gemini unavailable after multiple attempts.")
    return response.text
#--------------------#

report=report_agent(results,client)

#-----PDF-----------#
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import HexColor

# ---------------- Styles ---------------- #

styles = getSampleStyleSheet()

title_style = styles["Title"]
title_style.fontName = "Helvetica-Bold"
title_style.fontSize = 26
title_style.leading = 32
title_style.alignment = TA_CENTER
title_style.textColor = HexColor("#0B3C5D")

heading1_style = styles["Heading1"]
heading1_style.fontName = "Helvetica-Bold"
heading1_style.fontSize = 18
heading1_style.leading = 22
heading1_style.spaceBefore = 18
heading1_style.spaceAfter = 10
heading1_style.textColor = HexColor("#0B3C5D")

heading2_style = styles["Heading2"]
heading2_style.fontName = "Helvetica-Bold"
heading2_style.fontSize = 15
heading2_style.leading = 20
heading2_style.spaceBefore = 15
heading2_style.spaceAfter = 8
heading2_style.textColor = HexColor("#1F4E79")

normal_style = styles["BodyText"]
normal_style.fontName = "Helvetica"
normal_style.fontSize = 11
normal_style.leading = 18
normal_style.spaceAfter = 6

cover_style = styles["BodyText"]
cover_style.fontName = "Helvetica"
cover_style.fontSize = 14
cover_style.leading = 20
cover_style.alignment = TA_CENTER

# ---------------- PDF ---------------- #

def pdf(ticker, report):

    if report is None:
        print("No report available.")
        return

    doc = SimpleDocTemplate(
        f"{ticker}_Report.pdf",
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    story = []

    # ======================================================
    # COVER PAGE
    # ======================================================

    story.append(Spacer(1,120))

    story.append(
        Paragraph(
            "COMPANY ANALYZER AI",
            title_style
        )
    )

    story.append(Spacer(1,30))

    story.append(
        Paragraph(
            f"<b>{ticker} EQUITY RESEARCH REPORT</b>",
            heading1_style
        )
    )

    story.append(Spacer(1,35))

    story.append(
        Paragraph(
            "Institutional Equity Research Report",
            cover_style
        )
    )

    story.append(Spacer(1,15))

    story.append(
        Paragraph(
            "Generated using Gemini AI",
            cover_style
        )
    )

    story.append(Spacer(1,15))

    story.append(
        Paragraph(
            "Company Analyzer AI v1.0",
            cover_style
        )
    )

    story.append(Spacer(1,50))

    story.append(
        Paragraph(
            "<i>Confidential • For Educational Purposes Only</i>",
            cover_style
        )
    )

    story.append(PageBreak())

    # ======================================================
    # REPORT TITLE
    # ======================================================

    story.append(
        Paragraph(
            "EQUITY RESEARCH REPORT",
            title_style
        )
    )
    img = Image("revenue_profit.png", width=390, height=240)
    img.hAlign = "CENTER"
    story.append(img)
    story.append(Spacer(1,20))

    img = Image("cash_debt.png", width=390, height=240)
    img.hAlign = "CENTER"
    story.append(img)
    story.append(Spacer(1,20))

    img = Image("dcf.png", width=390, height=240)
    img.hAlign = "CENTER"
    story.append(img)
    story.append(Spacer(1,20))

    img = Image("industry.png", width=390, height=240)
    img.hAlign = "CENTER"
    story.append(img)

    # ======================================================
    # PARSE GEMINI MARKDOWN
    # ======================================================

    for line in report.split("\n"):

        line = line.strip()

        if line == "":
            story.append(Spacer(1,10))
            continue

        if line.startswith("# "):
            story.append(
                Paragraph(
                    line.replace("# ",""),
                    title_style
                )
            )

        elif line.startswith("## "):
            story.append(
                Paragraph(
                    line.replace("## ",""),
                    heading1_style
                )
            )

        elif line.startswith("### "):
            story.append(
                Paragraph(
                    line.replace("### ",""),
                    heading2_style
                )
            )

        elif line.startswith("---"):
            story.append(Spacer(1,15))

        else:
            story.append(
                Paragraph(
                    line,
                    normal_style
                )
            )

    doc.build(story)

    print(f"{ticker}_Report.pdf created successfully.")
print(type(report))
print(report)
if report is None:
    print("Report generation failed.")
else:
    pdf(ticker, report)