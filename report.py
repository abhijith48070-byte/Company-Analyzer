import matplotlib.pyplot as plt
import yfinance as yf
from reportlab.platypus import Image
from concurrent.futures import ThreadPoolExecutor
from data import (
    get_company_info,
    get_financialmetrics,
    peertickers,
    News,
)

from analysis import (
    dcfcalc,
    riskanalysis,
    Riskanalysis,
    rating,
)

#-------Total-------------#
def analyze_company(ticker):
    stock, info = get_company_info(ticker)
    metrics = get_financialmetrics(info)
    valuation = dcfcalc(metrics)
    peer_tickers = peertickers(metrics["Sector"])

    with ThreadPoolExecutor(max_workers=4) as executor:
        industry_future = executor.submit(rating, metrics, peer_tickers)
        altman_future = executor.submit(riskanalysis, stock)
        piotroski_future = executor.submit(Riskanalysis, stock)
        news_future = executor.submit(News, stock)
        industry = industry_future.result()
        altman = altman_future.result()
        piotroski = piotroski_future.result()
        news = news_future.result()
    return {
        "Metrics": metrics,
        "Valuation": valuation,
        "Industry": industry,
        "Altman Z'' score": altman,
        "Piotroski F-score": piotroski,
        "News": news,
    }



#------------------------#

#-----------------Graphs-----------------#

def create_graphs(results):

    metrics = results["metrics"]
    valuation = results["valuation"]
    industry = results["industry"]

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
