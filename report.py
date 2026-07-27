import json
import re
from datetime import datetime

import matplotlib.pyplot as plt
import yfinance as yf

from concurrent.futures import ThreadPoolExecutor

from reportlab.lib import colors
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
    Table,
    TableStyle
)

from data import (
    get_company_info,
    get_financialmetrics,
    peertickers,
    News,
)

from analysis import (
    dcfcalc,
    rating,
    riskanalysis,
    Riskanalysis,
)

# ============================================================
# COLOR PALETTE
# ============================================================

NAVY = HexColor("#0B3C5D")
BLUE = HexColor("#1F4E79")
LIGHT = HexColor("#F4F6F9")
LIGHT_GREY = HexColor("#DDDDDD")
GREY = HexColor("#777777")
GREEN = HexColor("#2E7D32")
RED = HexColor("#C62828")
BLACK = HexColor("#222222")
WHITE = colors.white

# ============================================================
# DOCUMENT STYLES
# ============================================================

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "Title",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=30,
    leading=34,
    alignment=TA_CENTER,
    textColor=NAVY,
    spaceAfter=18,
)

subtitle_style = ParagraphStyle(
    "Subtitle",
    parent=styles["Heading2"],
    fontName="Helvetica",
    fontSize=18,
    leading=24,
    alignment=TA_CENTER,
    textColor=GREY,
    spaceAfter=20,
)

heading1_style = ParagraphStyle(
    "Heading1",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    textColor=NAVY,
    spaceBefore=18,
    spaceAfter=12,
)

heading2_style = ParagraphStyle(
    "Heading2",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=15,
    leading=20,
    textColor=BLUE,
    spaceBefore=14,
    spaceAfter=8,
)

normal_style = ParagraphStyle(
    "Normal",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=11,
    leading=18,
    textColor=BLACK,
)
# ============================================================
# ADDITIONAL STYLES
# ============================================================

cover_style = ParagraphStyle(
    "Cover",
    parent=normal_style,
    fontName="Helvetica",
    fontSize=14,
    leading=22,
    alignment=TA_CENTER,
    textColor=GREY,
)

small_style = ParagraphStyle(
    "Small",
    parent=normal_style,
    fontName="Helvetica",
    fontSize=10,
    leading=12,
    alignment=TA_CENTER,
    textColor=GREY,
)

metric_style = ParagraphStyle(
    "Metric",
    parent=normal_style,
    fontName="Helvetica-Bold",
    fontSize=18,
    alignment=TA_CENTER,
    textColor=NAVY,
)

metric_title_style = ParagraphStyle(
    "MetricTitle",
    parent=normal_style,
    fontName="Helvetica-Bold",
    fontSize=11,
    alignment=TA_CENTER,
    textColor=GREY,
)

recommend_style = ParagraphStyle(
    "Recommend",
    parent=normal_style,
    fontName="Helvetica-Bold",
    fontSize=18,
    alignment=TA_CENTER,
    textColor=WHITE,
)

table_header_style = ParagraphStyle(
    "TableHeader",
    parent=normal_style,
    fontName="Helvetica-Bold",
    fontSize=11,
    textColor=WHITE,
    alignment=TA_CENTER,
)

table_cell_style = ParagraphStyle(
    "TableCell",
    parent=normal_style,
    fontName="Helvetica",
    fontSize=10,
    textColor=BLACK,
)

# ============================================================
# PDF HELPERS
# ============================================================

def divider(width=500):

    table = Table([[""]], colWidths=[width])

    table.setStyle(
        TableStyle(
            [
                ("LINEBELOW", (0, 0), (-1, -1), 0.6, LIGHT_GREY),
            ]
        )
    )

    return table


def recommendation_color(text):

    text = text.lower()

    if "strong buy" in text:
        return GREEN

    if "buy" in text:
        return GREEN

    if "hold" in text:
        return HexColor("#F9A825")

    if "sell" in text:
        return RED

    if "underperform" in text:
        return RED

    if "strong sell" in text:
        return RED

    return NAVY


def currency(value):

    if value is None:
        return "N/A"

    return f"${value:,.2f}"


def percentage(value):

    if value is None:
        return "N/A"

    return f"{value:.2f}%"
# ============================================================
# ANALYZE COMPANY
# ============================================================

def analyze_company(ticker):

    print(f"\nFetching data for {ticker}...\n")

    stock, info = get_company_info(ticker)

    metrics = get_financialmetrics(info)

    valuation = dcfcalc(metrics)

    peer_tickers = peertickers(metrics["Sector"])

    with ThreadPoolExecutor(max_workers=4) as executor:

        industry_future = executor.submit(
            rating,
            metrics,
            peer_tickers
        )

        altman_future = executor.submit(
            riskanalysis,
            stock
        )

        piotroski_future = executor.submit(
            Riskanalysis,
            stock
        )

        news_future = executor.submit(
            News,
            stock
        )

        industry = industry_future.result()

        altman = altman_future.result()

        piotroski = piotroski_future.result()

        news = news_future.result()

    return {

        "ticker": ticker,

        "metrics": metrics,

        "valuation": valuation,

        "industry": industry,

        "altman": altman,

        "piotroski": piotroski,

        "news": news,

    }
# ============================================================
# GRAPH GENERATION
# ============================================================

def create_graphs(results):

    metrics = results["metrics"]

    valuation = results["valuation"]

    industry = results["industry"]

    # --------------------------------------------------------

    colors = [

        "#0B5394",

        "#38761D",

        "#CC0000",

        "#674EA7"

    ]

    # Revenue vs Profit

    plt.figure(figsize=(7,4))

    plt.bar(

        ["Revenue","Profit"],

        [

            (metrics["Revenue"] or 0)/1e9,

            (metrics["Profit"] or 0)/1e9

        ],

        color=colors[:2]

    )

    plt.grid(axis="y",alpha=.3)

    plt.ylabel("USD (Billions)")

    plt.title("Revenue vs Profit")

    plt.tight_layout()

    plt.savefig(

        "revenue_profit.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    # --------------------------------------------------------

    plt.figure(figsize=(7,4))

    plt.bar(

        ["Cash","Debt"],

        [

            (metrics["Cash"] or 0)/1e9,

            (metrics["Debt"] or 0)/1e9

        ],

        color=colors[2:]

    )

    plt.grid(axis="y",alpha=.3)

    plt.ylabel("USD (Billions)")

    plt.title("Cash vs Debt")

    plt.tight_layout()

    plt.savefig(

        "cash_debt.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()
        # --------------------------------------------------------
    # DCF Valuation
    # --------------------------------------------------------

    plt.figure(figsize=(7,4))

    plt.bar(
        ["Current Price", "Intrinsic Value"],
        [
            valuation["current_price"],
            valuation["intrinsic_value"]
        ],
        color=["#0B5394", "#38761D"]
    )

    plt.grid(axis="y", alpha=.3)

    plt.ylabel("USD")

    plt.title("DCF Valuation")

    plt.tight_layout()

    plt.savefig(
        "dcf.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Industry Scores
    # --------------------------------------------------------

    plt.figure(figsize=(8,4))

    labels = [
        "Growth",
        "Profitability",
        "Risk",
        "Valuation"
    ]

    values = [
        industry["growthscore"],
        industry["profitabilityscore"],
        industry["riskscore"],
        industry["valuationscore"]
    ]

    plt.bar(
        labels,
        values,
        color=[
            "#1565C0",
            "#2E7D32",
            "#EF6C00",
            "#8E24AA"
        ]
    )

    plt.ylim(0,100)

    plt.grid(axis="y", alpha=.3)

    plt.ylabel("Score")

    plt.title("Industry Comparison")

    plt.tight_layout()

    plt.savefig(
        "industry.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Graphs created successfully.")
# ============================================================
# REPORT AGENT
# ============================================================

def report_agent(results, client):

    prompt = f"""
You are a Senior Equity Research Analyst working at Goldman Sachs, Morgan Stanley or J.P. Morgan.

Your job is to write an institutional-quality equity research report.

IMPORTANT RULES

- Use ONLY the supplied data.
- Never invent numbers.
- Never invent news.
- Explain WHY every metric matters.
- Mention strengths and weaknesses.
- Write professionally.
- Use markdown headings.
- Use markdown tables where appropriate.

======================================================
COMPANY METRICS
======================================================

{results["metrics"]}

======================================================
DCF
======================================================

{results["valuation"]}

======================================================
INDUSTRY
======================================================

{results["industry"]}

======================================================
ALTMAN SCORE
======================================================

{results["altman"]}

======================================================
PIOTROSKI SCORE
======================================================

{results["piotroski"]}

======================================================
NEWS
======================================================

{results["news"]}

Write the report using this structure:

# COMPANY ANALYSIS REPORT

## Executive Dashboard

## Executive Summary

## Business Overview

## Financial Performance

## Valuation Analysis

## Industry Comparison

## Risk Analysis

## Recent News

## Bull Case

## Bear Case

## Investment Thesis

## Final Recommendation

## Appendix

Finish with a JSON object enclosed inside

```json
"""
    import time

    response = None

    for attempt in range(5):

        try:

            response = client.models.generate_content(

                model="gemini-2.5-flash",

                contents=prompt

            )

            break

        except Exception as e:

            print(f"Gemini attempt {attempt+1} failed")

            print(e)

            time.sleep(5)

    if response is None:

        raise Exception("Gemini unavailable after 5 retries.")

    return response.text

# ============================================================
# PAGE FOOTER
# ============================================================

def page_footer(canvas, doc):

    canvas.saveState()

    width, height = letter

    canvas.setStrokeColor(LIGHT_GREY)

    canvas.line(40, 32, width - 40, 32)

    canvas.setFont("Helvetica", 9)

    canvas.setFillColor(GREY)

    canvas.drawString(
        40,
        18,
        "Company Analyzer AI • Institutional Equity Research"
    )

    canvas.drawRightString(
        width - 40,
        18,
        f"Page {doc.page}"
    )

    canvas.restoreState()
# ============================================================
# PDF GENERATOR
# ============================================================

def build_cover(story, results):
    """
    Cover Page
    """

    ticker = results["ticker"]
    metrics = results["metrics"]
    valuation = results["valuation"]

    company = metrics.get("Name", ticker)
    sector = metrics.get("Sector", "N/A")

    recommendation = valuation.get("valuation", "HOLD").upper()

    current_price = valuation.get("current_price")
    intrinsic = valuation.get("intrinsic_value")
    margin = valuation.get("margin_of_safety")

    today = datetime.now().strftime("%d %B %Y")

    # ---------------------------------------------------------
    # TITLE
    # ---------------------------------------------------------

    story.append(Spacer(1, 35))

    story.append(
        Paragraph(
            "COMPANY ANALYZER AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Institutional Equity Research Report",
            subtitle_style
        )
    )

    story.append(Spacer(1, 15))

    story.append(divider())

    story.append(Spacer(1, 25))

    # ---------------------------------------------------------
    # COMPANY DETAILS
    # ---------------------------------------------------------

    story.append(
        Paragraph(
            f"<b>{company}</b>",
            heading1_style
        )
    )

    story.append(
        Paragraph(
            f"Ticker : {ticker}",
            cover_style
        )
    )

    story.append(
        Paragraph(
            f"Sector : {sector}",
            cover_style
        )
    )

    story.append(
        Paragraph(
            today,
            cover_style
        )
    )

    story.append(Spacer(1, 25))

    # ---------------------------------------------------------
    # RECOMMENDATION BADGE
    # ---------------------------------------------------------

    badge = Table(
        [[Paragraph(recommendation, recommend_style)]],
        colWidths=[220]
    )

    badge.setStyle(TableStyle([

        ("BACKGROUND", (0,0), (-1,-1),
            recommendation_color(recommendation)
        ),

        ("TEXTCOLOR",(0,0),(-1,-1),WHITE),

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),

        ("TOPPADDING",(0,0),(-1,-1),12),

        ("BOTTOMPADDING",(0,0),(-1,-1),12),

        ("BOX",(0,0),(-1,-1),1,
            recommendation_color(recommendation)
        ),

    ]))

    story.append(badge)

    story.append(Spacer(1,30))

    # ---------------------------------------------------------
    # VALUATION
    # ---------------------------------------------------------

    valuation_table = Table(

        [

            [

                Paragraph("<b>Current Price</b>",table_header_style),

                Paragraph("<b>Intrinsic Value</b>",table_header_style),

                Paragraph("<b>Margin of Safety</b>",table_header_style)

            ],

            [

                Paragraph(currency(current_price),metric_style),

                Paragraph(currency(intrinsic),metric_style),

                Paragraph(percentage(margin),metric_style)

            ]

        ],

        colWidths=[160,160,160]

    )

    valuation_table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,0),NAVY),

        ("TEXTCOLOR",(0,0),(-1,0),WHITE),

        ("BACKGROUND",(0,1),(-1,-1),LIGHT),

        ("GRID",(0,0),(-1,-1),0.5,LIGHT_GREY),

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),

        ("TOPPADDING",(0,0),(-1,-1),10),

        ("BOTTOMPADDING",(0,0),(-1,-1),10),

    ]))

    story.append(valuation_table)

    story.append(Spacer(1,30))

    # ---------------------------------------------------------
    # AI WORKFLOW
    # ---------------------------------------------------------

    workflow = Table([

        [

            "AI Generated",

            "LangGraph",

            "Gemini",

            "DCF + Risk"

        ]

    ],

    colWidths=[120]*4

    )

    workflow.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,-1),LIGHT),

        ("GRID",(0,0),(-1,-1),0.4,LIGHT_GREY),

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("TOPPADDING",(0,0),(-1,-1),10),

        ("BOTTOMPADDING",(0,0),(-1,-1),10),

    ]))

    story.append(workflow)

    story.append(Spacer(1,25))

    story.append(divider())

    story.append(Spacer(1,10))

    story.append(
        Paragraph(
            "Company Analyzer AI v2.0",
            small_style
        )
    )

    story.append(
        Paragraph(
            "Automatically generated institutional equity research report",
            small_style
        )
    )

    story.append(
        Paragraph(
            "For educational purposes only",
            small_style
        )
    )

    story.append(PageBreak())
def build_dashboard(story, results):
    """
    Executive Dashboard
    """

    report = results["report"]

    metrics = results["metrics"]

    valuation = results["valuation"]

    ticker = results["ticker"]

    company = metrics.get("Name", ticker)

    sector = metrics.get("Sector", "N/A")

    today = datetime.now().strftime("%d %B %Y")

    recommendation = valuation.get("valuation", "HOLD")

    intrinsic = valuation.get("intrinsic_value")

    overall_score = "N/A"

    confidence = "N/A"

    target_price = currency(intrinsic)

    # ----------------------------------------------------
    # Parse JSON returned by Gemini
    # ----------------------------------------------------

    try:

        match = re.search(
            r"```json(.*?)```",
            report,
            re.DOTALL
        )

        if match:

            data = json.loads(match.group(1))

            overall_score = str(
                data.get("overall_score", "N/A")
            )

            confidence = str(
                data.get("confidence", "N/A")
            )

            recommendation = data.get(
                "recommendation",
                recommendation
            )

            target_price = currency(
                data.get(
                    "target_price",
                    intrinsic
                )
            )

    except Exception:
        pass

    # ----------------------------------------------------

    story.append(
        Paragraph(
            "EXECUTIVE DASHBOARD",
            title_style
        )
    )

    story.append(Spacer(1, 15))

    dashboard = [

        [

            Paragraph("<b>Overall Score</b>", table_header_style),

            Paragraph("<b>Recommendation</b>", table_header_style),

            Paragraph("<b>Confidence</b>", table_header_style),

            Paragraph("<b>Target Price</b>", table_header_style)

        ],

        [

            Paragraph(overall_score, metric_style),

            Paragraph(recommendation.upper(), metric_style),

            Paragraph(confidence + "%", metric_style),

            Paragraph(target_price, metric_style)

        ]

    ]

    dashboard_table = Table(
        dashboard,
        colWidths=[120] * 4
    )

    dashboard_table.setStyle(TableStyle([

        ("BACKGROUND", (0,0), (-1,0), NAVY),

        ("TEXTCOLOR", (0,0), (-1,0), WHITE),

        ("BACKGROUND", (0,1), (-1,-1), LIGHT),

        ("GRID", (0,0), (-1,-1), 0.5, LIGHT_GREY),

        ("ALIGN", (0,0), (-1,-1), "CENTER"),

        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),

        ("TOPPADDING", (0,0), (-1,-1), 10),

        ("BOTTOMPADDING", (0,0), (-1,-1), 10),

    ]))

    story.append(dashboard_table)

    story.append(Spacer(1, 25))

    # ----------------------------------------------------

    company_table = Table(

        [

            [

                Paragraph("<b>Company</b>", table_header_style),

                Paragraph("<b>Ticker</b>", table_header_style),

                Paragraph("<b>Sector</b>", table_header_style),

                Paragraph("<b>Date</b>", table_header_style)

            ],

            [

                Paragraph(company, table_cell_style),

                Paragraph(ticker, table_cell_style),

                Paragraph(sector, table_cell_style),

                Paragraph(today, table_cell_style)

            ]

        ],

        colWidths=[170, 90, 120, 100]

    )

    company_table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,0),BLUE),

        ("TEXTCOLOR",(0,0),(-1,0),WHITE),

        ("BACKGROUND",(0,1),(-1,-1),WHITE),

        ("GRID",(0,0),(-1,-1),0.4,LIGHT_GREY),

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("TOPPADDING",(0,0),(-1,-1),10),

        ("BOTTOMPADDING",(0,0),(-1,-1),10),

    ]))

    story.append(company_table)

    story.append(Spacer(1,25))

    # ----------------------------------------------------

    summary = Table(

        [

            ["Current Price", currency(valuation.get("current_price"))],

            ["Intrinsic Value", currency(valuation.get("intrinsic_value"))],

            ["Margin of Safety", percentage(valuation.get("margin_of_safety"))],

            ["Sector", sector]

        ],

        colWidths=[220,220]

    )

    summary.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,-1),LIGHT),

        ("GRID",(0,0),(-1,-1),0.4,LIGHT_GREY),

        ("BOTTOMPADDING",(0,0),(-1,-1),8),

        ("TOPPADDING",(0,0),(-1,-1),8),

    ]))

    story.append(summary)

    story.append(PageBreak())
def build_visuals(story):
    """
    Visual Analysis Page
    """

    story.append(
        Paragraph(
            "VISUAL ANALYSIS",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Financial and Valuation Charts",
            subtitle_style
        )
    )

    story.append(Spacer(1,20))

    charts = [

        ("Revenue vs Profit", "revenue_profit.png"),

        ("Cash vs Debt", "cash_debt.png"),

        ("DCF Valuation", "dcf.png"),

        ("Industry Comparison", "industry.png")

    ]

    rows = []

    current_row = []

    for title, filename in charts:

        try:

            image = Image(
                filename,
                width=235,
                height=170
            )

            card = Table(

                [

                    [

                        Paragraph(
                            f"<b>{title}</b>",
                            heading2_style
                        )

                    ],

                    [

                        image

                    ]

                ],

                colWidths=[240]

            )

            card.setStyle(

                TableStyle(

                    [

                        ("GRID",(0,0),(-1,-1),0.4,LIGHT_GREY),

                        ("BACKGROUND",(0,0),(-1,0),LIGHT),

                        ("ALIGN",(0,0),(-1,-1),"CENTER"),

                        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),

                        ("BOTTOMPADDING",(0,0),(-1,-1),10),

                        ("TOPPADDING",(0,0),(-1,-1),10),

                    ]

                )

            )

        except Exception:

            card = Table(

                [

                    [

                        Paragraph(

                            f"{title}<br/><br/>Chart Not Found",

                            normal_style

                        )

                    ]

                ],

                colWidths=[240],

                rowHeights=[200]

            )

            card.setStyle(

                TableStyle(

                    [

                        ("GRID",(0,0),(-1,-1),0.5,RED),

                        ("ALIGN",(0,0),(-1,-1),"CENTER"),

                        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),

                    ]

                )

            )

        current_row.append(card)

        if len(current_row) == 2:

            rows.append(current_row)

            current_row = []

    if current_row:

        while len(current_row) < 2:

            current_row.append("")

        rows.append(current_row)

    dashboard = Table(

        rows,

        colWidths=[250,250]

    )

    dashboard.setStyle(

        TableStyle(

            [

                ("VALIGN",(0,0),(-1,-1),"TOP"),

                ("ALIGN",(0,0),(-1,-1),"CENTER"),

                ("BOTTOMPADDING",(0,0),(-1,-1),12),

            ]

        )

    )

    story.append(dashboard)

    story.append(PageBreak())
def build_ai_report(story, results):
    """
    AI Generated Equity Research Report
    """

    report = results["report"]

    story.append(
        Paragraph(
            "DETAILED EQUITY RESEARCH REPORT",
            title_style
        )
    )

    story.append(Spacer(1, 15))

    # -------------------------------------------------------
    # Remove JSON block
    # -------------------------------------------------------

    report = re.sub(
        r"```json.*?```",
        "",
        report,
        flags=re.DOTALL
    )

    lines = report.split("\n")

    bullet_buffer = []

    numbered_buffer = []

    def flush_bullets():
        nonlocal bullet_buffer

        if bullet_buffer:

            for item in bullet_buffer:

                story.append(
                    Paragraph(
                        f"• {item}",
                        normal_style
                    )
                )

            story.append(Spacer(1,6))

            bullet_buffer = []

    def flush_numbers():
        nonlocal numbered_buffer

        if numbered_buffer:

            for item in numbered_buffer:

                story.append(
                    Paragraph(
                        item,
                        normal_style
                    )
                )

            story.append(Spacer(1,6))

            numbered_buffer = []

    # -------------------------------------------------------

    for raw in lines:

        line = raw.strip()

        if not line:

            flush_bullets()
            flush_numbers()

            story.append(Spacer(1,6))

            continue

        # --------------------------------------------

        if line.startswith("# "):

            flush_bullets()
            flush_numbers()

            story.append(
                Paragraph(
                    line[2:],
                    title_style
                )
            )

            continue

        if line.startswith("## "):

            flush_bullets()
            flush_numbers()

            story.append(
                Paragraph(
                    line[3:],
                    heading1_style
                )
            )

            continue

        if line.startswith("### "):

            flush_bullets()
            flush_numbers()

            story.append(
                Paragraph(
                    line[4:],
                    heading2_style
                )
            )

            continue

        # --------------------------------------------

        if line.startswith("---"):

            flush_bullets()
            flush_numbers()

            story.append(divider())

            story.append(Spacer(1,10))

            continue

        # --------------------------------------------

        if line.startswith("- "):

            bullet_buffer.append(line[2:])

            continue

        if re.match(r"^\d+\.", line):

            numbered_buffer.append(line)

            continue

        # --------------------------------------------
        # Markdown table
        # --------------------------------------------

        if "|" in line:

            flush_bullets()
            flush_numbers()

            cells = [

                c.strip()

                for c in line.split("|")

                if c.strip()

            ]

            if len(cells) >= 2:

                tbl = Table(
                    [cells]
                )

                tbl.setStyle(
                    TableStyle(
                        [

                            ("GRID",(0,0),(-1,-1),0.4,LIGHT_GREY),

                            ("BACKGROUND",(0,0),(-1,-1),LIGHT),

                            ("BOTTOMPADDING",(0,0),(-1,-1),6),

                            ("TOPPADDING",(0,0),(-1,-1),6),

                        ]
                    )
                )

                story.append(tbl)

                continue

        # --------------------------------------------

        flush_bullets()
        flush_numbers()

        line = re.sub(
            r"\*\*(.*?)\*\*",
            r"<b>\1</b>",
            line
        )

        line = re.sub(
            r"\*(.*?)\*",
            r"<i>\1</i>",
            line
        )

        story.append(
            Paragraph(
                line,
                normal_style
            )
        )

    flush_bullets()
    flush_numbers()

    story.append(PageBreak())
def build_summary(story, results):
    """
    Investment Summary
    """

    metrics = results["metrics"]
    valuation = results["valuation"]

    company = metrics.get("Name", results["ticker"])
    ticker = results["ticker"]
    sector = metrics.get("Sector", "N/A")

    recommendation = valuation.get(
        "valuation",
        "HOLD"
    ).upper()

    current = valuation.get("current_price")
    intrinsic = valuation.get("intrinsic_value")
    margin = valuation.get("margin_of_safety")

    story.append(
        Paragraph(
            "INVESTMENT SUMMARY",
            title_style
        )
    )

    story.append(Spacer(1,20))

    # --------------------------------------------------------
    # Recommendation Badge
    # --------------------------------------------------------

    badge = Table(

        [[

            Paragraph(
                recommendation,
                recommend_style
            )

        ]],

        colWidths=[500]

    )

    badge.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,-1),
            recommendation_color(recommendation)
        ),

        ("TEXTCOLOR",(0,0),(-1,-1),WHITE),

        ("ALIGN",(0,0),(-1,-1),"CENTER"),

        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),

        ("TOPPADDING",(0,0),(-1,-1),15),

        ("BOTTOMPADDING",(0,0),(-1,-1),15),

        ("BOX",(0,0),(-1,-1),1,
            recommendation_color(recommendation)
        )

    ]))

    story.append(badge)

    story.append(Spacer(1,25))

    # --------------------------------------------------------

    summary = Table(

        [

            ["Company", company],

            ["Ticker", ticker],

            ["Sector", sector],

            ["Current Price", currency(current)],

            ["Intrinsic Value", currency(intrinsic)],

            ["Margin of Safety", percentage(margin)],

            ["Recommendation", recommendation]

        ],

        colWidths=[200,280]

    )

    summary.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,-1),LIGHT),

        ("GRID",(0,0),(-1,-1),0.4,LIGHT_GREY),

        ("BOTTOMPADDING",(0,0),(-1,-1),8),

        ("TOPPADDING",(0,0),(-1,-1),8),

    ]))

    story.append(summary)

    story.append(Spacer(1,25))

    # --------------------------------------------------------

    story.append(
        Paragraph(
            "KEY TAKEAWAYS",
            heading1_style
        )
    )

    takeaways = [

        "DCF valuation is automatically calculated.",

        "Industry comparison uses peer companies.",

        "Altman Z-Score and Piotroski F-Score evaluate financial health.",

        "Recent news is incorporated into the AI analysis.",

        "Recommendation is generated using quantitative and qualitative analysis."

    ]

    for item in takeaways:

        story.append(
            Paragraph(
                "• " + item,
                normal_style
            )
        )

    story.append(Spacer(1,20))

    # --------------------------------------------------------

    story.append(
        Paragraph(
            "AI DISCLOSURE",
            heading1_style
        )
    )

    disclosure = """
This report was generated using Company Analyzer AI.

Financial statements, valuation models,
industry analysis, financial health metrics,
and recent news are combined using an
AI-powered workflow built with LangGraph
and Google Gemini.

Although extensive validation is performed,
all investment decisions should be verified
using independent research.
"""

    story.append(
        Paragraph(
            disclosure,
            normal_style
        )
    )

    story.append(Spacer(1,20))

    # --------------------------------------------------------

    story.append(
        Paragraph(
            "DISCLAIMER",
            heading1_style
        )
    )

    disclaimer = """
This report is intended solely for educational
and research purposes.

Nothing contained in this report constitutes
investment advice, financial advice,
or a recommendation to buy or sell securities.

Always perform your own due diligence.
"""

    story.append(
        Paragraph(
            disclaimer,
            normal_style
        )
    )

    story.append(PageBreak())
def build_appendix(story, results):
    """
    Appendix
    """

    metrics = results["metrics"]
    valuation = results["valuation"]
    industry = results["industry"]

    altman = results["altman"]
    piotroski = results["piotroski"]

    story.append(
        Paragraph(
            "APPENDIX",
            title_style
        )
    )

    story.append(Spacer(1,15))

    # ====================================================
    # FINANCIAL METRICS
    # ====================================================

    story.append(
        Paragraph(
            "Financial Metrics",
            heading1_style
        )
    )

    rows = [["Metric", "Value"]]

    for key, value in metrics.items():

        rows.append([str(key), str(value)])

    table = Table(
        rows,
        colWidths=[220,260]
    )

    table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,0),NAVY),

        ("TEXTCOLOR",(0,0),(-1,0),WHITE),

        ("GRID",(0,0),(-1,-1),0.4,LIGHT_GREY),

        ("BOTTOMPADDING",(0,0),(-1,-1),6),

        ("TOPPADDING",(0,0),(-1,-1),6),

    ]))

    story.append(table)

    story.append(Spacer(1,20))

    # ====================================================
    # DCF
    # ====================================================

    story.append(
        Paragraph(
            "DCF Valuation",
            heading1_style
        )
    )

    dcf_rows = [

        ["Metric","Value"],

        ["Current Price", currency(valuation.get("current_price"))],

        ["Intrinsic Value", currency(valuation.get("intrinsic_value"))],

        ["Margin of Safety", percentage(valuation.get("margin_of_safety"))],

        ["Recommendation", valuation.get("valuation")]

    ]

    dcf = Table(
        dcf_rows,
        colWidths=[220,260]
    )

    dcf.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,0),BLUE),

        ("TEXTCOLOR",(0,0),(-1,0),WHITE),

        ("GRID",(0,0),(-1,-1),0.4,LIGHT_GREY),

    ]))

    story.append(dcf)

    story.append(Spacer(1,20))

    # ====================================================
    # INDUSTRY SCORES
    # ====================================================

    story.append(
        Paragraph(
            "Industry Analysis",
            heading1_style
        )
    )

    industry_rows = [

        ["Score","Value"],

        ["Growth", industry.get("growthscore")],

        ["Profitability", industry.get("profitabilityscore")],

        ["Risk", industry.get("riskscore")],

        ["Valuation", industry.get("valuationscore")]

    ]

    industry_table = Table(
        industry_rows,
        colWidths=[220,260]
    )

    industry_table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,0),BLUE),

        ("TEXTCOLOR",(0,0),(-1,0),WHITE),

        ("GRID",(0,0),(-1,-1),0.4,LIGHT_GREY),

    ]))

    story.append(industry_table)

    story.append(Spacer(1,20))

    # ====================================================
    # FINANCIAL HEALTH
    # ====================================================

    story.append(
        Paragraph(
            "Financial Health",
            heading1_style
        )
    )

    health = [

        ["Metric","Value"],

        ["Altman Z Score", str(altman)],

        ["Piotroski F Score", str(piotroski)]

    ]

    health_table = Table(
        health,
        colWidths=[220,260]
    )

    health_table.setStyle(TableStyle([

        ("BACKGROUND",(0,0),(-1,0),NAVY),

        ("TEXTCOLOR",(0,0),(-1,0),WHITE),

        ("GRID",(0,0),(-1,-1),0.4,LIGHT_GREY),

    ]))

    story.append(health_table)

    story.append(Spacer(1,25))

    # ====================================================

    story.append(divider())

    story.append(Spacer(1,10))

    story.append(
        Paragraph(
            f"Generated on {datetime.now().strftime('%d %B %Y %H:%M')}",
            small_style
        )
    )
def pdf(results):

    doc = SimpleDocTemplate(
        f"{results['ticker']}_Report.pdf",
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=45,
    )

    story = []

    build_cover(story, results)

    build_dashboard(story, results)

    build_visuals(story)

    build_ai_report(story, results)

    build_summary(story, results)

    build_appendix(story, results)

    doc.build(
        story,
        onFirstPage=page_footer,
        onLaterPages=page_footer
    )

    print(f"{results['ticker']}_Report.pdf created successfully.")