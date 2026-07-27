"""
LangGraph version of the Company Analyzer pipeline.
"""

from typing import TypedDict, List
import time

from langgraph.graph import StateGraph, END

from data import get_company_info, get_financialmetrics, peertickers, News
from analysis import dcfcalc, riskanalysis, Riskanalysis, rating
from utils import client

import time
from IPython.display import Image
# ---------------- Shared State ----------------

class AnalysisState(TypedDict, total=False):
    ticker: str
    stock: object
    metrics: dict
    peer_tickers: list
    valuation: dict
    industry: dict
    altman: dict
    piotroski: dict
    news: List[dict]
    report: str
    evaluation: dict


# ---------------- Nodes ----------------

def node_fetch(state):
    stock, info = get_company_info(state["ticker"])
    metrics = get_financialmetrics(info)

    print(metrics.keys())   # <-- Add this

    peers = peertickers(metrics["Sector"])

    return {
        "stock": stock,
        "metrics": metrics,
        "peer_tickers": peers,
    }


def node_dcf(state: AnalysisState) -> dict:
    print("Running DCF...")
    return {"valuation": dcfcalc(state["metrics"])}


def node_industry(state: AnalysisState) -> dict:
    print("Running Industry Analysis...")
    return {"industry": rating(state["metrics"], state["peer_tickers"])}


def node_risk(state: AnalysisState) -> dict:
    print("Running Risk Analysis...")
    return {
        "altman": riskanalysis(state["stock"]),
        "piotroski": Riskanalysis(state["stock"]),
    }


def node_news(state: AnalysisState) -> dict:
    print("Fetching News...")
    return {"news": News(state["stock"])}


def node_report(state: AnalysisState) -> dict:

    print("Generating Report...")

    prompt = f"""
You are an equity research analyst.

Write a professional investment report using ONLY the supplied information.

METRICS:
{state['metrics']}

DCF:
{state['valuation']}

INDUSTRY:
{state['industry']}

ALTMAN:
{state['altman']}

PIOTROSKI:
{state['piotroski']}

NEWS:
{state['news']}
"""

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            return {
                "report": response.text
            }

        except Exception as e:
            print(f"Report attempt {attempt+1} failed.")
            print(e)

            if attempt < 4:
                time.sleep(5)
            else:
                raise


def node_evaluate(state: AnalysisState) -> dict:

    print("Evaluating Report...")

    prompt = f"""
You are a financial fact-checking AI.

Compare the REPORT against the RAW DATA.

Verify every:

- Financial ratio
- Percentage
- DCF value
- Altman Z Score
- Piotroski Score
- Industry comparison

If everything matches exactly, reply ONLY:

No discrepancies found.

Otherwise list every discrepancy.

RAW DATA

METRICS:
{state['metrics']}

DCF:
{state['valuation']}

INDUSTRY:
{state['industry']}

ALTMAN:
{state['altman']}

PIOTROSKI:
{state['piotroski']}

REPORT:

{state['report']}
"""

    for attempt in range(5):

        try:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            return {
                "evaluation": {
                    "discrepancy_check": response.text
                }
            }

        except Exception as e:

            print(f"Evaluation attempt {attempt+1} failed.")
            print(e)

            if attempt < 4:
                time.sleep(5)
            else:
                raise


# ---------------- Graph ----------------

graph = StateGraph(AnalysisState)

graph.add_node("fetch", node_fetch)
graph.add_node("dcf", node_dcf)
graph.add_node("industry", node_industry)
graph.add_node("risk", node_risk)
graph.add_node("news", node_news)
graph.add_node("report", node_report)
graph.add_node("evaluate", node_evaluate)

graph.set_entry_point("fetch")

graph.add_edge("fetch", "dcf")
graph.add_edge("fetch", "industry")
graph.add_edge("fetch", "risk")
graph.add_edge("fetch", "news")

graph.add_edge("dcf", "report")
graph.add_edge("industry", "report")
graph.add_edge("risk", "report")
graph.add_edge("news", "report")

graph.add_edge("report", "evaluate")
graph.add_edge("evaluate", END)

app = graph.compile()


if __name__ == "__main__":

    result = app.invoke({
        "ticker": "AAPL"
    })

    print("\n")
    print("=" * 80)
    print("REPORT")
    print("=" * 80)
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(result["report"])

    print("\n")
    print("=" * 80)
    print("GROUNDING CHECK")
    print("=" * 80)
    print(result["evaluation"]["discrepancy_check"])
    start = time.time()

    result = app.invoke({"ticker": "AAPL"})

    print(f"Completed in {time.time() - start:.2f} seconds")
    graph_image = app.get_graph().draw_mermaid_png()

    with open("graph.png", "wb") as f:

        f.write(graph_image)