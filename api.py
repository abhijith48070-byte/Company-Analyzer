"""REST API.  uvicorn api:app --reload   (docs at http://127.0.0.1:8000/docs)"""
import json
from functools import lru_cache
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

import analyze as az
import data
import discovery
import thesis as th
from tracker import Tracker

app = FastAPI(title="Company Analyzer", version="3.0")
TRACKER, PRIVATE = Tracker("tracker.db"), discovery.PrivateCompanies("private_companies.json")
_cache = {}


def _run(ticker, **kw):
    key = (ticker.upper(), tuple(sorted(kw.items())))
    if key not in _cache:
        t = data.find_company(ticker) if not ticker.isupper() else ticker
        if not t:
            raise HTTPException(404, "company not found")
        _cache[key] = az.analyze(t, **kw)
    return _cache[key]


def _clean(o):
    return json.loads(json.dumps(o, default=str))


@app.get("/analyze/{ticker}")
def analyze(ticker: str, peers: bool = False, edgar: bool = False, ml: bool = False, competitors: bool = False, events: bool = False):
    r = _run(ticker, peers=peers, edgar=edgar, do_ml=ml, do_comp=competitors, do_events=events)
    return _clean({k: v for k, v in r.items() if k not in ("periods",)})


@app.get("/thesis/{ticker}", response_class=PlainTextResponse)
def thesis(ticker: str):
    r = _run(ticker, peers=True, edgar=True, do_ml=True, do_comp=True, do_events=True, do_thesis=True)
    return th.render_thesis(r["thesis"], r["metrics"].get("Name") or ticker)


class Ask(BaseModel):
    ticker: str
    question: str
    llm: str = ""


@app.post("/ask")
def ask(a: Ask):
    r = _run(a.ticker, edgar=False, questions=(a.question,), llm=a.llm)
    return _clean(r["rag"][0])


class Screen(BaseModel):
    filters: str
    tickers: Optional[List[str]] = None
    limit: int = 25


@app.post("/screen")
def screen(s: Screen):
    sc = discovery.Screener().load(s.tickers or data.SP500["Symbol"].tolist())
    df = sc.screen(s.filters, limit=s.limit)
    return df.to_dict("records")


class Transaction(BaseModel):
    kind: str
    acquirer: str
    target: str
    date: str = ""
    value_usd: Optional[float] = None
    rationale: str = ""
    source: str = ""
    verified: bool = False


@app.get("/transactions")
def transactions(company: Optional[str] = None):
    return {"transactions": TRACKER.transactions(company), "investments": TRACKER.investments(company)}


@app.post("/transactions")
def add_transaction(t: Transaction):
    TRACKER.add_transaction(t.kind, t.acquirer, t.target, t.date, t.value_usd, None, t.rationale, "completed", t.source, t.verified)
    return {"ok": True}


@app.get("/graph/{company}")
def graph(company: str, depth: int = 2):
    g = TRACKER.graph(company, depth)
    return {"nodes": list(g.nodes), "edges": [{"from": a, "to": b, **d} for a, b, d in g.edges(data=True)]}


@app.get("/private")
def private(q: str = "", industry: Optional[str] = None):
    return PRIVATE.search(q, industry)


class Round(BaseModel):
    name: str
    date: str
    series: str
    amount_usd: Optional[float] = None
    valuation_usd: Optional[float] = None
    investors: List[str] = []
    source: str = ""


@app.post("/private/round")
def private_round(r: Round):
    PRIVATE.add_round(r.name, r.date, r.series, r.amount_usd, r.valuation_usd, r.investors, r.source)
    return PRIVATE.summary(r.name)
