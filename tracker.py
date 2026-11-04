"""M&A / investment tracker, corporate relationship graph, 13F holdings, Exhibit 21 subsidiaries."""
import json
import re
import sqlite3

import networkx as nx


class Tracker:
    def __init__(self, path="tracker.db"):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS transactions(id INTEGER PRIMARY KEY, kind TEXT, acquirer TEXT, target TEXT, date TEXT,
            value_usd REAL, ev_ebitda REAL, rationale TEXT, status TEXT, source TEXT, verified INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS investments(id INTEGER PRIMARY KEY, investor TEXT, investee TEXT, date TEXT,
            amount_usd REAL, stake_pct REAL, round TEXT, source TEXT);
        CREATE TABLE IF NOT EXISTS relationships(id INTEGER PRIMARY KEY, src TEXT, dst TEXT, kind TEXT, source TEXT);
        """)

    def add_transaction(self, kind, acquirer, target, date="", value_usd=None, ev_ebitda=None, rationale="",
                        status="completed", source="", verified=False):
        """kind: acquisition|merger|divestiture|spin-off|ipo|strategic investment|venture|private equity"""
        self.db.execute("INSERT INTO transactions(kind,acquirer,target,date,value_usd,ev_ebitda,rationale,status,source,verified)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?)", (kind, acquirer, target, date, value_usd, ev_ebitda, rationale, status, source, int(verified)))
        if kind in ("acquisition", "merger"):
            self.add_relationship(acquirer, target, "acquired", source)
        self.db.commit()

    def add_investment(self, investor, investee, date="", amount_usd=None, stake_pct=None, round="", source=""):
        self.db.execute("INSERT INTO investments(investor,investee,date,amount_usd,stake_pct,round,source) VALUES(?,?,?,?,?,?,?)",
                        (investor, investee, date, amount_usd, stake_pct, round, source))
        self.add_relationship(investor, investee, "invested_in", source)
        self.db.commit()

    def add_relationship(self, src, dst, kind, source=""):
        if not self.db.execute("SELECT 1 FROM relationships WHERE src=? AND dst=? AND kind=?", (src, dst, kind)).fetchone():
            self.db.execute("INSERT INTO relationships(src,dst,kind,source) VALUES(?,?,?,?)", (src, dst, kind, source))
            self.db.commit()

    def transactions(self, company=None):
        q, a = "SELECT kind,acquirer,target,date,value_usd,rationale,status,source,verified,ev_ebitda FROM transactions", ()
        if company:
            q += " WHERE acquirer LIKE ? OR target LIKE ?"; a = (f"%{company}%",) * 2
        cols = ["kind", "acquirer", "target", "date", "value_usd", "rationale", "status", "source", "verified", "ev_ebitda"]
        return [dict(zip(cols, r)) for r in self.db.execute(q + " ORDER BY date DESC", a)]

    def investments(self, company=None):
        q, a = "SELECT investor,investee,date,amount_usd,stake_pct,round,source FROM investments", ()
        if company:
            q += " WHERE investor LIKE ? OR investee LIKE ?"; a = (f"%{company}%",) * 2
        cols = ["investor", "investee", "date", "amount_usd", "stake_pct", "round", "source"]
        return [dict(zip(cols, r)) for r in self.db.execute(q, a)]

    def precedent_multiples(self, ):
        return [r[0] for r in self.db.execute("SELECT ev_ebitda FROM transactions WHERE ev_ebitda IS NOT NULL AND verified=1")]

    def graph(self, center=None, depth=2):
        g = nx.DiGraph()
        for s, d, k, src in self.db.execute("SELECT src,dst,kind,source FROM relationships"):
            g.add_edge(s, d, kind=k, source=src)
        if center and center in g:
            nodes = nx.single_source_shortest_path_length(g.to_undirected(), center, cutoff=depth)
            g = g.subgraph(nodes).copy()
        return g

    def graph_html(self, path, center=None, depth=2):
        """Interactive graph (vis-network from cdnjs; open in a browser)."""
        g = self.graph(center, depth)
        colors = {"acquired": "#d1495b", "invested_in": "#2e86ab", "subsidiary_of": "#6a994e", "holds": "#f4a261",
                  "parent_of": "#6a994e", "director_of": "#9b5de5", "partner_of": "#8d99ae"}
        nodes = [{"id": n, "label": n, "color": "#ffb703" if n == center else "#8ecae6"} for n in g.nodes]
        edges = [{"from": a, "to": b, "label": d["kind"], "arrows": "to", "color": colors.get(d["kind"], "#999"),
                  "title": d.get("source") or ""} for a, b, d in g.edges(data=True)]
        html = ("<!doctype html><meta charset=utf-8><title>Corporate graph</title>"
                "<script src='https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js'></script>"
                "<div id=n style='height:92vh;border:1px solid #ccc'></div><script>"
                f"new vis.Network(document.getElementById('n'),{{nodes:new vis.DataSet({json.dumps(nodes)}),"
                f"edges:new vis.DataSet({json.dumps(edges)})}},{{physics:{{stabilization:true}},edges:{{font:{{size:10}}}}}});</script>")
        open(path, "w").write(html)
        return path


# ------------------------------------------------------------------ extraction helpers
_VAL = re.compile(r"\$\s?([\d,.]+)\s?(billion|million|bn|mn|b|m)\b", re.I)
_ACQ = re.compile(r"(?:acqui(?:re|red|res|sition)|merger)\s+(?:of\s+|with\s+)?(?:all of the (?:outstanding )?(?:shares|equity|stock) of\s+)?"
                  r"([A-Z][\w&.,'\- ]{2,60}?)(?=,|\.|;| for | in a | pursuant| from | \()", re.S)


def extract_deals_from_8k(text, meta):
    """Heuristic candidates from 8-K text (Items 1.01/2.01). Always unverified: confirm against the filing."""
    out = []
    for m in _ACQ.finditer(text):
        ctx = text[m.start(): m.end() + 400]
        v = _VAL.search(ctx)
        val = None
        if v:
            num = float(v.group(1).replace(",", "").rstrip("."))
            val = num * (1e9 if v.group(2).lower() in ("billion", "bn", "b") else 1e6)
        out.append({"target": m.group(1).strip(), "value_usd": val, "date": meta.get("date"),
                    "source": f"{meta.get('form')} {meta.get('date')} {meta.get('url', '')}", "verified": False,
                    "context": re.sub(r"\s+", " ", text[max(m.start() - 100, 0): m.end() + 200])})
    return out[:10]


def parse_exhibit21(text):
    """Subsidiary names from an Exhibit 21 list (name + jurisdiction per line)."""
    subs = []
    for line in text.splitlines():
        parts = [p.strip() for p in re.split(r"\s{2,}|\t|\|", line) if p.strip()]
        if parts and 3 < len(parts[0]) < 80 and re.search(r"(inc|llc|ltd|corp|gmbh|s\.?a|b\.?v|plc|limited|co)\b", parts[0], re.I):
            subs.append({"name": parts[0], "jurisdiction": parts[1] if len(parts) > 1 else ""})
    return subs


def parse_13f_infotable(xml):
    """Holdings from a 13F information table XML."""
    from bs4 import BeautifulSoup
    s = BeautifulSoup(xml, "xml")
    rows = []
    for t in s.find_all(re.compile("infoTable$", re.I)):
        g = lambda n: (t.find(n).text.strip() if t.find(n) else None)
        rows.append({"issuer": g("nameOfIssuer"), "cusip": g("cusip"), "value_usd": float(g("value") or 0),
                     "shares": float(g("sshPrnamt") or 0)})
    return sorted(rows, key=lambda r: -r["value_usd"])
