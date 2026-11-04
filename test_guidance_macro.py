import guidance as G, macro as M, numpy as np
from rag import FilingIndex
r1 = "The Company expects revenue of $20.0 billion to $21.0 billion. We expect gross margin of 70% to 72%. We expect operating expenses of approximately $3.0 billion. Revenue grew 10%."
r2 = "Outlook: revenue is expected to be between $22 and $23 billion... Third quarter revenue is expected to be $22.0 billion to $23.0 billion, plus or minus 2%. Gross margins are expected to be 70% to 72%. We expect operating expenses of approximately $3.1 billion."
g1, g2 = G.extract_guidance(r1), G.extract_guidance(r2); print(g1, g2, sep="\n")
d = {x["metric"]: x for x in g1}; assert d["revenue"]["low"] == 20e9 and d["revenue"]["high"] == 21e9 and d["gross_margin"]["high"] == 72 and d["operating_expenses"]["low"] == 3e9
ch = G.compare_guidance(g1, g2); print(ch); lab = {c["metric"]: c["direction"] for c in ch}
assert lab["revenue"] == "raised" and lab["gross_margin"] == "maintained"
assert G.classify_guidance_event(ch) == "guidance_raise"
ev = G.guidance_history([{"date": "2024-05-01", "text": r1, "url": "u"}, {"date": "2024-08-01", "text": r2, "url": "u"}]); assert ev[0]["category"] == "guidance_raise"
lower = G.guidance_history([{"date": "2024-05-01", "text": r2, "url": ""}, {"date": "2024-08-01", "text": r1, "url": ""}]); assert lower[0]["category"] == "guidance_lower"
assert G.extract_guidance("Preliminary second quarter revenue of $6.70 billion versus outlook of $8.10 billion.") == []
assert G.extract_guidance("For the second quarter, we expect revenue of $45.0 billion, plus or minus 2%.")[0]["low"] == 45e9
# FRED parsing + changes + scenario replay (offline fake fetch)
def fake(url):
    sid = url.split("id=")[1]
    base = {"WPU101": ("2024-12-01", 300.0, 261.0), "DGS10": ("2024-12-01", 4.5, 3.9), "DHHNGSP": ("2024-12-01", 3.0, 2.0)}[sid]
    return f"DATE,{sid}\n2023-12-01,{base[2]}\n2024-06-01,.\n{base[0]},{base[1]}\n"
rows = M.fetch_series("steel_ppi", fake); print(rows)
c = M.change_over(rows); assert abs(c["pct_change"] - (300 / 261 - 1)) < 1e-9, c
sc = M.scenario_from_history(dict(Revenue=1e11, OperatingMargin=.08, Debt=3e10), "auto", fake); print(sc["observed_shocks"])
assert abs(sc["observed_shocks"]["steel"] - 0.1494) < 1e-3 and abs(sc["observed_shocks"]["rates_bp"] - 60) < 1e-6 and sc["steel"]["delta_pp"] < 0
# embeddings in hybrid search: semantic match without word overlap (LSA needs co-occurrence)
ix = FilingIndex()
for t in ["The firm's profitability weakened as input costs rose sharply and margins compressed.", "Cost inflation squeezed profit margins during the year.",
          "Margins compressed because raw material costs climbed.", "The cafeteria menu now includes vegetarian options.", "Employees enjoy flexible holidays and remote work.",
          "Profit margin squeezed by higher input costs and cost inflation."]:
    ix.add_document("doc", t, kind="filing")
ix.build(); print("embeddings:", ix._embed_kind)
h = ix.search("why did earnings quality weaken from rising costs", k=3); print([x["text"][:40] for x in h])
assert ix._emb is not None and all("cafeteria" not in x["text"] and "holiday" not in x["text"] for x in h)
print("GUIDANCE/MACRO/EMBED OK")
