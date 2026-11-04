import os, tempfile
from rag import FilingIndex
from tracker import *
tenk_old = "Item 1A. Risk Factors\n\nWe depend on a single foundry partner for most manufacturing capacity and supply disruptions could harm results.\n\nItem 1B. Unresolved\n\nNone"
tenk_new = ("Item 1A. Risk Factors\n\nWe depend on a single foundry partner for most manufacturing capacity and supply disruptions could harm results.\n\n"
            "Export controls restrict sales of our data center products to China, which represented 13% of revenue last year and could reduce revenue materially.\n\n"
            "Item 1B. Unresolved\n\nNone")
idx = FilingIndex()
idx.add_filing({"form": "10-K", "date": "2026-02-20", "url": "https://sec.gov/x"}, tenk_new)
idx.add_document("Yahoo News", "Shares rose after the company reported record data center revenue of 41 billion dollars.", "2026-05-01", "https://news/x")
r = idx.answer("What are the export control risks in China?")
print(r["answer"][:200]); print(r["sources"][0]["citation"], r["unsupported_numbers"])
llm = lambda p: "China was 13% of revenue [1] and revenue was 999 billion [1]."
r2 = idx.answer("China revenue share?", llm=llm); print(r2["unsupported_numbers"], r2["bad_citations"])
import edgar
d = edgar.compare_10ks({"form": "10-K", "date": "2025", "url": "u1"}, {"form": "10-K", "date": "2026", "url": "u2"}, tenk_old, tenk_new, items=("1A",))
print([n["text"][:60] for n in d[0]["new"]])
p = os.path.join(tempfile.mkdtemp(), "t.db"); t = Tracker(p)
t.add_transaction("acquisition", "NVIDIA", "Mellanox", "2020-04-27", 6.9e9, 20.0, "networking for data centers", source="8-K 2020-04-27", verified=True)
t.add_investment("NVIDIA", "Arm Holdings", "2023-09-14", None, None, "IPO anchor", "13F")
t.add_relationship("Vanguard", "NVIDIA", "holds"); t.add_relationship("Mellanox", "Mellanox Technologies Ltd", "subsidiary_of")
print(t.transactions("NVIDIA")[0]["target"], t.precedent_multiples(), list(t.graph("NVIDIA").edges)[:3])
print(t.graph_html(os.path.join(os.path.dirname(p), "g.html"), "NVIDIA"))
print(extract_deals_from_8k("On May 1, the Company announced the acquisition of Acme Robotics, Inc. for approximately $1.2 billion in cash.", {"form": "8-K", "date": "2026-05-01"})[0]["value_usd"])
print(parse_exhibit21("Mellanox Technologies Ltd    Israel\nNVIDIA Singapore Pte. Ltd.   Singapore\nRandom heading"))
print(parse_13f_infotable("<informationTable><infoTable><nameOfIssuer>APPLE INC</nameOfIssuer><cusip>037833100</cusip><value>1000</value><shrsOrPrnAmt><sshPrnamt>50</sshPrnamt></shrsOrPrnAmt></infoTable></informationTable>"))
print("OK")
