from rag import FilingIndex
ix=FilingIndex()
ix.add_document("10-K 2024","Operating margin was 31.2% in fiscal 2024 compared to 28.5% in fiscal 2023. Revenue was $60.9 billion in fiscal 2024, driven by data center demand.",date="2024",kind="filing")
ix.add_document("Blog","Revenue was 75 billion in fiscal 2024 according to rumours. Weather was nice.",date="2024",kind="news")
ix.build()
h=ix.search("operating margin fiscal 2024",k=3); print([x["form"] for x in h])
a="Operating margin was 31.2% in fiscal 2024 [1]. Revenue was $61 billion in fiscal 2024 [1]. Operating margin was 45% in fiscal 2024 [1]. Operating margin was 31.2% in fiscal 2022 [1]."
for c in ix.verify_claims(a,h[:1] if h[0]["form"]=="10-K 2024" else h): print(c["status"],"|",c["claim"][:50],"|",c["reason"])
print(ix.detect_contradictions(ix.chunks))
assert ix.verify_claims("Operating margin was 45% in fiscal 2024 [1].",[ix.chunks[0]])[0]["status"]=="unsupported"
assert ix.verify_claims("Operating margin was 31.2% in fiscal 2024 [1].",[ix.chunks[0]])[0]["status"]=="supported"
assert ix.verify_claims("Operating margin was 31.2% in fiscal 2022 [1].",[ix.chunks[0]])[0]["status"]=="weak"
assert ix.verify_claims("Revenue was $61 billion in fiscal 2024 [1].",[ix.chunks[0]])[0]["status"]=="supported"
print("RAG2 OK")
# --- multi-document "why" answer
import fin_model
ix2 = FilingIndex()
per = [dict(date="2024-12-31", revenue=1000., gross_profit=400., ebit=150., net_income=100.), dict(date="2023-12-31", revenue=900., gross_profit=420., ebit=190., net_income=130.)]
ix2.add_statements(per, "ACME")
ix2.add_filing({"form": "10-K", "date": "2024-12-31", "url": "u"}, "Operating margin declined as higher input costs and increased research spending outpaced revenue growth.")
ix2.add_transcript("Q4 call", "CFO: We saw operating margin pressure from freight costs and one-time restructuring charges.", "2025-02-01")
ix2.add_document("News", "Analysts say ACME margins slipped.", kind="news"); ix2.build()
e = ix2.explain("Why did ACME operating margin decline and what did management say?")
kinds = {s["kind"] for s in e["sources"]}; print(kinds); assert {"statement", "filing", "transcript"} <= kinds
assert any("Operating margin FY2024 was 15.0%" in c["text"] and "-6.1 pp" in c["text"] for c in ix2.chunks if c["kind"] == "statement")
assert FilingIndex.verify_claims("ACME operating margin was 15.0% in FY2024 [1].", [c for c in ix2.chunks if c["kind"] == "statement" and "Operating margin FY2024" in c["text"]])[0]["status"] == "supported"
print("RAG multi-doc OK")
