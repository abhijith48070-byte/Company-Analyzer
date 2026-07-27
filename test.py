from info import CompanyInfo

info = CompanyInfo()

filings = info.get_recent_filings(1045810, 1)

print(filings[0]["form"])
print(filings[0]["filing_date"])
print(len(filings[0]["text"]))
print(filings[0]["text"][:500])