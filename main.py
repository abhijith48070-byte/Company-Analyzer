from report import (
    analyze_company,
    create_graphs,
    report_agent,
    pdf,
)
from utils import client

print("Starting main.py")

def main():
    from data import find_company

    company = input("Enter Company or Ticker: ")

    ticker = find_company(company)

    if ticker is None:
        print("Company not found.")
        return

    print(f"\nSelected: {ticker}\n")

    results = analyze_company(ticker)

    create_graphs(results)

    report = report_agent(results, client)

    pdf(ticker, report)

if __name__ == "__main__":
    main()