from graph import app
from report import create_graphs, pdf

print("Starting main.py")


def main():
    from data import find_company

    company = input("Enter Company or Ticker: ")

    ticker = find_company(company)

    if ticker is None:
        print("Company not found.")
        return

    print(f"\nSelected: {ticker}\n")

    result = app.invoke({"ticker": ticker})

    create_graphs(result)

    pdf(result)

    print("\n==============================")
    print("GROUNDING CHECK")
    print("==============================")
    print(result["evaluation"]["discrepancy_check"])


if __name__ == "__main__":
    main()