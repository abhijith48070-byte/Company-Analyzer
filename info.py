import os
import requests
import feedparser
from newspaper import Article

GOOGLE_NEWS = (
    "https://news.google.com/rss/search?"
    "q={}&hl=en-US&gl=US&ceid=US:en"
)

HEADERS = {
    "User-Agent": "Company Analyzer (abhijith48070@gmail.com)",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}


class CompanyInfo:
    

    def __init__(self):
        pass

    # -------------------------
    # GOOGLE NEWS
    # -------------------------

    def get_news(self, company, limit=10):

        url = GOOGLE_NEWS.format(company.replace(" ", "+"))

        feed = feedparser.parse(url)

        news = []

        for article in feed.entries[:limit]:

            news.append({
                "source": article.source.title if hasattr(article, "source") else "Unknown",
                "title": article.title,
                "published": article.published,
                "url": article.link,
                "content": self.get_article(article.link)
            })

        return news

    def get_article(self, url):

        try:

            article = Article(url)
            article.download()
            article.parse()

            return article.text

        except Exception:

            return ""
    def get_filing_text(self, cik, accession_number, primary_document):

        cik = str(cik).zfill(10)

        accession = accession_number.replace("-", "")

        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accession}/{primary_document}"
        )

        response = requests.get(url, headers=HEADERS)

        if response.status_code == 200:
            return response.text

        return ""

    # -------------------------
    # SEC API
    # -------------------------

    def get_company_submissions(self, cik):

        """
        Example:
        NVIDIA CIK = 1045810
        Apple CIK = 320193
        Microsoft CIK = 789019
        """

        cik = str(cik).zfill(10)

        url = f"https://data.sec.gov/submissions/CIK{cik}.json"

        response = requests.get(url, headers=HEADERS)

        response.raise_for_status()

        return response.json()

    def get_recent_filings(self, cik, limit=10):

        data = self.get_company_submissions(cik)

        recent = data["filings"]["recent"]

        filings = []
        count=min(limit,len(recent["form"]))
        for i in range(count):
            filings.append({

            "form": recent["form"][i],

            "filing_date": recent["filingDate"][i],

            "accession_number": recent["accessionNumber"][i],

            "primary_document": recent["primaryDocument"][i],

            "text": self.get_filing_text(
                cik,
                recent["accessionNumber"][i],
                recent["primaryDocument"][i]
                )

            })

        return filings
    def get_filing_text(self, cik, accession_number, primary_document):

        cik = str(cik).zfill(10)

        accession = accession_number.replace("-", "")

        url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(cik)}/{accession}/{primary_document}"
        )

        response = requests.get(url, headers=HEADERS)

        print(url)
        print(response.status_code)

        if response.status_code == 200:
            return response.text

        return ""