import edgar, ml, numpy as np
# fake companyfacts: 8 fiscal years, restated value for FY2020 revenue, mixed quarterly rows that must be ignored
def rows(vals, flow=True):
    out = []
    for y, v in vals.items():
        r = dict(end=f"{y}-12-31", val=v, form="10-K", fp="FY", filed=f"{y+1}-02-15", accn="a")
        if flow: r["start"] = f"{y}-01-01"
        out.append(r)
    return out
Y = range(2016, 2024)
g = lambda f: {y: f(y) for y in Y}
facts = {"facts": {"us-gaap": {
    "Revenues": {"units": {"USD": rows(g(lambda y: 1000 + 100 * (y - 2016))) + [dict(end="2020-12-31", start="2020-01-01", val=1,form="10-K",fp="FY",filed="2021-01-01"),
                                                                              dict(end="2020-12-31", start="2020-01-01", val=1200, form="10-K", fp="FY", filed="2021-03-01"),
                                                                              dict(end="2020-03-31", start="2020-01-01", val=5, form="10-Q", fp="Q1", filed="2020-05-01")]}},
    "Assets": {"units": {"USD": rows(g(lambda y: 2000 + 50 * (y - 2016)), False)}},
    "Liabilities": {"units": {"USD": rows(g(lambda y: 1000), False)}},
    "OperatingIncomeLoss": {"units": {"USD": rows(g(lambda y: 150))}}}}}
ps = edgar.periods_from_xbrl(facts); print(len(ps), ps[0]["date"], ps[0]["filed"])
assert len(ps) == 8 and ps[0]["date"] == "2023-12-31" and ps[0]["filed"] == "2024-02-15"
rev2020 = next(p for p in ps if p["date"] == "2020-12-31")["revenue"]; assert rev2020 == 1200, rev2020   # latest filing wins; quarterly row ignored
assert ps[0]["total_liabilities"] == 1000 and ps[0]["ebit"] == 150
print("XBRL OK")
