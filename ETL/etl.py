import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_pickle(RAW)
log = []


def step(name, before, after, note):
    log.append({"Step": len(log) + 1, "Rule": name, "Rows Before": int(before), "Rows After": int(after),
                "Rows Removed": int(before - after), "Note": note})


n0 = len(df)
step("Raw file", n0, n0, "UCI Online Retail II, Dec 2009 - Dec 2011")

df = df.drop_duplicates()
step("Remove exact duplicate rows", n0, len(df), "Identical invoice lines entered twice")

n1 = len(df)
df = df[df["StockCode"].str.match(r"^\d{5}[A-Za-z]{0,2}$", na=False)]
step("Keep real products only", n1, len(df), "Drops postage, bank charges, Amazon fees, bad-debt and manual adjustments, tests, vouchers")

df = df[~df["Invoice"].str.startswith("A")]

df["StockCode"] = df["StockCode"].str.upper()
df["CustomerID"] = df["CustomerID"].fillna(-1).astype(int)
df["Description"] = df["Description"].fillna("").str.strip().str.upper()
is_return = df["Invoice"].str.startswith("C")
returns = df[is_return & (df["Quantity"] < 0) & (df["Price"] > 0)].copy()
sales = df[~is_return].copy()

n3 = len(sales)
sales = sales[(sales["Quantity"] > 0) & (sales["Price"] > 0)]
step("Split cancellations into Returns table", len(df), n3, f"{len(returns):,} cancellation lines kept separately to compute net revenue")
step("Remove zero/negative quantity or price", n3, len(sales), "Stock write-offs and free samples without a sale price")

sales["Quantity"] = sales["Quantity"].astype(int)
sales["Revenue"] = (sales["Quantity"] * sales["Price"]).round(2)
returns["ReturnValue"] = (-returns["Quantity"] * returns["Price"]).round(2)
returns["ReturnQty"] = (-returns["Quantity"]).astype(int)

big = sales[sales["Revenue"] >= 10000].reset_index()
pairs = big.merge(returns.reset_index(), left_on=["CustomerID", "StockCode", "Quantity"],
                  right_on=["CustomerID", "StockCode", "ReturnQty"], suffixes=("_s", "_r"))
pairs = pairs[(pairs["InvoiceDate_r"] - pairs["InvoiceDate_s"]).dt.days.between(0, 30)]
n4 = len(sales)
sales = sales.drop(index=pairs["index_s"])
returns = returns.drop(index=pairs["index_r"])
step("Remove reversed bulk orders", n4, len(sales), f"{len(pairs)} orders over 10K GBP fully cancelled within 30 days (data-entry errors), removed with their cancellations")

for t in (sales, returns):
    t["Date"] = t["InvoiceDate"].dt.normalize()
    t["Hour"] = t["InvoiceDate"].dt.hour
    t["Invoice"] = t["Invoice"].str.lstrip("C").astype(int)

desc = (sales[sales["Description"] != ""].groupby(["StockCode", "Description"]).size().reset_index(name="n")
        .sort_values(["StockCode", "n"], ascending=[True, False]).drop_duplicates("StockCode"))
products = pd.DataFrame({"StockCode": sorted(set(sales["StockCode"]) | set(returns["StockCode"]))})
products = products.merge(desc[["StockCode", "Description"]], on="StockCode", how="left")
products["Description"] = products["Description"].fillna("UNKNOWN ITEM").str.title()

ret_by_prod = returns.groupby("StockCode")["ReturnValue"].sum()
prod_rev = sales.groupby("StockCode")["Revenue"].sum().sub(ret_by_prod, fill_value=0).clip(lower=0)
products["NetRevenueAllTime"] = products["StockCode"].map(prod_rev).fillna(0).round(2)
products = products.sort_values("NetRevenueAllTime", ascending=False).reset_index(drop=True)
total = products["NetRevenueAllTime"].sum()
products["RevenueRank"] = np.arange(1, len(products) + 1)
products["CumulativeShare"] = (products["NetRevenueAllTime"].cumsum() / total).round(6)
products["ABCClass"] = np.select([products["CumulativeShare"] <= 0.8, products["CumulativeShare"] <= 0.95], ["A", "B"], "C")
products["ProductPercentile"] = np.ceil(products["RevenueRank"] / len(products) * 20).astype(int) * 5
products["Label"] = products["Description"].str.slice(0, 34) + " (" + products["StockCode"] + ")"

known = sales[sales["CustomerID"] > 0]
snapshot = sales["Date"].max() + pd.Timedelta(days=1)
first = known.groupby("CustomerID")["Date"].min()
last = known.groupby("CustomerID")["Date"].max()
freq = known.groupby("CustomerID")["Invoice"].nunique()
gross = known.groupby("CustomerID")["Revenue"].sum()
ret_c = returns[returns["CustomerID"] > 0].groupby("CustomerID")["ReturnValue"].sum()
country = known.groupby("CustomerID")["Country"].agg(lambda s: s.value_counts().index[0])
cust = pd.DataFrame({"CustomerID": first.index, "Country": country.values, "FirstPurchase": first.values,
                     "LastPurchase": last.values, "Frequency": freq.values,
                     "Monetary": gross.sub(ret_c, fill_value=0).reindex(first.index).values})
cust["Monetary"] = cust["Monetary"].round(2)
cust["RecencyDays"] = (snapshot - cust["LastPurchase"]).dt.days
cust["CohortMonth"] = cust["FirstPurchase"].dt.to_period("M").dt.to_timestamp()
cohort_n = cust.groupby("CohortMonth")["CustomerID"].transform("count")
cust["CohortLabel"] = cust["CohortMonth"].dt.strftime("%b %Y") + "  (n=" + cohort_n.astype(str) + ")"
cust["CohortKey"] = cust["CohortMonth"].dt.year * 100 + cust["CohortMonth"].dt.month
cust["Market"] = np.where(cust["Country"] == "United Kingdom", "United Kingdom", "International")


def score(series, ascending):
    ranked = series.rank(method="first", ascending=ascending)
    return pd.qcut(ranked, 5, labels=[1, 2, 3, 4, 5]).astype(int)


cust["R"] = score(cust["RecencyDays"], ascending=False)
cust["F"] = score(cust["Frequency"], ascending=True)
cust["M"] = score(cust["Monetary"], ascending=True)
cust["RFMScore"] = cust["R"].astype(str) + cust["F"].astype(str) + cust["M"].astype(str)
fm = (cust["F"] + cust["M"]) / 2


def segment(r, f, m, fm_avg):
    result = "Hibernating"
    if r >= 4 and fm_avg >= 4:
        result = "Champions"
    elif r <= 2 and fm_avg >= 4.5:
        result = "Can't Lose Them"
    elif r <= 2 and fm_avg >= 3:
        result = "At Risk"
    elif r >= 3 and fm_avg >= 3:
        result = "Loyal Customers"
    elif r >= 4 and f == 1:
        result = "New Customers"
    elif r >= 4:
        result = "Potential Loyalists"
    elif r == 3:
        result = "Need Attention"
    elif r == 1 and fm_avg <= 1.5:
        result = "Lost"
    return result


cust["Segment"] = [segment(r, f, m, a) for r, f, m, a in zip(cust["R"], cust["F"], cust["M"], fm)]
cust = cust.sort_values("Monetary", ascending=False).reset_index(drop=True)
cust["ValueRankPct"] = ((np.arange(1, len(cust) + 1)) / len(cust)).round(6)
cust["IsTop20Pct"] = cust["ValueRankPct"] <= 0.2
guest = pd.DataFrame([{"CustomerID": -1, "Country": "Unknown", "Market": "Unknown", "Segment": "Guest (no customer ID)", "IsTop20Pct": False}])
customers = pd.concat([cust, guest], ignore_index=True)
for col in ["RecencyDays", "Frequency", "R", "F", "M", "CohortKey"]:
    customers[col] = customers[col].astype("Int64")

segments = pd.DataFrame([
    ("Champions", 1, "Bought recently, buy often and spend the most", "VIP rewards, early access to new ranges, reviews"),
    ("Loyal Customers", 2, "Buy regularly with solid spend", "Upsell higher-value products, loyalty programme"),
    ("Potential Loyalists", 3, "Recent customers with a few orders", "Offer membership, recommend related products"),
    ("New Customers", 4, "First order very recently", "Onboarding sequence, second-order incentive"),
    ("Need Attention", 5, "Average recency, low frequency and spend", "Time-limited offers based on past purchases"),
    ("At Risk", 6, "Used to buy often and spend well, not lately", "Personalised win-back campaign"),
    ("Can't Lose Them", 7, "Former top spenders who have gone quiet", "Direct outreach, account manager call"),
    ("Hibernating", 8, "Low recency, frequency and spend", "Low-cost reactivation, otherwise deprioritise"),
    ("Lost", 9, "Lowest scores on every dimension", "Exclude from paid campaigns"),
    ("Guest (no customer ID)", 10, "Orders without a customer account", "Encourage account creation at checkout"),
], columns=["Segment", "SegmentOrder", "Definition", "RecommendedAction"])

returns.loc[~returns["CustomerID"].isin(cust["CustomerID"]), "CustomerID"] = -1
cohort_map = cust.set_index("CustomerID")["CohortMonth"]
for t in (sales, returns):
    cm = t["CustomerID"].map(cohort_map)
    period = t["Date"].dt.to_period("M")
    t["MonthsSinceFirst"] = np.where(t["CustomerID"] > 0,
                                     (period.dt.year - cm.dt.year) * 12 + (period.dt.month - cm.dt.month), np.nan)

dates = pd.DataFrame({"Date": pd.date_range("2009-12-01", "2011-12-31", freq="D")})
dates["Year"] = dates["Date"].dt.year
dates["Quarter"] = "Q" + dates["Date"].dt.quarter.astype(str)
dates["MonthNum"] = dates["Date"].dt.month
dates["Month"] = dates["Date"].dt.strftime("%b")
dates["MonthStart"] = dates["Date"].dt.to_period("M").dt.to_timestamp()
dates["YearMonth"] = dates["Date"].dt.strftime("%b %Y")
dates["MonthShort"] = dates["Date"].dt.strftime("%b %y")
dates["YearMonthKey"] = dates["Year"] * 100 + dates["MonthNum"]
dates["WeekdayNum"] = dates["Date"].dt.weekday + 1
dates["Weekday"] = dates["Date"].dt.strftime("%a")
dates["HasSales"] = dates["Date"] <= sales["Date"].max()

month_index = pd.DataFrame({"MonthIndex": range(0, 25)})
month_index["MonthLabel"] = "M" + month_index["MonthIndex"].astype(str)

countries = pd.DataFrame({"Country": sorted(set(sales["Country"]) | set(returns["Country"]) | {"Unknown"})})
countries["Market"] = np.where(countries["Country"] == "United Kingdom", "United Kingdom", "International")

sales["MonthsSinceFirst"] = sales["MonthsSinceFirst"].astype("Int64")
sales_out = sales[["Invoice", "Date", "Hour", "CustomerID", "StockCode", "Country", "Quantity", "Price", "Revenue", "MonthsSinceFirst"]]
returns_out = returns[["Invoice", "Date", "CustomerID", "StockCode", "Country", "ReturnQty", "ReturnValue"]]
cust_out = customers[["CustomerID", "Country", "Market", "FirstPurchase", "LastPurchase", "CohortMonth", "CohortLabel", "CohortKey", "RecencyDays", "Frequency",
                      "Monetary", "R", "F", "M", "RFMScore", "Segment", "ValueRankPct", "IsTop20Pct"]]
prod_out = products[["StockCode", "Description", "Label", "NetRevenueAllTime", "RevenueRank", "CumulativeShare", "ABCClass", "ProductPercentile"]]

fmt = {"date_format": "%Y-%m-%d", "index": False}
sales_out.to_csv(OUT / "fact_sales.csv.gz", compression="gzip", float_format="%.2f", **fmt)
returns_out.to_csv(OUT / "fact_returns.csv", float_format="%.2f", **fmt)
cust_out.to_csv(OUT / "dim_customer.csv", **fmt)
prod_out.to_csv(OUT / "dim_product.csv", **fmt)
dates.to_csv(OUT / "dim_date.csv", **fmt)
countries.to_csv(OUT / "dim_country.csv", **fmt)
segments.to_csv(OUT / "dim_segment.csv", **fmt)
month_index.to_csv(OUT / "dim_month_index.csv", index=False)
pd.DataFrame(log).to_csv(OUT / "etl_log.csv", index=False)

summary = {
    "sales_rows": len(sales_out), "return_rows": len(returns_out), "customers": int(len(cust)), "products": int(len(products)),
    "gross_revenue": round(float(sales["Revenue"].sum()), 2), "returns_value": round(float(returns["ReturnValue"].sum()), 2),
    "snapshot": str(snapshot.date()), "abc_A_products": int((products["ABCClass"] == "A").sum()),
    "top20_customer_share": round(float(cust.loc[cust["IsTop20Pct"], "Monetary"].sum() / cust["Monetary"].sum()), 4),
    "segments": cust["Segment"].value_counts().to_dict(),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
print(pd.DataFrame(log).to_string())
