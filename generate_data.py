"""
Generate a SIMULATED online-retail transactions dataset.

The data is synthetic (not real customers). It mirrors the structure of a typical
e-commerce order export and deliberately includes real-world data quality problems
so the cleaning step has something to fix:
  - duplicate rows
  - cancelled orders / returns (negative quantity, invoice starting with 'C')
  - missing CustomerID
  - zero or negative prices
  - inconsistent text (extra spaces, mixed case) in product and country names

Run:  python data/generate_data.py
Output: data/online_retail_simulated.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = Path(__file__).parent / "online_retail_simulated.csv"

N_CUSTOMERS = 3_000
START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")

countries = ["United Kingdom", "Germany", "France", "Netherlands", "Ireland",
             "Spain", "Belgium", "Nigeria", "United States", "Australia"]
country_p = [0.55, 0.09, 0.08, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.02]

categories = {
    "Home Decor": (4.0, 25.0), "Kitchenware": (3.0, 18.0), "Stationery": (0.8, 6.0),
    "Gifts": (2.5, 15.0), "Lighting": (6.0, 40.0), "Textiles": (3.5, 22.0),
}
products = []
for cat, (lo, hi) in categories.items():
    for i in range(40):
        products.append({
            "StockCode": f"{cat[:3].upper()}{1000 + i}",
            "Description": f"{cat} Item {i + 1:02d}",
            "Category": cat,
            "BasePrice": round(rng.uniform(lo, hi), 2),
        })
products = pd.DataFrame(products)
# a few best-sellers get much higher popularity (long-tail demand)
pop = np.clip(rng.pareto(2.0, len(products)) + 1, 1, 8)
products["pop"] = pop / pop.sum()

# customers: sign-up date, country, and a latent "loyalty" that drives order frequency
cust = pd.DataFrame({
    "CustomerID": np.arange(12001, 12001 + N_CUSTOMERS),
    "Country": rng.choice(countries, N_CUSTOMERS, p=country_p),
    "first": START + pd.to_timedelta(rng.integers(0, (END - START).days - 30, N_CUSTOMERS), "D"),
    "rate": rng.gamma(0.6, 1.6, N_CUSTOMERS),  # expected orders per month, skewed
    "churn_after": rng.exponential(220, N_CUSTOMERS),  # days until customer lapses
})

# seasonality: stronger Q4 (holiday) demand
month_boost = {1: 0.8, 2: 0.8, 3: 0.9, 4: 0.9, 5: 0.95, 6: 0.95,
               7: 0.9, 8: 0.95, 9: 1.1, 10: 1.25, 11: 1.6, 12: 1.4}

rows, inv = [], 536000
for c in cust.itertuples():
    active_end = min(END, c.first + pd.Timedelta(days=int(c.churn_after) + 30))
    months = max(1, (active_end - c.first).days / 30)
    n_orders = max(1, rng.poisson(c.rate * months * 0.5))
    days = np.sort(rng.integers(0, max(1, (active_end - c.first).days), n_orders))
    days[0] = 0
    for d in days:
        date = c.first + pd.Timedelta(days=int(d))
        if rng.random() > month_boost[date.month] / 1.6 and d != 0:
            continue
        date += pd.Timedelta(minutes=int(rng.integers(8 * 60, 20 * 60)))
        inv += 1
        n_lines = rng.integers(1, 12)
        pick_idx = np.unique(rng.choice(len(products), n_lines, p=products["pop"].values))
        for p in products.iloc[pick_idx].itertuples():
            rows.append([str(inv), p.StockCode, p.Description, int(rng.integers(1, 13)),
                         date, round(p.BasePrice * rng.uniform(0.9, 1.1), 2), c.CustomerID, c.Country])

df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                 "InvoiceDate", "UnitPrice", "CustomerID", "Country"])

# ---- inject data-quality problems -------------------------------------------------
n = len(df)
# returns / cancellations (~2%)
ret = df.sample(frac=0.02, random_state=1).copy()
ret["InvoiceNo"] = "C" + ret["InvoiceNo"]
ret["Quantity"] = -ret["Quantity"]
ret["InvoiceDate"] += pd.Timedelta(days=3)
df = pd.concat([df, ret])
# missing CustomerID (~4%)
df.loc[df.sample(frac=0.04, random_state=2).index, "CustomerID"] = np.nan
# zero / negative prices (~0.3%)
df.loc[df.sample(frac=0.003, random_state=3).index, "UnitPrice"] = rng.choice([0.0, -1.0])
# messy text (~5%)
idx = df.sample(frac=0.05, random_state=4).index
df.loc[idx, "Description"] = "  " + df.loc[idx, "Description"].str.upper() + " "
idx = df.sample(frac=0.03, random_state=5).index
df.loc[idx, "Country"] = df.loc[idx, "Country"].str.lower() + " "
# exact duplicates (~1%)
df = pd.concat([df, df.sample(frac=0.01, random_state=6)])

df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df["InvoiceDate"] = df["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M")
df.to_csv(OUT, index=False)
print(f"Wrote {len(df):,} rows ({n:,} clean base rows) to {OUT}")
