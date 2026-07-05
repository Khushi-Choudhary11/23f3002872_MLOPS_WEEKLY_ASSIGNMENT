import pandas as pd

df = pd.read_csv("data/raw/iris.csv")

df["iris_id"] = range(len(df))
df["event_timestamp"] = "2026-07-05 00:00:00 UTC"

df.to_csv("iris_bigquery.csv", index=False)

print(df.head())
print("Saved iris_bigquery.csv")
