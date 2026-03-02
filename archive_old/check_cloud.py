import pandas as pd

df = pd.read_csv("gold_model_dataset.csv")
df = df.dropna(subset=["sector", "year", "cloud_intensity"])

last = (
    df.sort_values("year")
      .groupby("sector")
      .tail(1)[["sector", "year", "cloud_intensity"]]
      .sort_values("sector")
)

print(last.to_string(index=False))
