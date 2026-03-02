# forecast_next_year
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

df = pd.read_csv("gold_model_dataset.csv")
df = df.dropna(subset=["value_added_real", "cloud_intensity", "sector", "year"])

X = df[["cloud_intensity", "sector", "year"]]
y = df["value_added_real"]

preprocess = ColumnTransformer(
    transformers=[("sector_ohe", OneHotEncoder(handle_unknown="ignore"), ["sector"])],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

model.fit(X, y)

next_year = int(df["year"].max()) + 1

# pega última cloud_intensity por setor (mais recente)
last_ci = (
    df.sort_values("year")
      .groupby("sector")
      .tail(1)[["sector", "cloud_intensity"]]
      .copy()
)

last_ci["year"] = next_year

pred = model.predict(last_ci[["cloud_intensity", "sector", "year"]])
last_ci["pred_value_added_real"] = pred

last_ci = last_ci.sort_values("pred_value_added_real", ascending=False)
last_ci.to_csv("forecast_next_year.csv", index=False)

print("Saved: forecast_next_year.csv")
print(last_ci.head(10).to_string(index=False))
