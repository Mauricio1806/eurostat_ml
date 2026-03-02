import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "output")

CSV_PATH = os.path.join(OUT_DIR, "gold_model_dataset.csv")
OUT_CSV  = os.path.join(OUT_DIR, "forecast_2026_2030.csv")

START_YEAR = 2026
END_YEAR   = 2030
GROWTH = 0.05  # 5% ao ano

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Não achei: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

required = ["sector", "year", "target_value_added", "cloud_intensity", "cloud_intensity_sector"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Faltam colunas: {missing}")

# tipos
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["target_value_added"] = pd.to_numeric(df["target_value_added"], errors="coerce")
df["cloud_intensity"] = pd.to_numeric(df["cloud_intensity"], errors="coerce")
df["cloud_intensity_sector"] = pd.to_numeric(df["cloud_intensity_sector"], errors="coerce")
df = df.dropna(subset=["sector", "year", "target_value_added"]).copy()

# ---------- treino (XGBoost) ----------
X = df[["sector", "year", "cloud_intensity", "cloud_intensity_sector"]]
y = df["target_value_added"]

numeric_features = ["year", "cloud_intensity", "cloud_intensity_sector"]
categorical_features = ["sector"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    ]
)

model = xgb.XGBRegressor(
    n_estimators=900,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

pipe.fit(X, y)

# ---------- base cloud (último ano disponível) ----------
last_year = int(df["year"].max())

# Pega uma linha do último ano (para base de cloud)
base_row = df[df["year"] == last_year][["cloud_intensity", "cloud_intensity_sector"]].head(1)
if base_row.empty:
    # fallback absoluto (se der algum problema)
    base_ci = float(df["cloud_intensity"].median())
    base_cis = float(df["cloud_intensity_sector"].median())
else:
    base_ci  = float(base_row["cloud_intensity"].iloc[0]) if pd.notna(base_row["cloud_intensity"].iloc[0]) else float(df["cloud_intensity"].median())
    base_cis = float(base_row["cloud_intensity_sector"].iloc[0]) if pd.notna(base_row["cloud_intensity_sector"].iloc[0]) else float(df["cloud_intensity_sector"].median())

sectors = sorted(df["sector"].dropna().unique().tolist())

rows = []
for sector in sectors:
    for year in range(START_YEAR, END_YEAR + 1):
        years_ahead = year - last_year
        ci_sim  = base_ci * ((1 + GROWTH) ** years_ahead)
        cis_sim = base_cis * ((1 + GROWTH) ** years_ahead)

        X_pred = pd.DataFrame([{
            "sector": sector,
            "year": year,
            "cloud_intensity": ci_sim,
            "cloud_intensity_sector": cis_sim
        }])

        pred = float(pipe.predict(X_pred)[0])

        rows.append({
            "sector": sector,
            "year": year,
            "cloud_intensity_sim": ci_sim,
            "cloud_intensity_sector_sim": cis_sim,
            "pred_target_value_added": pred
        })

out = pd.DataFrame(rows).sort_values(["sector", "year"])
out.to_csv(OUT_CSV, index=False)

print(f"[OK] Simulação salva: {OUT_CSV}")
print(out.head(20).to_string(index=False))
print(f"\n[INFO] Obs: {len(out)} | setores: {out['sector'].nunique()} | anos simulados: {out['year'].nunique()}")
