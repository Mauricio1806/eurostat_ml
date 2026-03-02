import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "output")

CSV_PATH = os.path.join(OUT_DIR, "gold_model_dataset.csv")
REPORT_JSON = os.path.join(OUT_DIR, "ml_report.json")
PRED_CSV = os.path.join(OUT_DIR, "predictions_holdout.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Não achei: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# -------------------------
# COLUNAS ESPERADAS
# -------------------------
required = ["sector", "year", "target_value_added", "cloud_intensity", "cloud_intensity_sector"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Faltam colunas no dataset: {missing}\nColunas disponíveis: {df.columns.tolist()}")

# -------------------------
# LIMPEZA MÍNIMA
# -------------------------
df = df.dropna(subset=["sector", "year", "target_value_added"]).copy()

# força tipos
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["target_value_added"] = pd.to_numeric(df["target_value_added"], errors="coerce")
df["cloud_intensity"] = pd.to_numeric(df["cloud_intensity"], errors="coerce")
df["cloud_intensity_sector"] = pd.to_numeric(df["cloud_intensity_sector"], errors="coerce")

df = df.dropna(subset=["year", "target_value_added"])

# -------------------------
# FEATURES
# -------------------------
X = df[["sector", "year", "cloud_intensity", "cloud_intensity_sector"]]
y = df["target_value_added"]

# -------------------------
# PREPROCESS (IMPUTE + OHE)
# -------------------------
numeric_features = ["year", "cloud_intensity", "cloud_intensity_sector"]
categorical_features = ["sector"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# -------------------------
# SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# MODELO 1: LinearRegression
# -------------------------
lr = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LinearRegression())
])

lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# -------------------------
# MODELO 2: XGBoost (robusto, melhor pro TCC)
# -------------------------
xgb_model = xgb.XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

xgb_pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", xgb_model)
])

xgb_pipe.fit(X_train, y_train)
pred_xgb = xgb_pipe.predict(X_test)

# -------------------------
# MÉTRICAS
# -------------------------
def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}

m_lr = metrics(y_test, pred_lr)
m_xgb = metrics(y_test, pred_xgb)

report = {
    "rows_total": int(len(df)),
    "rows_train": int(len(X_train)),
    "rows_test": int(len(X_test)),
    "nan_cloud_intensity_before_impute": int(pd.isna(df["cloud_intensity"]).sum()),
    "nan_cloud_intensity_sector_before_impute": int(pd.isna(df["cloud_intensity_sector"]).sum()),
    "model_linear_regression": m_lr,
    "model_xgboost": m_xgb
}

with open(REPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# salva previsões do holdout
out = X_test.copy()
out["y_true"] = y_test.values
out["pred_lr"] = pred_lr
out["pred_xgb"] = pred_xgb
out.to_csv(PRED_CSV, index=False)

print("\n[OK] Treino concluído.")
print("[OK] Relatório:", REPORT_JSON)
print("[OK] Previsões holdout:", PRED_CSV)

print("\n=== MÉTRICAS (HOLDOUT) ===")
print("LinearRegression:", m_lr)
print("XGBoost:", m_xgb)

print("\n[INFO] NaNs (antes do imputer):")
print(" - cloud_intensity:", report["nan_cloud_intensity_before_impute"])
print(" - cloud_intensity_sector:", report["nan_cloud_intensity_sector_before_impute"])
