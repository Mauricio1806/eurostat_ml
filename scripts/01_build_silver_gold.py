import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, "output")

BRONZE_GVA   = os.path.join(OUT_DIR, "bronze_gva_de.parquet")
BRONZE_CLOUD = os.path.join(OUT_DIR, "bronze_cloud_de.parquet")

SILVER_GVA   = os.path.join(OUT_DIR, "silver_gva.parquet")
SILVER_CLOUD = os.path.join(OUT_DIR, "silver_cloud.parquet")

GOLD_GVA     = os.path.join(OUT_DIR, "gold_gva.parquet")
GOLD_CLOUD   = os.path.join(OUT_DIR, "gold_cloud.parquet")
GOLD_DATASET = os.path.join(OUT_DIR, "gold_model_dataset.parquet")
GOLD_CSV     = os.path.join(OUT_DIR, "gold_model_dataset.csv")

YEAR_MIN = 2010
YEAR_MAX = 2025

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def pick_col(df: pd.DataFrame, candidates):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        for real in df.columns:
            if c in real:
                return real
    return None

def to_year(series):
    s = series.astype(str).str.extract(r"(\d{4})")[0]
    return pd.to_numeric(s, errors="coerce")

print("[INFO] Lendo Bronze Parquet...")
if not os.path.exists(BRONZE_GVA):
    raise FileNotFoundError(f"Não achei: {BRONZE_GVA}")
if not os.path.exists(BRONZE_CLOUD):
    raise FileNotFoundError(f"Não achei: {BRONZE_CLOUD}")

gva = pd.read_parquet(BRONZE_GVA)
cloud = pd.read_parquet(BRONZE_CLOUD)

gva = norm_cols(gva)
cloud = norm_cols(cloud)

print("[INFO] Colunas GVA:", gva.columns.tolist())
print("[INFO] Colunas CLOUD:", cloud.columns.tolist())

# =========================
# SILVER - GVA
# =========================
sector_col = pick_col(gva, ["nace_r2", "nace", "sector"])
time_col   = pick_col(gva, ["time_period", "time", "year"])
value_col  = pick_col(gva, ["obs_value", "value", "values"])
unit_col   = pick_col(gva, ["unit"])
na_item_col= pick_col(gva, ["na_item"])

if sector_col is None or time_col is None or value_col is None:
    raise ValueError("Não consegui detectar colunas chave no GVA.")

gva_s = gva.copy()
gva_s["year"] = to_year(gva_s[time_col])
gva_s["sector"] = gva_s[sector_col].astype(str).str.strip()

gva_s = gva_s[gva_s["year"].between(YEAR_MIN, YEAR_MAX)]
gva_s = gva_s[~gva_s["sector"].str.upper().isin(["TOTAL", "TOT", "TOTAL_NACE"])]
gva_s = gva_s[pd.to_numeric(gva_s[value_col], errors="coerce").notna()]
gva_s[value_col] = pd.to_numeric(gva_s[value_col], errors="coerce")

if na_item_col is not None:
    gva_s = gva_s[gva_s[na_item_col].astype(str).str.strip().str.upper() == "B1G"]

if unit_col is None:
    gva_s["unit"] = "NA"
else:
    gva_s["unit"] = gva_s[unit_col].astype(str).str.strip()

silver_gva = (
    gva_s[["sector", "year", "unit", value_col]]
      .rename(columns={value_col: "value_added"})
      .dropna(subset=["sector", "year", "value_added"])
      .drop_duplicates(subset=["sector", "year", "unit", "value_added"])
)

silver_gva.to_parquet(SILVER_GVA, index=False)
print(f"[OK] Silver GVA salvo: {SILVER_GVA} | linhas:", len(silver_gva), "| setores:", silver_gva["sector"].nunique())

# =========================
# SILVER - CLOUD
# =========================
time_c  = pick_col(cloud, ["time_period", "time", "year"])
val_c   = pick_col(cloud, ["obs_value", "value", "values"])
unit_c  = pick_col(cloud, ["unit"])
indic_c = pick_col(cloud, ["indic_is", "indic", "indicator"])

if time_c is None or val_c is None:
    raise ValueError("Não consegui detectar colunas chave no CLOUD.")

cloud_s = cloud.copy()
cloud_s["year"] = to_year(cloud_s[time_c])
cloud_s = cloud_s[cloud_s["year"].between(YEAR_MIN, YEAR_MAX)]
cloud_s = cloud_s[pd.to_numeric(cloud_s[val_c], errors="coerce").notna()]
cloud_s[val_c] = pd.to_numeric(cloud_s[val_c], errors="coerce")

cloud_s["unit"] = cloud_s[unit_c].astype(str).str.strip() if unit_c else "NA"
cloud_s["indic"] = cloud_s[indic_c].astype(str).str.strip() if indic_c else "NA"

if indic_c is not None and "E_CC" in set(cloud_s["indic"].str.upper()):
    cloud_s = cloud_s[cloud_s["indic"].str.upper() == "E_CC"]

if unit_c is not None and "PC_ENT" in set(cloud_s["unit"].str.upper()):
    cloud_s = cloud_s[cloud_s["unit"].str.upper() == "PC_ENT"]

silver_cloud = cloud_s[["year", "indic", "unit", val_c]].rename(columns={val_c: "cloud_value"})
silver_cloud.to_parquet(SILVER_CLOUD, index=False)
print(f"[OK] Silver CLOUD salvo: {SILVER_CLOUD} | linhas:", len(silver_cloud))

# =========================
# GOLD - GVA (pivot por unit)
# =========================
gold_gva = (
    silver_gva
      .pivot_table(index=["sector", "year"], columns="unit", values="value_added", aggfunc="mean")
      .reset_index()
)
gold_gva.columns = [("value_added_" + str(c).lower()) if c not in ["sector", "year"] else c for c in gold_gva.columns]
gold_gva.to_parquet(GOLD_GVA, index=False)
print(f"[OK] Gold GVA salvo: {GOLD_GVA} | linhas:", len(gold_gva), "| setores:", gold_gva["sector"].nunique())

# =========================
# GOLD - CLOUD (pivot por indic/unit)  <-- FIX AQUI
# =========================
gold_cloud = (
    silver_cloud
      .pivot_table(index=["year"], columns=["indic","unit"], values="cloud_value", aggfunc="mean")
      .reset_index()
)

# achata nomes corretamente, preservando year mesmo se vier como ('year','')
new_cols = []
for c in gold_cloud.columns:
    if c == "year":
        new_cols.append("year")
        continue
    if isinstance(c, tuple) and len(c) >= 1 and str(c[0]).lower() == "year":
        new_cols.append("year")
        continue
    if isinstance(c, tuple) and len(c) == 2:
        indic, unit = c
        new_cols.append(f"cloud_{str(indic).lower()}_{str(unit).lower()}")
    else:
        new_cols.append(f"cloud_{str(c).lower()}")

gold_cloud.columns = new_cols

# garante year presente
if "year" not in gold_cloud.columns:
    raise ValueError(f"Year não foi preservado. Colunas finais: {gold_cloud.columns.tolist()}")

gold_cloud.to_parquet(GOLD_CLOUD, index=False)
print(f"[OK] Gold CLOUD salvo: {GOLD_CLOUD} | anos:", gold_cloud["year"].nunique())

# =========================
# DATASET FINAL
# =========================
ds = gold_gva.merge(gold_cloud, on="year", how="left")

cloud_cols = [c for c in ds.columns if c.startswith("cloud_")]
if len(cloud_cols) >= 1:
    ds["cloud_intensity"] = ds[cloud_cols[0]]
else:
    ds["cloud_intensity"] = np.nan

target_candidates = [c for c in ds.columns if c.startswith("value_added_")]
if len(target_candidates) == 0:
    raise ValueError("Não achei value_added_* na Gold.")

prefer = [c for c in target_candidates if "clv" in c or "i20" in c or "real" in c]
target_col = prefer[0] if prefer else target_candidates[0]

ds["target_value_added"] = ds[target_col]

tmp = ds.groupby("sector")["target_value_added"].mean().replace([np.inf,-np.inf], np.nan).dropna()
if len(tmp) > 1:
    w = np.log1p(tmp)
    w = (w - w.min()) / (w.max() - w.min() + 1e-9)
    w = 0.5 + 0.5*w
    ds = ds.merge(w.rename("sector_weight").reset_index(), on="sector", how="left")
else:
    ds["sector_weight"] = 1.0

ds["cloud_intensity_sector"] = ds["cloud_intensity"] * ds["sector_weight"]

ds.to_parquet(GOLD_DATASET, index=False)
ds.to_csv(GOLD_CSV, index=False)

print("\n[OK] Dataset final salvo:")
print(" -", GOLD_DATASET)
print(" -", GOLD_CSV)
print("\n[INFO] Linhas:", len(ds), "| setores:", ds["sector"].nunique(), "| anos:", ds["year"].nunique())
print("[INFO] Target principal:", target_col)
