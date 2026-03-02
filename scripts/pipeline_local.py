import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR  = os.path.join(BASE_DIR, "output")

os.makedirs(OUT_DIR, exist_ok=True)

GVA_FILE   = os.path.join(DATA_DIR, "estat_nama_10_a64_en.csv")
CLOUD_FILE = os.path.join(DATA_DIR, "estat_isoc_cicce_usen2_en.csv")

print("[INFO] GVA_FILE:", GVA_FILE)
print("[INFO] CLOUD_FILE:", CLOUD_FILE)

if not os.path.exists(GVA_FILE):
    raise FileNotFoundError(f"GVA não encontrado: {GVA_FILE}")

if not os.path.exists(CLOUD_FILE):
    raise FileNotFoundError(f"CLOUD não encontrado: {CLOUD_FILE}")

# =========================
# 1) LEITURA
# =========================
print("\n[INFO] Lendo GVA (pode demorar)...")
gva = pd.read_csv(GVA_FILE, low_memory=False)

print("[INFO] Lendo CLOUD (pode demorar)...")
cloud = pd.read_csv(CLOUD_FILE, low_memory=False)

print("\n[INFO] Colunas GVA:", gva.columns.tolist())
print("[INFO] Colunas CLOUD:", cloud.columns.tolist())

# =========================
# 2) FILTRO ALEMANHA
# =========================
if "geo" not in gva.columns:
    raise ValueError("Coluna 'geo' não existe no GVA. Veja as colunas impressas acima.")
if "geo" not in cloud.columns:
    raise ValueError("Coluna 'geo' não existe no CLOUD. Veja as colunas impressas acima.")

gva_de = gva[gva["geo"] == "DE"].copy()
cloud_de = cloud[cloud["geo"] == "DE"].copy()

print("\n[INFO] Linhas GVA total:", len(gva), " | Alemanha:", len(gva_de))
print("[INFO] Linhas CLOUD total:", len(cloud), " | Alemanha:", len(cloud_de))

# salva bronze local (Alemanha)
gva_de.to_parquet(os.path.join(OUT_DIR, "bronze_gva_de.parquet"), index=False)
cloud_de.to_parquet(os.path.join(OUT_DIR, "bronze_cloud_de.parquet"), index=False)

print("\n[OK] Bronze (DE) salvo em output/:")
print(" - bronze_gva_de.parquet")
print(" - bronze_cloud_de.parquet")
