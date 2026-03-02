import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW = os.path.join(ROOT, "data", "raw")
OUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUT_DIR, exist_ok=True)

GVA_CSV = os.path.join(DATA_RAW, "estat_nama_10_a64_en.csv")
CLOUD_CSV = os.path.join(DATA_RAW, "estat_isoc_cicce_usen2_en.csv")

BRONZE_GVA = os.path.join(OUT_DIR, "bronze_gva_de.parquet")
BRONZE_CLOUD = os.path.join(OUT_DIR, "bronze_cloud_de.parquet")

def main():
    if not os.path.exists(GVA_CSV):
        raise FileNotFoundError(f"Arquivo não encontrado: {GVA_CSV}")
    if not os.path.exists(CLOUD_CSV):
        raise FileNotFoundError(f"Arquivo não encontrado: {CLOUD_CSV}")

    print("[INFO] Lendo CSV GVA (pode demorar)...")
    gva = pd.read_csv(GVA_CSV, low_memory=False)
    print("[INFO] Lendo CSV CLOUD (pode demorar)...")
    cloud = pd.read_csv(CLOUD_CSV, low_memory=False)

    print(f"[INFO] Linhas GVA total: {len(gva)}")
    print(f"[INFO] Linhas CLOUD total: {len(cloud)}")

    gva_de = gva[gva["geo"] == "DE"].copy()
    cloud_de = cloud[cloud["geo"] == "DE"].copy()

    print(f"[INFO] Linhas GVA Alemanha: {len(gva_de)}")
    print(f"[INFO] Linhas CLOUD Alemanha: {len(cloud_de)}")

    gva_de.to_parquet(BRONZE_GVA, index=False)
    cloud_de.to_parquet(BRONZE_CLOUD, index=False)

    print("\n[OK] Bronze (DE) salvo em output/:")
    print(" -", os.path.basename(BRONZE_GVA))
    print(" -", os.path.basename(BRONZE_CLOUD))

if __name__ == "__main__":
    main()
