# forecast_multi_year
import pandas as pd
from sklearn.linear_model import LinearRegression

# =========================
# CONFIGURAÇÕES
# =========================
INPUT_CSV = "gold_model_dataset.csv"
OUTPUT_CSV = "forecast_multi_year.csv"

START_YEAR = 2026
END_YEAR = 2030

# Se você quiser crescimento fixo:
USE_FIXED_GROWTH = True
FIXED_GROWTH_RATE = 0.05  # 5% ao ano (mude aqui)

# =========================
# 1) LÊ E LIMPA DADOS
# =========================
df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=["value_added_real", "cloud_intensity", "sector", "year"])

# =========================
# 2) TREINA 1 MODELO POR SETOR
#    (value_added_real ~ cloud_intensity + year)
# =========================
models = {}
last_cloud_by_sector = {}
last_year_by_sector = {}

for sector, g in df.groupby("sector"):
    g = g.sort_values("year")

    # precisa de um mínimo de dados
    if len(g) < 4:
        continue

    X = g[["cloud_intensity", "year"]]
    y = g["value_added_real"]

    model = LinearRegression()
    model.fit(X, y)

    models[sector] = model
    last_cloud_by_sector[sector] = float(g["cloud_intensity"].iloc[-1])
    last_year_by_sector[sector] = int(g["year"].iloc[-1])

# =========================
# 3) FUNÇÃO PRA PROJETAR CLOUD
# =========================
def project_cloud(sector, year, base_year, base_cloud):
    # Opção A: crescimento fixo ao ano
    if USE_FIXED_GROWTH:
        years_ahead = year - base_year
        return base_cloud * ((1 + FIXED_GROWTH_RATE) ** years_ahead)

    # Opção B: tenta aprender tendência linear de cloud por ano com o histórico do setor
    g = df[df["sector"] == sector].sort_values("year")
    if len(g) < 3:
        # fallback: se não tiver histórico suficiente
        years_ahead = year - base_year
        return base_cloud * ((1 + 0.03) ** years_ahead)

    Xc = g[["year"]]
    yc = g["cloud_intensity"]
    cloud_model = LinearRegression()
    cloud_model.fit(Xc, yc)

    return float(cloud_model.predict(pd.DataFrame({"year": [year]}))[0])

# =========================
# 4) SIMULA ANO A ANO
# =========================
rows = []

for sector, model in models.items():
    base_year = last_year_by_sector[sector]
    base_cloud = last_cloud_by_sector[sector]

    for year in range(START_YEAR, END_YEAR + 1):
        cloud = project_cloud(sector, year, base_year, base_cloud)

        X_pred = pd.DataFrame({
            "cloud_intensity": [cloud],
            "year": [year]
        })

        pred_value = float(model.predict(X_pred)[0])

        rows.append({
            "sector": sector,
            "year": year,
            "cloud_intensity_sim": cloud,
            "pred_value_added_real": pred_value
        })

out = pd.DataFrame(rows).sort_values(["sector", "year"])
out.to_csv(OUTPUT_CSV, index=False)

print(f"Saved: {OUTPUT_CSV}")
print(out.head(20).to_string(index=False))
