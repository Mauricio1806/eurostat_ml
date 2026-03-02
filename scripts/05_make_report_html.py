import os
import json
import base64
from io import BytesIO

import pandas as pd
import matplotlib.pyplot as plt


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "output")
ASSETS = os.path.join(OUT, "report_assets")
os.makedirs(OUT, exist_ok=True)
os.makedirs(ASSETS, exist_ok=True)

PATH_REPORT_JSON = os.path.join(OUT, "ml_report.json")
PATH_HOLDOUT = os.path.join(OUT, "predictions_holdout.csv")
PATH_FORECAST = os.path.join(OUT, "forecast_2026_2030.csv")
PATH_FORECAST_LR = os.path.join(OUT, "forecast_2026_2030_lr.csv")
PATH_HTML = os.path.join(OUT, "report.html")


def fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def safe_read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_column(df: pd.DataFrame, candidates_exact, candidates_contains):
    # 1) match exato (case-insensitive)
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates_exact:
        if k.lower() in lower_map:
            return lower_map[k.lower()]

    # 2) match por substring
    for c in df.columns:
        cl = c.lower()
        for frag in candidates_contains:
            if frag in cl:
                return c
    return None


def main():
    # --------- Load inputs ---------
    report = safe_read_json(PATH_REPORT_JSON) or {}

    if not os.path.exists(PATH_HOLDOUT):
        raise FileNotFoundError(f"Não achei: {PATH_HOLDOUT}")
    hold = pd.read_csv(PATH_HOLDOUT)

    if not os.path.exists(PATH_FORECAST):
        raise FileNotFoundError(f"Não achei: {PATH_FORECAST}")
    fc = pd.read_csv(PATH_FORECAST)

    fc_lr = None
    if os.path.exists(PATH_FORECAST_LR):
        fc_lr = pd.read_csv(PATH_FORECAST_LR)

    # --------- Gráfico 1: Holdout Actual vs Predicted ---------
    col_true = pick_column(
        hold,
        candidates_exact=["y_true", "true", "actual", "target"],
        candidates_contains=["y_true", "true", "actual"],
    )
    col_pred = pick_column(
        hold,
        candidates_exact=["y_pred", "pred", "prediction", "predicted"],
        candidates_contains=["y_pred", "pred"],
    )

    img_holdout = ""
    if col_true and col_pred:
        # remove NaN para evitar crash em scatter
        tmp = hold[[col_true, col_pred]].dropna()
        fig = plt.figure()
        plt.scatter(tmp[col_true], tmp[col_pred], s=10)
        plt.xlabel("Actual (holdout)")
        plt.ylabel("Predicted (holdout)")
        plt.title("Holdout: Actual vs Predicted")
        img_holdout = fig_to_base64(fig)

    # --------- Gráfico 2: Forecast (Top 6 setores por último ano) ---------
    required = {"sector", "year", "pred_target_value_added"}
    if not required.issubset(set(map(str, fc.columns))):
        raise ValueError(
            f"forecast_2026_2030.csv precisa ter colunas {sorted(list(required))}. "
            f"Colunas encontradas: {list(fc.columns)}"
        )

    fc = fc.copy()
    fc["year"] = pd.to_numeric(fc["year"], errors="coerce")
    fc = fc.dropna(subset=["year"])
    fc["year"] = fc["year"].astype(int)

    last_year = int(fc["year"].max())
    top_sectors = (
        fc[fc["year"] == last_year]
        .sort_values("pred_target_value_added", ascending=False)
        .head(6)["sector"]
        .astype(str)
        .tolist()
    )

    fig = plt.figure()
    for s in top_sectors:
        d = fc[fc["sector"].astype(str) == str(s)].sort_values("year")
        plt.plot(d["year"], d["pred_target_value_added"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Predicted value added")
    plt.title("Forecast 2026–2030 (Top sectors by 2030)")
    img_forecast = fig_to_base64(fig)

    # --------- Gráfico 3 (opcional): comparação LR vs principal (um setor) ---------
    img_compare = ""
    if fc_lr is not None and {"sector", "year", "pred_target_value_added"}.issubset(set(fc_lr.columns)):
        sector_ref = top_sectors[0] if top_sectors else str(fc["sector"].iloc[0])

        d1 = fc[fc["sector"].astype(str) == str(sector_ref)].sort_values("year")
        d2 = fc_lr[fc_lr["sector"].astype(str) == str(sector_ref)].copy()
        d2["year"] = pd.to_numeric(d2["year"], errors="coerce")
        d2 = d2.dropna(subset=["year"]).sort_values("year")

        fig = plt.figure()
        plt.plot(d1["year"], d1["pred_target_value_added"], marker="o", label="Main")
        plt.plot(d2["year"], d2["pred_target_value_added"], marker="o", label="LinearRegression")
        plt.xlabel("Year")
        plt.ylabel("Predicted value added")
        plt.title(f"Model comparison (sector {sector_ref}): Main vs LinearRegression")
        plt.legend()
        img_compare = fig_to_base64(fig)

    # --------- HTML blocks ---------
    if isinstance(report, dict) and report:
        metrics_block = f"<pre>{json.dumps(report, indent=2, ensure_ascii=False)}</pre>"
    else:
        metrics_block = "<p><i>ml_report.json não encontrado ou vazio.</i></p>"

    holdout_head = hold.head(20).to_html(index=False)
    fc_head = fc.head(30).to_html(index=False)

    holdout_img_block = (
        f'<img src="data:image/png;base64,{img_holdout}"/>'
        if img_holdout
        else "<p><i>Não foi possível montar o gráfico do holdout (não identifiquei colunas de real/predito).</i></p>"
    )

    compare_block = (
        f'<img src="data:image/png;base64,{img_compare}"/>'
        if img_compare
        else "<p><i>forecast_2026_2030_lr.csv não encontrado (ou sem colunas compatíveis).</i></p>"
    )

    html = f"""<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Eurostat ML Report — Germany</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.35; }}
    h1, h2 {{ margin: 0.4em 0; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 16px 0; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    code {{ background: #f2f2f2; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Eurostat Cloud Adoption × Economic Performance (Germany)</h1>
  <p>
    Relatório gerado automaticamente a partir dos artefatos em <code>output/</code>.
    Foco: pipeline Bronze/Silver/Gold + avaliação em holdout + simulação 2026–2030.
  </p>

  <div class="card">
    <h2>Métricas (holdout)</h2>
    {metrics_block}
  </div>

  <div class="card">
    <h2>Holdout: Actual vs Predicted</h2>
    {holdout_img_block}
  </div>

  <div class="card">
    <h2>Forecast 2026–2030 (Top setores por 2030)</h2>
    <img src="data:image/png;base64,{img_forecast}"/>
  </div>

  <div class="card">
    <h2>Comparação de modelos (opcional)</h2>
    {compare_block}
  </div>

  <div class="card">
    <h2>Prévia: predictions_holdout.csv</h2>
    {holdout_head}
  </div>

  <div class="card">
    <h2>Prévia: forecast_2026_2030.csv</h2>
    {fc_head}
  </div>

  <p style="margin-top: 28px; color: #666;">
    Gerado por <code>scripts/05_make_report_html.py</code>.
  </p>
</body>
</html>
"""

    with open(PATH_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print("[OK] Report HTML salvo em:", PATH_HTML)

if __name__ == "__main__":
    main()
