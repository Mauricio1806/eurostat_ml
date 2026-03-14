# scripts/05_make_report_html.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# =========================
# Config (TCC / Paper-like)
# =========================
DECIMALS_TABLE = 2
DECIMALS_METRICS = 2
MAX_ROWS_PREVIEW = 20  # mantém HTML leve
DPI = 360              # mais nítido para Word/Docs
EXPORT_SVG = True      # SVG é perfeito para colar no Word sem perder qualidade

TITLE = "Eurostat Cloud Adoption × Economic Performance (Germany)"
SUBTITLE = (
    "Automatically generated report from artifacts in <code>output/</code>. "
    "Scope: Bronze/Silver/Gold pipeline + holdout evaluation + 2026–2030 simulation."
)

# =========================
# Paths
# =========================
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

PATHS = {
    "ml_report": OUT / "ml_report.json",
    "pred_holdout": OUT / "predictions_holdout.csv",
    "forecast_main": OUT / "forecast_2026_2030.csv",
    "forecast_lr": OUT / "forecast_2026_2030_lr.csv",
    "report_html": OUT / "report.html",
}

# =========================
# Matplotlib style (academic)
# =========================
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": DPI,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.8,
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.8,
})


# =========================
# Helpers
# =========================
def _pt_thousands(x: float) -> str:
    # 1234567 -> "1.234.567"
    try:
        return f"{int(round(x)):,}".replace(",", ".")
    except Exception:
        return str(x)


def _fmt_num(x: float, decimals: int = 2) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    # pt-BR: milhar ".", decimal ","
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_thousands_axis(x, pos):
    return _pt_thousands(x)


def _safe_read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _format_df_for_html(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].map(lambda v: "" if pd.isna(v) else _fmt_num(float(v), DECIMALS_TABLE))
    return df2


def _df_to_html(df: pd.DataFrame, caption: str) -> str:
    df_prev = df.head(MAX_ROWS_PREVIEW).copy()
    df_prev = _format_df_for_html(df_prev)

    html = df_prev.to_html(
        index=False,
        escape=True,
        border=0,
        classes="table",
    )
    return f"""
    <div class="card">
      <h2>{caption}</h2>
      <div class="table-wrap">{html}</div>
      <p class="hint">Preview: showing {min(len(df), MAX_ROWS_PREVIEW)} of {len(df)} rows.</p>
    </div>
    """


def _savefig(base_path_no_ext: Path):
    # salva PNG + (opcional) SVG
    png_path = base_path_no_ext.with_suffix(".png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=DPI, bbox_inches="tight")
    if EXPORT_SVG:
        svg_path = base_path_no_ext.with_suffix(".svg")
        plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    return png_path


def _img_tag(img_path: Path, alt: str) -> str:
    if not img_path.exists():
        return f'<p class="warn">[Image missing: {img_path.name}]</p>'
    rel = os.path.relpath(img_path, OUT).replace("\\", "/")
    return f'<img alt="{alt}" src="{rel}"/>'


def _pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =========================
# Plots (academic)
# =========================
def plot_holdout_scatter(pred: pd.DataFrame, r2_text: Optional[str] = None) -> Path:
    pred_col = _pick_col(pred, ["pred_xgb", "pred_main", "pred", "y_pred", "pred_lr"])
    if pred_col is None or "y_true" not in pred.columns:
        plt.figure(figsize=(8.2, 6.0))
        plt.title("Holdout: Actual vs Predicted")
        plt.text(0.5, 0.5, "Missing columns for holdout plot", ha="center", va="center")
        return _savefig(PLOTS / "holdout_actual_vs_predicted")

    df = pred[["y_true", pred_col]].dropna().copy()
    y = df["y_true"].astype(float).to_numpy()
    yhat = df[pred_col].astype(float).to_numpy()

    # Resíduo para faixa visual
    resid = yhat - y
    abs_resid = np.abs(resid)

    plt.figure(figsize=(8.2, 6.0))
    # pontos: tamanho pequeno e alpha leve (mais acadêmico)
    plt.scatter(y, yhat, s=18, alpha=0.85)

    # linha 45°
    mn = float(min(y.min(), yhat.min()))
    mx = float(max(y.max(), yhat.max()))
    plt.plot([mn, mx], [mn, mx], linewidth=1.2)

    # título mais “paper”
    title = "Holdout performance: actual vs predicted"
    if r2_text:
        title += f" (R²={r2_text})"
    plt.title(title)
    plt.xlabel("Actual value added (holdout)")
    plt.ylabel("Predicted value added (holdout)")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_thousands_axis))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands_axis))

    # anotação discreta com MAE aproximado
    mae = float(np.mean(abs_resid)) if len(abs_resid) else np.nan
    plt.text(
        0.02, 0.96,
        f"MAE ≈ {_pt_thousands(mae)}",
        transform=ax.transAxes,
        va="top"
    )

    return _savefig(PLOTS / "holdout_actual_vs_predicted")


def plot_forecast_top_sectors(forecast: pd.DataFrame, title: str, out_name: str) -> Path:
    value_col = _pick_col(forecast, ["pred_target_value_added", "y_pred", "pred", "forecast", "pred_target_value_added_lr"])
    if value_col is None:
        # tenta achar coluna predita por padrão
        for c in forecast.columns:
            if "pred" in c and "value" in c:
                value_col = c
                break

    if value_col is None or not {"sector", "year"}.issubset(set(forecast.columns)):
        plt.figure(figsize=(9.2, 6.0))
        plt.title(title)
        plt.text(0.5, 0.5, "Missing columns for forecast plot", ha="center", va="center")
        return _savefig(PLOTS / out_name)

    df = forecast[["sector", "year", value_col]].dropna().copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    last_year = int(df["year"].max())
    top_sectors = (
        df[df["year"] == last_year]
        .sort_values(value_col, ascending=False)
        .head(6)["sector"]
        .astype(str)
        .tolist()
    )

    plt.figure(figsize=(9.6, 6.0))
    for s in top_sectors:
        d = df[df["sector"].astype(str) == s].sort_values("year")
        plt.plot(
            d["year"], d[value_col],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=str(s),
        )

    plt.title(f"{title} — Top sectors by {last_year}")
    plt.xlabel("Year")
    plt.ylabel("Predicted value added")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands_axis))

    # legenda fora (paper-like)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

    return _savefig(PLOTS / out_name)


def plot_model_comparison_optional(f_main: pd.DataFrame, f_lr: pd.DataFrame) -> Path:
    def pick_value_col(df):
        c = _pick_col(df, ["pred_target_value_added", "pred", "y_pred"])
        if c:
            return c
        for cc in df.columns:
            if "pred" in cc and "value" in cc:
                return cc
        return None

    v1 = pick_value_col(f_main)
    v2 = pick_value_col(f_lr)
    if v1 is None or v2 is None:
        plt.figure(figsize=(8.2, 5.6))
        plt.title("Model comparison")
        plt.text(0.5, 0.5, "Missing prediction columns", ha="center", va="center")
        return _savefig(PLOTS / "model_comparison_main_vs_lr")

    # preferir B-E, senão pega o topo do último ano
    sector_focus = "B-E"
    d1 = f_main[f_main["sector"].astype(str) == sector_focus].copy() if "sector" in f_main.columns else pd.DataFrame()
    d2 = f_lr[f_lr["sector"].astype(str) == sector_focus].copy() if "sector" in f_lr.columns else pd.DataFrame()

    if d1.empty or d2.empty:
        if {"sector", "year"}.issubset(set(f_main.columns)):
            f_main2 = f_main.copy()
            f_main2["year"] = pd.to_numeric(f_main2["year"], errors="coerce")
            last_year = int(f_main2["year"].max())
            sector_focus = (
                f_main2[f_main2["year"] == last_year]
                .sort_values(v1, ascending=False)
                .head(1)["sector"]
                .astype(str)
                .iloc[0]
            )
            d1 = f_main2[f_main2["sector"].astype(str) == sector_focus].copy()
            d2 = f_lr.copy()
            d2["year"] = pd.to_numeric(d2["year"], errors="coerce")
            d2 = d2[d2["sector"].astype(str) == sector_focus].copy()

    d1["year"] = pd.to_numeric(d1["year"], errors="coerce")
    d2["year"] = pd.to_numeric(d2["year"], errors="coerce")
    d1 = d1.dropna(subset=["year"]).sort_values("year")
    d2 = d2.dropna(subset=["year"]).sort_values("year")

    plt.figure(figsize=(8.4, 5.8))
    plt.plot(d1["year"], d1[v1], marker="o", linewidth=1.9, label="Main model")
    plt.plot(d2["year"], d2[v2], marker="o", linewidth=1.9, label="Linear Regression (baseline)")

    plt.title(f"Model comparison — sector {sector_focus}")
    plt.xlabel("Year")
    plt.ylabel("Predicted value added")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands_axis))
    plt.legend()

    return _savefig(PLOTS / "model_comparison_main_vs_lr")


# =========================
# Metrics block (executive + academic)
# =========================
def metrics_to_html(ml_report: Dict) -> str:
    if not ml_report:
        return """
        <div class="card">
          <h2>Holdout metrics</h2>
          <p class="warn">ml_report.json not found.</p>
        </div>
        """

    rows = []

    # counts
    for k in ["rows_total", "rows_train", "rows_test",
              "nan_cloud_intensity_before_impute", "nan_cloud_intensity_sector_before_impute"]:
        if k in ml_report:
            rows.append((k.replace("_", " "), _fmt_num(float(ml_report[k]), 0)))

    # models
    def add_model(model_key: str, label: str):
        if model_key in ml_report and isinstance(ml_report[model_key], dict):
            m = ml_report[model_key]
            if "rmse" in m:
                rows.append((f"{label} — RMSE", _fmt_num(float(m["rmse"]), DECIMALS_METRICS)))
            if "mae" in m:
                rows.append((f"{label} — MAE", _fmt_num(float(m["mae"]), DECIMALS_METRICS)))
            if "r2" in m:
                rows.append((f"{label} — R²", _fmt_num(float(m["r2"]), 4)))  # R² com mais precisão

    add_model("model_linear_regression", "Linear Regression")
    add_model("model_xgboost", "XGBoost")

    dfm = pd.DataFrame(rows, columns=["Metric", "Value"])
    html = dfm.to_html(index=False, escape=True, border=0, classes="table")

    return f"""
    <div class="card">
      <h2>Holdout metrics</h2>
      <div class="table-wrap">{html}</div>
      <p class="hint">Values are formatted for reporting (tables with {DECIMALS_TABLE} decimals, R² with higher precision).</p>
    </div>
    """


# =========================
# HTML
# =========================
def build_html(
    metrics_html: str,
    img_scatter: Path,
    img_forecast_main: Path,
    img_forecast_lr: Optional[Path],
    img_model_compare: Optional[Path],
    df_pred: Optional[pd.DataFrame],
    df_forecast: Optional[pd.DataFrame],
) -> str:
    def section_img(title: str, img_path: Path, note: str = "") -> str:
        return f"""
        <div class="card">
          <h2>{title}</h2>
          {_img_tag(img_path, title)}
          {f'<p class="hint">{note}</p>' if note else ''}
        </div>
        """

    html_pred = _df_to_html(df_pred, "Preview: predictions_holdout.csv") if df_pred is not None else ""
    html_forecast = _df_to_html(df_forecast, "Preview: forecast_2026_2030.csv") if df_forecast is not None else ""

    extra = ""
    if img_model_compare is not None:
        extra += section_img(
            "Model comparison (optional)",
            img_model_compare,
            note="Comparison on a representative sector (defaults to B-E when available).",
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Eurostat ML Report — Germany</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 28px;
      line-height: 1.45;
      color: #111;
      max-width: 1080px;
    }}
    h1 {{
      margin: 0 0 6px 0;
      font-size: 28px;
    }}
    .sub {{
      margin: 0 0 18px 0;
      color: #444;
      font-size: 14px;
    }}
    .card {{
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 16px 18px;
      margin: 14px 0;
      background: #fff;
    }}
    .card h2 {{
      margin: 0 0 10px 0;
      font-size: 18px;
    }}
    img {{
      max-width: 100%;
      height: auto;
      border: 1px solid #eee;
      border-radius: 10px;
      background: #fff;
    }}
    .table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .table th, .table td {{
      border: 1px solid #ddd;
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    .table th {{
      background: #f6f6f6;
    }}
    .table-wrap {{
      overflow-x: auto;
      overflow-y: hidden;
      border-radius: 10px;
    }}
    .hint {{
      margin: 10px 0 0 0;
      color: #555;
      font-size: 12px;
    }}
    .warn {{
      color: #a00;
      font-size: 12px;
      margin: 8px 0 0 0;
    }}
    code {{
      background: #f4f4f4;
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <h1>{TITLE}</h1>
  <p class="sub">{SUBTITLE}</p>

  {metrics_html}

  {section_img(
      "Holdout performance: actual vs predicted",
      img_scatter,
      note="Figure note: 45° line indicates perfect calibration; points above/below indicate positive/negative residuals."
  )}

  {section_img(
      "Forecast 2026–2030 (Main model): top sectors",
      img_forecast_main,
      note="Top sectors selected by predicted value added in the last simulation year."
  )}

  {section_img(
      "Forecast 2026–2030 (Linear Regression baseline): top sectors",
      img_forecast_lr,
      note="Baseline model for comparison (interpretation-focused)."
  ) if img_forecast_lr is not None else ""}

  {extra}

  {html_pred}
  {html_forecast}

  <div class="card">
    <h2>Artifacts & exports</h2>
    <ul>
      <li><code>output/ml_report.json</code></li>
      <li><code>output/predictions_holdout.csv</code></li>
      <li><code>output/forecast_2026_2030.csv</code></li>
      <li><code>output/forecast_2026_2030_lr.csv</code> (optional)</li>
      <li><code>output/plots/*.png</code> (high-DPI)</li>
      <li><code>output/plots/*.svg</code> (vector, best for Word)</li>
    </ul>
    <p class="hint">Tip: use the SVG files in your TCC (vector quality). PNG is included for quick preview.</p>
  </div>
</body>
</html>
"""


def main() -> None:
    # load artifacts
    ml_report = _safe_read_json(PATHS["ml_report"])
    df_pred = _read_csv(PATHS["pred_holdout"])
    df_forecast = _read_csv(PATHS["forecast_main"])
    df_forecast_lr = _read_csv(PATHS["forecast_lr"])

    # capture R² for title
    r2_text = None
    try:
        r2 = ml_report.get("model_xgboost", {}).get("r2", None)
        if r2 is None:
            r2 = ml_report.get("model_linear_regression", {}).get("r2", None)
        if r2 is not None:
            r2_text = f"{float(r2):.4f}"
    except Exception:
        r2_text = None

    # plots
    img_scatter = plot_holdout_scatter(df_pred, r2_text=r2_text) if df_pred is not None else (PLOTS / "holdout_actual_vs_predicted.png")

    img_forecast_main = (
        plot_forecast_top_sectors(df_forecast, "Forecast 2026–2030", "forecast_main_top_sectors")
        if df_forecast is not None
        else (PLOTS / "forecast_main_top_sectors.png")
    )

    img_forecast_lr = None
    if df_forecast_lr is not None and not df_forecast_lr.empty:
        img_forecast_lr = plot_forecast_top_sectors(df_forecast_lr, "Forecast 2026–2030 (Linear Regression)", "forecast_lr_top_sectors")

    img_model_compare = None
    if df_forecast is not None and df_forecast_lr is not None and not df_forecast.empty and not df_forecast_lr.empty:
        img_model_compare = plot_model_comparison_optional(df_forecast, df_forecast_lr)

    # html blocks
    metrics_html = metrics_to_html(ml_report)

    # build html
    html = build_html(
        metrics_html=metrics_html,
        img_scatter=img_scatter,
        img_forecast_main=img_forecast_main,
        img_forecast_lr=img_forecast_lr,
        img_model_compare=img_model_compare,
        df_pred=df_pred,
        df_forecast=df_forecast,
    )

    PATHS["report_html"].write_text(html, encoding="utf-8")

    print(f"[OK] Report HTML: {PATHS['report_html']}")
    print(f"[OK] Plots folder: {PLOTS}")
    if EXPORT_SVG:
        print("[OK] SVG exports enabled (best for Word).")


if __name__ == "__main__":
    main()