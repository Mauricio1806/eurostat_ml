# scripts/02_train_ml_robust.py
import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ---------------- Paths ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "output")

PATH_DATASET_CSV = os.path.join(OUT, "gold_model_dataset.csv")
PATH_DATASET_PARQUET = os.path.join(OUT, "gold_model_dataset.parquet")

PATH_REPORT_JSON = os.path.join(OUT, "ml_report.json")
PATH_PRED_HOLDOUT = os.path.join(OUT, "predictions_holdout.csv")
PATH_FEAT_IMP = os.path.join(OUT, "feature_importance.csv")

# ---------------- Config ----------------
TARGET_COL = "target_value_added"  # você confirmou que existe
SECTOR_COL = "sector"
YEAR_COL = "year"

CLOUD_COL_MAIN = "cloud_intensity"
CLOUD_COL_SECTOR = "cloud_intensity_sector"

TEST_LAST_N_YEARS = 3  # holdout temporal = últimos N anos

# Se quiser incluir peso do setor como feature, deixe True
INCLUDE_SECTOR_WEIGHT = True
SECTOR_WEIGHT_COL = "sector_weight"

# ---------------- Utils ----------------
def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": _rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

def _ensure_output_dir():
    os.makedirs(OUT, exist_ok=True)

def _read_gold_dataset() -> pd.DataFrame:
    if os.path.exists(PATH_DATASET_CSV):
        return pd.read_csv(PATH_DATASET_CSV)
    if os.path.exists(PATH_DATASET_PARQUET):
        return pd.read_parquet(PATH_DATASET_PARQUET)
    raise FileNotFoundError(
        f"Não achei dataset Gold em:\n- {PATH_DATASET_CSV}\n- {PATH_DATASET_PARQUET}"
    )

def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _temporal_split(df: pd.DataFrame, year_col: str, last_n_years: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    years = sorted(df[year_col].dropna().unique().tolist())
    if len(years) <= last_n_years:
        raise ValueError(f"Poucos anos ({len(years)}) para holdout de {last_n_years} anos.")
    test_years = years[-last_n_years:]
    train_years = years[:-last_n_years]

    train_df = df[df[year_col].isin(train_years)].copy()
    test_df = df[df[year_col].isin(test_years)].copy()

    info = {
        "years_all": [int(y) for y in years],
        "years_train": [int(y) for y in train_years],
        "years_test": [int(y) for y in test_years],
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
    }
    return train_df, test_df, info

def _build_features(df: pd.DataFrame, use_sector_cloud: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Features alinhadas ao TCC (evita leakage):
      - year (num)
      - sector (one-hot)
      - cloud_intensity (num)
      - cloud_intensity_sector (num, opcional)
      - sector_weight (opcional)
    Exclui colunas value_added* para evitar vazar o target.
    """
    required = {SECTOR_COL, YEAR_COL, TARGET_COL, CLOUD_COL_MAIN}
    if use_sector_cloud:
        required.add(CLOUD_COL_SECTOR)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas obrigatórias ausentes: {missing}")

    # Seleção base
    cols = [YEAR_COL, SECTOR_COL, CLOUD_COL_MAIN]
    if use_sector_cloud:
        cols.append(CLOUD_COL_SECTOR)

    if INCLUDE_SECTOR_WEIGHT and SECTOR_WEIGHT_COL in df.columns:
        cols.append(SECTOR_WEIGHT_COL)

    X = df[cols].copy()

    # One-hot do setor
    X = pd.get_dummies(X, columns=[SECTOR_COL], drop_first=True)

    meta = {
        "feature_cols_final": list(X.columns),
        "use_sector_cloud": bool(use_sector_cloud),
        "include_sector_weight": bool(INCLUDE_SECTOR_WEIGHT and SECTOR_WEIGHT_COL in df.columns),
    }
    return X, meta

def _impute_cloud(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Imputação transparente:
      - cloud_intensity: mediana global
      - cloud_intensity_sector: mediana por setor (fallback global)
    Cria flag is_imputed_cloud = 1 se qualquer imputação foi aplicada na linha.
    """
    df = df.copy()

    before_nan_main = int(df[CLOUD_COL_MAIN].isna().sum()) if CLOUD_COL_MAIN in df.columns else None
    before_nan_sector = int(df[CLOUD_COL_SECTOR].isna().sum()) if CLOUD_COL_SECTOR in df.columns else None

    imputed_any = pd.Series(False, index=df.index)

    if CLOUD_COL_MAIN in df.columns:
        med_main = float(df[CLOUD_COL_MAIN].median(skipna=True))
        mask = df[CLOUD_COL_MAIN].isna()
        if mask.any():
            df.loc[mask, CLOUD_COL_MAIN] = med_main
            imputed_any = imputed_any | mask

    if CLOUD_COL_SECTOR in df.columns:
        # mediana por setor
        global_med = float(df[CLOUD_COL_SECTOR].median(skipna=True))
        # preenche por grupo
        def fill_group(g: pd.DataFrame) -> pd.DataFrame:
            m = float(g[CLOUD_COL_SECTOR].median(skipna=True))
            if np.isnan(m):
                m = global_med
            gmask = g[CLOUD_COL_SECTOR].isna()
            if gmask.any():
                g.loc[gmask, CLOUD_COL_SECTOR] = m
            return g

        mask_before = df[CLOUD_COL_SECTOR].isna()
        if mask_before.any():
            df = df.groupby(SECTOR_COL, group_keys=False).apply(fill_group)
            imputed_any = imputed_any | mask_before

    df["is_imputed_cloud"] = imputed_any.astype(int)

    after_nan_main = int(df[CLOUD_COL_MAIN].isna().sum()) if CLOUD_COL_MAIN in df.columns else None
    after_nan_sector = int(df[CLOUD_COL_SECTOR].isna().sum()) if CLOUD_COL_SECTOR in df.columns else None

    info = {
        "nan_cloud_intensity_before": before_nan_main,
        "nan_cloud_intensity_after": after_nan_main,
        "nan_cloud_intensity_sector_before": before_nan_sector,
        "nan_cloud_intensity_sector_after": after_nan_sector,
        "rows_flagged_is_imputed_cloud": int(df["is_imputed_cloud"].sum()),
        "imputation_rules": {
            "cloud_intensity": "global median",
            "cloud_intensity_sector": "sector median (fallback global median)",
        },
    }
    return df, info

def _fit_predict_lr(X_train, y_train, X_test) -> np.ndarray:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)

def _fit_predict_xgb(X_train, y_train, X_test) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], str]:
    """
    Tenta XGBoost. Se não existir no ambiente, retorna None e uma mensagem.
    """
    try:
        from xgboost import XGBRegressor
    except Exception as e:
        return None, None, f"xgboost indisponível: {e}"

    # Hiperparâmetros conservadores (evita overfit em dataset pequeno)
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # Feature importance
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return preds, fi, "ok"

@dataclass
class ScenarioResult:
    scenario_name: str
    use_sector_cloud: bool
    split_info: Dict[str, Any]
    imputation_info: Dict[str, Any]
    feature_meta: Dict[str, Any]
    metrics: Dict[str, Any]
    notes: Dict[str, Any]

def run_scenario(df: pd.DataFrame, use_sector_cloud: bool) -> ScenarioResult:
    # 1) Cobertura
    coverage = {
        "year_min": int(df[YEAR_COL].min()),
        "year_max": int(df[YEAR_COL].max()),
        "n_years": int(df[YEAR_COL].nunique()),
        "n_sectors": int(df[SECTOR_COL].nunique()),
        "rows_total": int(len(df)),
    }

    # 2) Imputação (antes do split para consistência do pipeline)
    df_imp, imputation_info = _impute_cloud(df)

    # salva dataset com flag (ajuda muito na metodologia)
    # (mantém o mesmo arquivo, só adiciona a coluna)
    try:
        df_imp.to_csv(PATH_DATASET_CSV, index=False)
    except Exception:
        pass

    # 3) Split temporal
    train_df, test_df, split_info = _temporal_split(df_imp, YEAR_COL, TEST_LAST_N_YEARS)

    # 4) Features
    X_train, feature_meta = _build_features(train_df, use_sector_cloud=use_sector_cloud)
    X_test, _ = _build_features(test_df, use_sector_cloud=use_sector_cloud)

    y_train = train_df[TARGET_COL].astype(float).values
    y_test = test_df[TARGET_COL].astype(float).values

    # 5) Linear baseline
    pred_lr = _fit_predict_lr(X_train, y_train, X_test)
    m_lr = _metrics(y_test, pred_lr)

    # 6) XGBoost (se disponível)
    pred_xgb, fi, xgb_status = _fit_predict_xgb(X_train, y_train, X_test)
    m_xgb = None
    if pred_xgb is not None:
        m_xgb = _metrics(y_test, pred_xgb)

    # 7) Saída holdout com colunas relevantes
    holdout_out = test_df[[SECTOR_COL, YEAR_COL, CLOUD_COL_MAIN] + ([CLOUD_COL_SECTOR] if (use_sector_cloud and CLOUD_COL_SECTOR in test_df.columns) else [])].copy()
    holdout_out["y_true"] = y_test
    holdout_out["pred_lr"] = pred_lr
    if pred_xgb is not None:
        holdout_out["pred_xgb"] = pred_xgb

    # salva sempre (o último cenário executado sobrescreve; ok porque você usa o cenário "full" como principal)
    holdout_out.to_csv(PATH_PRED_HOLDOUT, index=False)

    # salva feature importance se existir
    if fi is not None:
        fi.to_csv(PATH_FEAT_IMP, index=False)

    metrics_pack = {
        "linear_regression": m_lr,
        "xgboost": (m_xgb if m_xgb is not None else None),
    }

    notes = {
        "coverage_summary": coverage,
        "xgboost_status": xgb_status,
        "methodological_alignment": {
            "split": "temporal (train early years / test last years)",
            "leakage_control": "excluded value_added* columns; used year+sector one-hot + cloud indicators (+ optional sector_weight)",
            "imputation": imputation_info["imputation_rules"],
        },
    }

    return ScenarioResult(
        scenario_name=("with_sector_cloud" if use_sector_cloud else "main_cloud_only"),
        use_sector_cloud=use_sector_cloud,
        split_info=split_info,
        imputation_info=imputation_info,
        feature_meta=feature_meta,
        metrics=metrics_pack,
        notes=notes,
    )

def main():
    _ensure_output_dir()
    df = _read_gold_dataset()

    # sanity
    needed_base = [SECTOR_COL, YEAR_COL, TARGET_COL, CLOUD_COL_MAIN]
    missing = [c for c in needed_base if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset gold_model_dataset está sem colunas base: {missing}")

    # garante tipos
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").astype("Int64")
    df = df.dropna(subset=[YEAR_COL, SECTOR_COL, TARGET_COL]).copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)

    # Roda robustez: (A) só cloud_intensity; (B) cloud_intensity + cloud_intensity_sector
    results = []
    results.append(run_scenario(df, use_sector_cloud=False))
    # só roda cenário B se a coluna existir
    if CLOUD_COL_SECTOR in df.columns:
        results.append(run_scenario(df, use_sector_cloud=True))

    # Report final (estrutura pensada pro TCC)
    report: Dict[str, Any] = {
        "rows_total": int(len(df)),
        "n_sectors": int(df[SECTOR_COL].nunique()),
        "year_min": int(df[YEAR_COL].min()),
        "year_max": int(df[YEAR_COL].max()),
        "nan_cloud_intensity_before_impute": int(pd.isna(df[CLOUD_COL_MAIN]).sum()),
        "nan_cloud_intensity_sector_before_impute": int(pd.isna(df[CLOUD_COL_SECTOR]).sum()) if CLOUD_COL_SECTOR in df.columns else None,
        "temporal_holdout_policy": {"test_last_n_years": TEST_LAST_N_YEARS},
        "scenarios": [],
        "recommended_for_writeup": "with_sector_cloud" if any(r.use_sector_cloud for r in results) else "main_cloud_only",
    }

    for r in results:
        report["scenarios"].append({
            "scenario_name": r.scenario_name,
            "use_sector_cloud": r.use_sector_cloud,
            "split_info": r.split_info,
            "imputation_info": r.imputation_info,
            "feature_meta": r.feature_meta,
            "metrics": r.metrics,
            "notes": r.notes,
        })

    _write_json(PATH_REPORT_JSON, report)

    print("[OK] Temporal ML report saved:", PATH_REPORT_JSON)
    print("[OK] Holdout predictions saved:", PATH_PRED_HOLDOUT)
    if os.path.exists(PATH_FEAT_IMP):
        print("[OK] Feature importance saved:", PATH_FEAT_IMP)

if __name__ == "__main__":
    main()