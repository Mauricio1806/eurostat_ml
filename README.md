# Cloud Computing Adoption vs Economic Performance (Germany) — Databricks + ML

This repository implements an end-to-end **data engineering + analytics** pipeline to study the association between **cloud adoption** and **economic performance** in Germany, using public and aggregated European data.

The project follows a modern lakehouse approach (**Bronze → Silver → Gold**) implemented in **Databricks (Delta Lake)** and connects to **VS Code** via **Databricks SQL Connector** to run machine learning experiments and scenario simulations.

---

## Project Goals

- Build a reproducible **data pipeline** in Databricks using the **Bronze–Silver–Gold** architecture.
- Create an analytical dataset combining:
  - **Economic performance proxy** (e.g., sector value added / productivity indicators)
  - **Cloud adoption proxy** (`cloud_intensity`)
- Train baseline ML models and run **forecast simulations (2026–2030)** under different cloud adoption assumptions.

> Important: this is an **exploratory and predictive** project using **public, aggregated** data. It does **not** claim causal inference.

---

## Architecture (Bronze → Silver → Gold)

### Bronze (Raw ingestion)
- Stores raw datasets in original format for traceability and auditability.

### Silver (Cleaning and standardization)
- Handles missing values, duplicates, schema normalization, type casting, and harmonization of sector/year keys.

### Gold (Analytics-ready)
- Produces the final dataset used for modeling:
  - `sector`
  - `year`
  - `value_added_real` (economic performance proxy)
  - `cloud_intensity` (cloud adoption proxy)

---

## Repository Structure

- `databricks_sql/`: SQL scripts to build Silver/Gold tables in Databricks
- `src/`: Python scripts (VS Code) for extraction, ML training, and forecasting
- `data/`: small sample CSVs only (no secrets / no full raw datasets)
- `outputs/`: sample outputs and metrics
- `docs/`: short documentation + screenshots

---

## Setup

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
2) Install dependencies
pip install -r requirements.txt

3) Configure Databricks credentials

Create a .env file in the project root (DO NOT COMMIT THIS FILE):

DATABRICKS_SERVER_HOSTNAME=dbc-xxxx.cloud.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/xxxxxxxxxxxxxxxx
DATABRICKS_TOKEN=YOUR_TOKEN_HERE


You can copy from .env.example.

Databricks SQL (build tables)

Run the SQL scripts in this order:

01_silver_gva.sql

02_gold_gva.sql

03_silver_cloud_long.sql

04_gold_cloud_de.sql

05_gold_model_dataset.sql

After that, validate the final table:

SELECT * FROM default.gold_model_dataset ORDER BY sector, year;

Run (VS Code / Python)
1) Test connection to Databricks SQL
python src/test_connection.py

2) Extract Gold dataset to local CSV
python src/extract_gold.py


This generates:

gold_model_dataset.csv (local)

3) Baseline model (simple regression)
python src/train_ml.py

4) Model with sector + year + cloud_intensity (OneHot + LinearRegression)
python src/train_ml_v2.py

5) Sector-level models
python src/train_by_sector.py

6) Forecasts

Next-year forecast:

python src/forecast_next_year.py


Multi-year scenario simulation (2026–2030):

python src/forecast_multi_year.py

Outputs

Typical outputs include:

forecast_next_year.csv

forecast_multi_year.csv

results_by_sector.csv

Only small samples are stored in data/ and outputs/ in this repository.

Notes on Interpretation

cloud_intensity is an aggregated proxy (macro/annual).
Depending on the data join, it can be shared across sectors in the same year.
This affects interpretation: sector differences may be driven mainly by sector and year, not only by cloud adoption.

This project is predictive/exploratory and does not establish causal effects.

Data Sources (Public)

Examples of sources used:

Bitkom Research — Cloud Monitor (Germany)

OECD — Digital Transformation Indicators

Eurostat — sector macroeconomic and productivity indicators

Supporting context: ifo Institute, PwC, KPMG (not directly merged into the analytical dataset)
