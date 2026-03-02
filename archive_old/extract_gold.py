import os
from dotenv import load_dotenv
import databricks.sql as dbsql
import pandas as pd

load_dotenv(".env")

SERVER = os.getenv("DATABRICKS_SERVER_HOSTNAME")
HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
TOKEN = os.getenv("DATABRICKS_TOKEN")

QUERY = """
SELECT
  sector,
  year,
  value_added_real,
  cloud_intensity
FROM default.gold_model_dataset
ORDER BY sector, year
"""

with dbsql.connect(
    server_hostname=SERVER,
    http_path=HTTP_PATH,
    access_token=TOKEN
) as conn:
    df = pd.read_sql(QUERY, conn)

print(df.head())
print("rows:", len(df))

# salva local para o ML
df.to_csv("gold_model_dataset.csv", index=False)
print("Saved: gold_model_dataset.csv")
