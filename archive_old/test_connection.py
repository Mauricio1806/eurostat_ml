import os
from dotenv import load_dotenv
import databricks.sql as sql

load_dotenv(dotenv_path=".env")

server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
access_token = os.getenv("DATABRICKS_TOKEN")

print("HOST ok?", bool(server_hostname))
print("HTTP_PATH ok?", bool(http_path))
print("TOKEN ok?", bool(access_token))

with sql.connect(
    server_hostname=server_hostname,
    http_path=http_path,
    access_token=access_token,
) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 as ok")
        print(cur.fetchall())

