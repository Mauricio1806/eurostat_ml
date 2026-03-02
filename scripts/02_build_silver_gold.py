import os, runpy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET = os.path.join(ROOT, "scripts", "01_build_silver_gold.py")

if not os.path.exists(TARGET):
    raise FileNotFoundError(f"Não achei: {TARGET}")

print("[RUN] 02_build_silver_gold ->", TARGET)
runpy.run_path(TARGET, run_name="__main__")
