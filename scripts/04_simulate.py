import os, runpy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET = os.path.join(ROOT, "scripts", "03_simulate_2026_2030.py")

if not os.path.exists(TARGET):
    raise FileNotFoundError(f"Não achei: {TARGET}")

print("[RUN] 04_simulate ->", TARGET)
runpy.run_path(TARGET, run_name="__main__")
