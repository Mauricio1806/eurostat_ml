import os, runpy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET = os.path.join(ROOT, "scripts", "02_train_ml_robust.py")

if not os.path.exists(TARGET):
    raise FileNotFoundError(f"Não achei: {TARGET}")

print("[RUN] 03_train_ml ->", TARGET)
runpy.run_path(TARGET, run_name="__main__")
