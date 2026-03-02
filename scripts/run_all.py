import os
import runpy
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS = os.path.join(ROOT, "scripts")
OUT = os.path.join(ROOT, "output")

def run(script_name):
    path = os.path.join(SCRIPTS, script_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Script não encontrado: {path}")
    print(f"\n================ RUN {script_name} ================\n")
    runpy.run_path(path, run_name="__main__")

def main():
    os.makedirs(OUT, exist_ok=True)

    run("00_check_environment.py")

    bronze_gva = os.path.join(OUT, "bronze_gva_de.parquet")
    bronze_cloud = os.path.join(OUT, "bronze_cloud_de.parquet")

    if not (os.path.exists(bronze_gva) and os.path.exists(bronze_cloud)):
        run("01_build_bronze.py")
    else:
        print("[SKIP] Bronze já existe em output/. Reutilizando.")

    # seus scripts atuais são chamados pelos wrappers
    run("02_build_silver_gold.py")
    run("03_train_ml.py")
    run("04_simulate.py")

    # HTML report
    run("05_make_report_html.py")

    print("\n[FINAL] Pipeline completo. Veja output/ (CSV/JSON/HTML).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO] Falhou:", repr(e))
        sys.exit(1)
