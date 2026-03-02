import sys
import importlib

REQUIRED = [
    "pandas",
    "numpy",
    "pyarrow",
    "sklearn",
    "matplotlib",
    "xgboost",
]

def main():
    missing = []
    for pkg in REQUIRED:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing.append(pkg)

    if missing:
        print("[ERRO] Pacotes faltando:", missing)
        print("Instale com: pip install -r requirements.txt")
        sys.exit(1)

    print("[OK] Ambiente Python ok. Pacotes principais disponíveis.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
