import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("gold_model_dataset.csv")
df = df.dropna(subset=["value_added_real", "cloud_intensity", "sector", "year"])

results = []

MIN_ROWS = 4  # antes era 8, por isso provavelmente zerou tudo

for sector, g in df.groupby("sector"):
    if len(g) < MIN_ROWS:
        continue

    X = g[["cloud_intensity", "year"]]
    y = g["value_added_real"]

    # se ficar MUITO pequeno, pode dar problema no split; então protege
    if len(g) < 6:
        # treina e avalia no mesmo conjunto (só pra não quebrar)
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(X)

        results.append({
            "sector": sector,
            "rows": len(g),
            "mae": mean_absolute_error(y, pred),
            "r2": r2_score(y, pred),
            "note": "no_split_small_sample"
        })
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append({
        "sector": sector,
        "rows": len(g),
        "mae": mean_absolute_error(y_test, pred),
        "r2": r2_score(y_test, pred),
        "note": "train_test_split"
    })

out = pd.DataFrame(results)

if out.empty:
    print("Nenhum setor teve linhas suficientes após a limpeza. Tente diminuir MIN_ROWS ou revisar o dataset.")
else:
    out = out.sort_values("r2", ascending=False)
    print(out.to_string(index=False))
    out.to_csv("results_by_sector.csv", index=False)
    print("\nSaved: results_by_sector.csv")

