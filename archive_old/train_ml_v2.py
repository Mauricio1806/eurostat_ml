import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("gold_model_dataset.csv")

# limpeza básica
df = df.dropna(subset=["value_added_real", "cloud_intensity", "sector", "year"])

X = df[["cloud_intensity", "sector", "year"]]
y = df["value_added_real"]

# transforma 'sector' em one-hot e mantém números como estão
preprocess = ColumnTransformer(
    transformers=[
        ("sector_ohe", OneHotEncoder(handle_unknown="ignore"), ["sector"]),
    ],
    remainder="passthrough"  # mantém cloud_intensity e year
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Rows used:", len(df))
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))
