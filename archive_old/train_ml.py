import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("gold_model_dataset.csv")

# limpeza básica
df = df.dropna(subset=["value_added_real", "cloud_intensity"])

X = df[["cloud_intensity"]]
y = df["value_added_real"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))
print("coef:", model.coef_[0])
print("intercept:", model.intercept_)
