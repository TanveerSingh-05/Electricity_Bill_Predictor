import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib
import os

#synthetic dataset
np.random.seed(42)

n = 1000

data = {
    "family_members": np.random.randint(1, 8, n),
    "ac_count": np.random.randint(0, 4, n),
    "ac_hours": np.random.uniform(0, 12, n),
    "fan_count": np.random.randint(1, 9, n),
    "fan_hours": np.random.uniform(2, 20, n),
    "fridge_count": np.random.randint(1, 3, n),
    "washing_loads": np.random.randint(0, 9, n),
    "tv_count": np.random.randint(0, 4, n),
    "tv_hours": np.random.uniform(0, 10, n),
    "lights_count": np.random.randint(2, 12, n),
    "light_hours": np.random.uniform(2, 12, n)
}

df = pd.DataFrame(data)

#monthly electricity consumption
df["monthly_kwh"] = (
    df["ac_count"] * df["ac_hours"] * 30 * 1.2
    + df["fan_count"] * df["fan_hours"] * 30 * 0.075
    + df["fridge_count"] * 1.2 * 30
    + df["washing_loads"] * 4.3 * 0.7
    + df["tv_count"] * df["tv_hours"] * 30 * 0.1
    + df["lights_count"] * df["light_hours"] * 30 * 0.012
)

#random variation
df["monthly_kwh"] += np.random.normal(0, 15, n)

df["monthly_kwh"] = df["monthly_kwh"].clip(lower=0)

#saving dataset
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/electricity.csv", index=False)

print("Dataset created successfully!")
print(df.head())

#input and output
X = df.drop("monthly_kwh", axis=1)
y = df["monthly_kwh"]

#dataset split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#ann
model = Pipeline([
    ("scaler", StandardScaler()),

    ("ann", MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        early_stopping=True,
        random_state=42
    ))
])

#training
model.fit(X_train, y_train)

print("\nANN Training Completed!")

#prediction making
predictions = model.predict(X_test)

#evvaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("----------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

#saving model
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/ann_model.pkl")

print("\nModel saved successfully!")