"""
train_model.py

Trains a Linear Regression model to predict house prices using the
King County House Sales dataset (kc_house_data.csv), and saves the
trained model as house_price_model.pkl for use in app.py.

Features used (kept in sync with app.py's inputs):
    - sqft_living
    - bedrooms
    - bathrooms
    - floors
    - sqft_living15

Run:
    python train_model.py
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ---- Load data ----
df = pd.read_csv("kc_house_data.csv")

FEATURES = ["sqft_living", "bedrooms", "bathrooms", "floors", "sqft_living15"]
TARGET = "price"

X = df[FEATURES]
y = df[TARGET]

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Train model ----
model = LinearRegression()
model.fit(X_train, y_train)

# ---- Evaluate ----
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Model trained on {len(X_train)} rows, tested on {len(X_test)} rows.")
print(f"MAE:  ${mae:,.2f}")
print(f"R^2:  {r2:.4f}")

# ---- Save model ----
joblib.dump(model, "house_price_model.pkl")
print("Saved trained model to house_price_model.pkl")
