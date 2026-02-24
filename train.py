# train.py

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

print("Loading dataset...")
df = pd.read_csv("data/train.csv")

# Convert datetime
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

# Feature Engineering
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.dayofweek
df["is_weekend"] = df["day"].apply(lambda x: 1 if x >= 5 else 0)

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

print("Calculating distance...")
df["distance"] = haversine(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"]
)

# Select features
features = ["distance", "passenger_count", "hour", "day", "is_weekend"]

df = df[features + ["trip_duration"]].dropna()

X = df[features]
y = df["trip_duration"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Performance:")
print("R² Score:", r2)
print("MAE:", mae)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully in models/model.pkl")