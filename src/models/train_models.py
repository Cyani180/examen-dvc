import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Load data
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

# Load best parameters
best_params = joblib.load("models/best_params.pkl")

# Train model
model = GradientBoostingRegressor(**best_params)
model.fit(X_train, y_train.values.ravel())

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Save model with expected filename for DVC
joblib.dump(model, "models/model.pkl")
print("Model training completed and model saved.")
