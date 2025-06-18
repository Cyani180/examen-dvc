import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import json

# Load data
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Load trained model
model = joblib.load("models/gradient_boosting_model.pkl")

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save predictions
predictions_df = pd.DataFrame({"y_test": y_test.values.ravel(), "y_pred": y_pred})
predictions_df.to_csv("data/processed/predictions.csv", index=False)

# Save metrics
metrics = {"MSE": mse, "R2": r2}
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f)

print("Model evaluation completed. Metrics and predictions saved.")
