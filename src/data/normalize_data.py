import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

X_train_path = "data/processed/X_train.csv"
X_test_path = "data/processed/X_test.csv"
X_train_scaled_path = "data/processed/X_train_scaled.csv"
X_test_scaled_path = "data/processed/X_test_scaled.csv"
scaler_path = "models/scaler.pkl"

# Load data
X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)

# Select only numeric columns (all columns with numbers)
X_train_num = X_train.select_dtypes(include=[np.number])
X_test_num = X_test[X_train_num.columns]  # use the same columns as training set

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_num)
X_test_scaled = scaler.transform(X_test_num)

# Save results as DataFrame to keep column names
pd.DataFrame(X_train_scaled, columns=X_train_num.columns).to_csv(X_train_scaled_path, index=False)
pd.DataFrame(X_test_scaled, columns=X_test_num.columns).to_csv(X_test_scaled_path, index=False)

# Save the scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, scaler_path)

print("Data normalized and saved.")
