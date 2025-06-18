import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib

# Load data
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

# Drop rows with missing values
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]

# Convert datetime columns to features if exist
for col in X_train.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        X_train[col] = pd.to_datetime(X_train[col], errors='coerce')  # Convert to datetime or NaT
        X_train[f'{col}_year'] = X_train[col].dt.year
        X_train[f'{col}_month'] = X_train[col].dt.month
        X_train[f'{col}_day'] = X_train[col].dt.day
        X_train[f'{col}_hour'] = X_train[col].dt.hour
        X_train.drop(columns=[col], inplace=True)

# Select only numeric columns
X_train_numeric = X_train.select_dtypes(include=[np.number])

# Optional: print columns that were dropped
non_numeric_cols = X_train.columns.difference(X_train_numeric.columns)
if len(non_numeric_cols) > 0:
    print(f"Non-numeric columns dropped: {list(non_numeric_cols)}")

# Define model and parameters for GridSearch
model = GradientBoostingRegressor()
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.01]
}

gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit GridSearch with clean data
gs.fit(X_train_numeric, y_train.values.ravel())

# Save best params
joblib.dump(gs.best_params_, 'models/best_params.pkl')

print("GridSearch completed. Best parameters saved:", gs.best_params_)
