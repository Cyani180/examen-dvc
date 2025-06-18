import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Path to the raw data file
RAW_DATA_PATH = "data/raw/raw.csv"
PROCESSED_PATH = "data/processed/"

# Ensure directories exist
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Read in the data
df = pd.read_csv(RAW_DATA_PATH)

# Separate target variable and features
X = df.drop(columns=["silica_concentrate"])
y = df["silica_concentrate"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save results
X_train.to_csv(os.path.join(PROCESSED_PATH, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_PATH, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(PROCESSED_PATH, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_PATH, "y_test.csv"), index=False)
