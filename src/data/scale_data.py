import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def main():
    input_folder = 'data/processed'
    output_folder = 'data/processed'

    X_train_path = os.path.join(input_folder, 'X_train.csv')
    X_test_path = os.path.join(input_folder, 'X_test.csv')

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)

    print("Daten-Typen von X_train:")
    print(X_train.dtypes)
    print(X_train.head())

    # Entferne Datums-Spalte, falls vorhanden
    if 'date' in X_train.columns:
        X_train = X_train.drop(columns=['date'])
    if 'date' in X_test.columns:
        X_test = X_test.drop(columns=['date'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(os.path.join(output_folder, 'X_train_scaled.csv'), index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(os.path.join(output_folder, 'X_test_scaled.csv'), index=False)

if __name__ == "__main__":
    main()
