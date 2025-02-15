import os
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Calculate the base directory (project root)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

def fetch_and_save_data(filename="california_housing.csv", as_frame=True):
    data = fetch_california_housing(as_frame=as_frame)
    df = data.frame if as_frame else pd.DataFrame(data.data, columns=data.feature_names)
    if as_frame:
        df["MedHouseVal"] = data.target
    else:
        df["MedHouseVal"] = data.target

    file_path = os.path.join(RAW_DIR, filename)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

if __name__ == "__main__":
    fetch_and_save_data()

