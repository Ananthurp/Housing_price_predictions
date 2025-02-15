# src/data_preprocessing.py
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Calculate the project root dynamically so that file paths are correct no matter where you run the script from.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def load_raw_data(filename="california_housing.csv"):
    """
    Loads raw data from the data/raw folder.
    """
    file_path = os.path.join(RAW_DIR, filename)
    print(f"Loading raw data from: {file_path}")
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Cleans and processes the data.
    
    For the California housing dataset, typical steps might include:
    - Dropping missing values (if any)
    - Scaling numerical features (excluding the target)
    - Removing duplicates
    """
    df_clean = df.copy()
    
    # Drop rows with missing values (modify as necessary)
    df_clean.dropna(inplace=True)
    
    # Define the target column and feature columns
    target_column = "MedHouseVal"  # Ensure this matches your CSV column name
    feature_columns = list(df_clean.columns)
    feature_columns.remove(target_column)
    
    # Scale the feature columns using StandardScaler
    scaler = StandardScaler()
    df_clean[feature_columns] = scaler.fit_transform(df_clean[feature_columns])
    
    return df_clean

def save_processed_data(df, filename="california_housing_processed.csv"):
    """
    Saves the processed data to the data/processed folder.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    file_path = os.path.join(PROCESSED_DIR, filename)
    df.to_csv(file_path, index=False)
    print(f"Processed data saved to: {file_path}")

if __name__ == "__main__":
    # Load raw data
    raw_df = load_raw_data()
    
    # Preprocess the data
    processed_df = preprocess_data(raw_df)
    
    # Save the processed data
    save_processed_data(processed_df)
