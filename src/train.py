# src/train.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import src
# Import your modules
# from src.data_preprocessing import load_raw_data, preprocess_data, save_processed_data
# from src.model_scratch import LinearRegressionScratch
# from src.model_lib import train_model, evaluate_model

# Calculate base directory for consistency
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def load_processed_data(filename="california_housing_processed.csv"):
    """
    Loads processed data from the data/processed folder.
    """
    file_path = os.path.join(PROCESSED_DIR, filename)
    return pd.read_csv(file_path)

def prepare_features_targets(df):
    """
    Prepares the feature matrix X and target vector y.
    Here, 'MedHouseVal' is used as the target variable.
    """
    target_column = "MedHouseVal"
    X = df.drop(target_column, axis=1).values
    y = df[target_column].values
    return X, y

def main():
    # Option 1: If you haven't run preprocessing separately, you can run it here
    # raw_df = load_raw_data()
    # processed_df = preprocess_data(raw_df)
    # save_processed_data(processed_df)
    # Option 2: Directly load the processed data if already saved
    processed_df = load_processed_data()

    X, y = prepare_features_targets(processed_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Approach 1: From Scratch ---
    print("Training using from-scratch implementation...")
    model_scratch = src.model_scratch.LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
    model_scratch.fit(X_train, y_train)
    predictions_scratch = model_scratch.predict(X_test)
    mse_scratch = np.mean((predictions_scratch - y_test) ** 2)
    print("Scratch Model MSE:", mse_scratch)

    # --- Approach 2: Using scikit-learn ---
    print("Training using scikit-learn implementation...")
    model_lib = src.model_lib.train_model(X_train, y_train)
    mse_lib, predictions_lib = src.model_lib.evaluate_model(model_lib, X_test, y_test)
    print("Library Model MSE:", mse_lib)

if __name__ == "__main__":
    main()
