# src/model_lib.py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse, predictions

# Example test if needed
if __name__ == "__main__":
    import numpy as np
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    model = train_model(X, y)
    mse, predictions = evaluate_model(model, X, y)
    print("MSE:", mse)
    print("Predictions:", predictions)
