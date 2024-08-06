import pandas as pd
import numpy as np

data = pd.read_excel("Doviz_Satislari_20050101_20231205_Training_Set.xlsx")

# Assuming the first column is the date and the next seven are input features, with the last being the output feature
input_features = data.columns[1:8]  # Adjust column indices as needed
output_feature = data.columns[-1]  # Assuming the last column is the output feature

def query_record(i):
    inputs = data.iloc[i][input_features]
    output = data.iloc[i][output_feature]
    print(f"Inputs: {inputs}\nOutput: {output}")

# Example usage
query_record(4512)  # Replace 4512 with any valid index

# Example simple model parameters (weights and bias)
weights = np.random.rand(7)  # 7 input features
bias = np.random.rand()

# Cost/Loss Function - Mean Squared Error (MSE)
def mse_cost(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Example Prediction Function
def predict(inputs):
    return np.dot(inputs, weights) + bias

lambda_l1 = 0.01  # L1 regularization coefficient
lambda_l2 = 0.01  # L2 regularization coefficient

def cost_with_regularization(y_true, y_pred, weights):
    l1_penalty = lambda_l1 * np.sum(np.abs(weights))
    l2_penalty = lambda_l2 * np.sum(weights ** 2)
    return mse_cost(y_true, y_pred) + l1_penalty + l2_penalty
