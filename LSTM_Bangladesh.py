# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 17:47:33 2025

@author: Dell
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load datasets
df4 = pd.read_feather("df_finalArea4.feather")
df3 = pd.read_feather("df_finalArea3.feather")
df2 = pd.read_feather("df_finalArea2.feather")
new_df1 = pd.read_feather("df_finalArea1.feather")

df = pd.concat([df4, df3, df2])

features = ['NDVI_Value', 'Albedo_Value', 'lst_Value', 'GEOCODE', 
            'DISTCODE', 'precipitation_Value','wind_Value','wind_Direction','uhi_Value']
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

def create_dataset(data, time_step=12):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])  # Last 12 months as input
        y.append(data[i + time_step, -1])  # Next month's UHI Value
    return np.array(X), np.array(y)

districts = df['DISTNAME'].unique()
all_X, all_y = [], []

for district in districts:
    district_data = df[df['DISTNAME'] == district][features].values
    X, y = create_dataset(district_data, time_step=12)
    all_X.append(X)
    all_y.append(y)

all_X = np.concatenate(all_X, axis=0)
all_y = np.concatenate(all_y, axis=0)
all_X = all_X.reshape(all_X.shape[0], all_X.shape[1], all_X.shape[2])

X_tensor = torch.tensor(all_X, dtype=torch.float32)
y_tensor = torch.tensor(all_y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, y_tensor)

# Sequential Split
train_size = int(0.8 * len(dataset))
X_train, X_test = all_X[:train_size], all_X[train_size:]
y_train, y_test = all_y[:train_size], all_y[train_size:]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

model = LSTMModel(len(features), 512, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluation
model.eval()
predictions, actual = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        predictions.append(y_pred.numpy())
        actual.append(y_batch.numpy())

predictions = np.concatenate(predictions, axis=0)
actual = np.concatenate(actual, axis=0)

mse = mean_squared_error(actual, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual, predictions)
r2 = r2_score(actual, predictions)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R² Score: {r2}")

# Predict on new dataset
new_df1[features] = scaler.transform(new_df1[features])

def predict_for_month(new_df, model, target_year, target_month, time_step=12):
    new_df = new_df.sort_values(by=["DISTNAME", "Year", "Month"])
    new_df[features] = scaler.transform(new_df[features])
    
    districts = new_df["DISTNAME"].unique()
    all_X_new, actual_y, district_names = [], [], []
    
    for district in districts:
        district_data = new_df[new_df["DISTNAME"] == district]
        district_data = district_data[(district_data["Year"] < target_year) | 
                                      ((district_data["Year"] == target_year) & (district_data["Month"] < target_month))]
        if len(district_data) >= time_step:
            X_new = district_data[-time_step:][features].values
            all_X_new.append(X_new)
            district_names.append(district)
            actual_row = new_df[(new_df["DISTNAME"] == district) &
                                (new_df["Year"] == target_year) &
                                (new_df["Month"] == target_month)]
            actual_y.append(actual_row["uhi_Value"].values[0])
    
    all_X_new = np.array(all_X_new)
    X_new_tensor = torch.tensor(all_X_new, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_new_tensor).numpy().flatten()
    
    results = pd.DataFrame({"DISTNAME": district_names, "Predicted_UHI": predictions, "Actual_UHI": actual_y})
    return results

predicted_results = predict_for_month(new_df1, model, 2014, 4)
print(predicted_results)




