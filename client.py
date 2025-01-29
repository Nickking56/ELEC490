import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import pandas as pd
import os
import sys

# Define the neural network model
class SleepModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SleepModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

SERVER_URL = "https://special-dolphin-incredibly.ngrok-free.app" 

def client_program(client_id, data_dir):
    # Load client data
    X_client = pd.read_csv(os.path.join(data_dir, f"client_{client_id}", "X_client.csv"))
    y_client = pd.read_csv(os.path.join(data_dir, f"client_{client_id}", "y_client.csv"))

    X_tensor = torch.tensor(X_client.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_client.values.flatten(), dtype=torch.long)

    input_size = X_client.shape[1]
    num_classes = len(y_client["Sleep Disorder"].unique())

    model = SleepModel(input_size, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for iteration in range(10):  # Match global iterations
        print(f"Client {client_id} - Starting Training (Iteration {iteration + 1})")

        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate accuracy
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y_tensor).sum().item() / len(y_tensor)

        print(f"Client {client_id} - Local Accuracy: {accuracy:.2f}")

        # Convert model to NumPy
        model_weights = [p.data.numpy().tolist() for p in model.parameters()]


        # Send model update to server
        response = requests.post(f"{SERVER_URL}/upload", json={
            "weights": model_weights,
            "accuracy": accuracy
        })

        if response.status_code == 200:
            print("Model uploaded successfully.")

        # Get updated global model
        response = requests.get(f"{SERVER_URL}/download")
        if response.status_code == 200:
            global_weights = response.json().get("weights")
            if global_weights:
                for param, new_weights in zip(model.parameters(), global_weights):
                    param.data = torch.tensor(np.array(new_weights), dtype=torch.float32)
                print("Updated model with new global weights.")

if __name__ == "__main__":
    client_id = int(sys.argv[1])
    data_dir = "client_data"
    client_program(client_id, data_dir)
