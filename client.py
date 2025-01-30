import requests
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys
import threading
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

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

SERVER_URL = "https://special-dolphin-incredibly.ngrok-free.app"  # Replace with actual Ngrok URL

# Global variables for tracking model updates
global_model_lock = threading.Lock()  # Lock to handle global model updates safely
global_model_received = None
global_round = 0  # Track the current round

@app.route('/receive', methods=['POST'])
def receive_model():
    """Receive updated global model from the server."""
    global global_model_received, global_round

    data = request.json
    server_round = data.get("round")
    new_weights = data.get("weights")

    with global_model_lock:
        if server_round == global_round + 1:  # ✅ Ensure update matches expected round
            print(f"✅ Client received the global model update for round {server_round}.")
            global_model_received = new_weights
            global_round = server_round  # ✅ Move to the next round
        else:
            print(f"⚠️ Received update for round {server_round}, but expected {global_round + 1}. Ignoring.")

    return jsonify({"status": "received"}), 200

def start_flask_server(port):
    """Run the Flask server in a separate thread."""
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

def client_program(client_id, data_dir):
    global global_model_received, global_round

    # Assign a unique port based on `5000 + client_id + 1`
    client_port = 5000 + client_id + 1  
    print(f"🔄 Client {client_id} will use port {client_port} to receive global model updates.")

    X_client = pd.read_csv(os.path.join(data_dir, f"client_{client_id}", "X_client.csv"))
    y_client = pd.read_csv(os.path.join(data_dir, f"client_{client_id}", "y_client.csv"))

    X_tensor = torch.tensor(X_client.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_client.values.flatten(), dtype=torch.long)

    input_size = X_client.shape[1]
    num_classes = len(y_client["Sleep Disorder"].unique())

    model = SleepModel(input_size, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Start Flask server only once in a separate thread
    flask_thread = threading.Thread(target=start_flask_server, args=(client_port,), daemon=True)
    flask_thread.start()

    # Notify the server of the connection with the correct port
    response = requests.post(f"{SERVER_URL}/connect", json={"client_address": f"localhost:{client_port}"}).json()
    global_iterations = response["global_iterations"]
    print(f"Client {client_id} connected. Global iterations set to {global_iterations}. Waiting for all clients to join...")

    while True:
        response = requests.get(f"{SERVER_URL}/ready")
        status = response.json()["status"]
        if status == "ready":
            break
        print(f"Client {client_id}: {response.json()['message']}")
        time.sleep(5)

    for iteration in range(global_iterations):  
        print(f"Client {client_id} - Starting Training (Iteration {iteration + 1}/{global_iterations})")

        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y_tensor).sum().item() / len(y_tensor)

        print(f"Client {client_id} - Local Accuracy: {accuracy:.2f}")

        # Send model update to server
        model_weights = [p.data.numpy().tolist() for p in model.parameters()]
        requests.post(f"{SERVER_URL}/upload", json={"weights": model_weights, "accuracy": accuracy})

        print(f"Client {client_id} - Sent local model to server for iteration {iteration + 1}. Now waiting for the global model...")

        # ✅ Clients now wait without polling
        global_model_received = None  # Reset received model
        expected_round = iteration + 1

        while True:
            with global_model_lock:
                if global_model_received is not None and global_round == expected_round:
                    print(f"✅ Client {client_id} - Applying new global model for round {global_round}.")
                    for param, new_weights in zip(model.parameters(), global_model_received):
                        param.data = torch.tensor(new_weights, dtype=torch.float32)
                    break  # Exit waiting loop once model is updated

            print(f"⏳ Client {client_id} - Still waiting for global model update for round {expected_round}...")
            time.sleep(3)

if __name__ == "__main__":
    client_id = int(sys.argv[1])  # Client ID from command line
    data_dir = "client_data"
    client_program(client_id, data_dir)
