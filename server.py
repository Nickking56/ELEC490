import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from pyngrok import ngrok
import threading
import requests

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

# Initialize Flask app
app = Flask(__name__)

# Federated Learning Parameters
global_iterations = 15  
num_clients_required = 2  
connected_clients = {}  # Stores {"client_id": "IP:PORT"}
received_models = []  
received_accuracies = []
current_round = 1
global_model = SleepModel(11, 3)
global_state = {key: val.clone() for key, val in global_model.state_dict().items()}  
round_completed = False  # ✅ Ensures the server doesn't send updates prematurely

@app.route('/')
def home():
    return jsonify({"message": "Federated Learning Server Running"}), 200

@app.route('/connect', methods=['POST'])
def connect_client():
    """Client signals connection and registers its IP:PORT."""
    global connected_clients

    data = request.json
    client_address = data.get("client_address")

    if client_address:
        client_id = len(connected_clients) + 1  # Assign a unique client ID
        connected_clients[client_id] = client_address
        print(f"✅ Client {client_id} connected on {client_address}. Total clients: {len(connected_clients)}/{num_clients_required}")
    else:
        return jsonify({"status": "error", "message": "Missing client address"}), 400

    return jsonify({"status": "connected", "client_count": len(connected_clients), "global_iterations": global_iterations}), 200

@app.route('/ready', methods=['GET'])
def check_ready():
    """Check if all required clients have connected before training starts."""
    if len(connected_clients) >= num_clients_required:
        return jsonify({"status": "ready", "message": "All clients connected. Start training!"}), 200
    return jsonify({"status": "waiting", "message": f"Waiting for more clients ({len(connected_clients)}/{num_clients_required})"}), 200

@app.route('/upload', methods=['POST'])
def receive_weights():
    """Receive and store model weights from clients."""
    global received_models, received_accuracies, current_round, global_state, round_completed

    try:
        data = request.json  
        weights = [np.array(w) for w in data["weights"]]
        accuracy = data["accuracy"]

        received_models.append(weights)
        received_accuracies.append(accuracy)

        print(f"📩 Received model update from client | Accuracy: {accuracy:.2f} | Round {current_round}")

        if len(received_models) == num_clients_required:
            aggregate_models()  # Trigger aggregation once all clients send updates

        return jsonify({"status": "success", "message": "Model received"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

def aggregate_models():
    """Perform Federated Averaging (FedAvg)."""
    global global_state, received_models, received_accuracies, current_round, round_completed

    print(f"🔄 Aggregating client models for round {current_round}...")

    if len(received_models) != num_clients_required:
        print("⚠️ Not all clients have sent their models yet. Waiting...")
        return  # Wait until all clients send data

    # Convert received models to tensor format and perform averaging
    avg_weights = [torch.tensor(sum(layer) / len(received_models)) for layer in zip(*received_models)]

    # Update global model state
    for param, new_weights in zip(global_model.state_dict().keys(), avg_weights):
        global_state[param] = new_weights

    global_model.load_state_dict(global_state)

    print(f"✅ Global Model Aggregated for round {current_round}")

    # ✅ Send the updated model to all connected clients
    send_global_model_to_clients()

    # ✅ Reset received models for next round
    received_models.clear()
    received_accuracies.clear()
    round_completed = True
    current_round += 1

def send_global_model_to_clients():
    """Send the updated global model to all connected clients."""
    global global_state, current_round

    model_data = {
        "status": "success",
        "weights": [global_state[key].tolist() for key in global_state],
        "round": current_round
    }

    print(f"📤 Sending global model for round {current_round} to {len(connected_clients)} clients...")

    for client_id, client_address in connected_clients.items():
        try:
            response = requests.post(f"http://{client_address}/receive", json=model_data)
            if response.status_code == 200:
                print(f"✅ Successfully sent model to Client {client_id} at {client_address}.")
            else:
                print(f"⚠️ Failed to send model to Client {client_id}. Response: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error sending model to Client {client_id}: {e}")

@app.route('/download', methods=['GET'])
def send_weights():
    """Legacy endpoint - clients no longer request updates, server pushes them."""
    return jsonify({"status": "error", "message": "Clients no longer need to request updates. Server pushes models now."}), 400

def run_ngrok():
    """Start Ngrok Tunnel"""
    ngrok.set_auth_token("2sB8FAfQjsOAKPY1RKWWkaXx3Up_5BQ4NYgKiLDmTeNjMrVKz")  
    pub_url = ngrok.connect(addr="https://localhost:5001", proto="http", url="special-dolphin-incredibly.ngrok-free.app")
    ngrok_url = pub_url.public_url.replace('http://', 'https://')
    print(f"Ngrok URL: {ngrok_url}")

if __name__ == '__main__':
    threading.Thread(target=run_ngrok, daemon=True).start()
    app.run(host="0.0.0.0", port=5001, ssl_context=('cert.pem', 'key.pem'), debug=False)
