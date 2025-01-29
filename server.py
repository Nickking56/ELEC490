import torch
import torch.nn as nn
import io
import json
import numpy as np
from flask import Flask, request, jsonify
from pyngrok import ngrok
import threading

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
global_iterations = 10  # Change this as needed
num_clients = 2
global_model = SleepModel(11, 3)
global_state = None  # Stores the latest global model weights
received_models = []  # Stores client model updates
received_accuracies = []

@app.route('/')
def home():
    return jsonify({"message": "Federated Learning Server Running"}), 200

@app.route('/upload', methods=['POST'])
def receive_weights():
    """Receive and store model weights from clients"""
    global received_models, received_accuracies

    try:
        data = request.json  # Expecting JSON
        weights = np.array(data["weights"])
        accuracy = data["accuracy"]

        received_models.append(weights)
        received_accuracies.append(accuracy)

        print(f"Received model update from client | Accuracy: {accuracy:.2f}")

        # If all clients sent updates, aggregate
        if len(received_models) >= num_clients:
            aggregate_models()
        
        return jsonify({"status": "success", "message": "Model received"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

def aggregate_models():
    """Perform Federated Averaging (FedAvg)"""
    global global_state, received_models, received_accuracies

    print("Aggregating client models...")

    avg_weights = np.mean(np.array(received_models), axis=0).tolist()
    global_state = avg_weights  # Store updated global model
    received_models = []  # Reset for next round
    received_accuracies = []  # Reset accuracies

    print(f"Global Model Aggregated | Clients Processed: {num_clients}")

@app.route('/download', methods=['GET'])
def send_weights():
    """Send latest global model to clients"""
    global global_state

    if global_state is None:
        return jsonify({"status": "error", "message": "No global model available"}), 400
    
    return jsonify({
        "status": "success",
        "weights": global_state
    }), 200

def run_ngrok():
    """Start Ngrok Tunnel"""
    ngrok.set_auth_token("2sB8FAfQjsOAKPY1RKWWkaXx3Up_5BQ4NYgKiLDmTeNjMrVKz")  # Replace with your actual Ngrok token
    pub_url = ngrok.connect(addr="https://localhost:5001", proto = "http", url = "special-dolphin-incredibly.ngrok-free.app")
    ngrok_url = pub_url.public_url.replace('http://', 'https://')
    print(f"Ngrok URL: {ngrok_url}")

if __name__ == '__main__':
    threading.Thread(target=run_ngrok, daemon=True).start()
    app.run(host="0.0.0.0", port=5001, ssl_context=('cert.pem', 'key.pem'), debug=False)
