import socket
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import io
import sys
import struct

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

# Client Program
def client_program(client_id, data_dir, host="0.tcp.ngrok.io", port=19259):  # Use ngrok's TCP address
    print(f"Client {client_id} attempting to connect to server at {host}:{port}...")

    # Load client data
    X_client = pd.read_csv(os.path.join(data_dir, f"client_{client_id}", "X_client.csv"))
    y_client = pd.read_csv(os.path.join(data_dir, f"client_{client_id}", "y_client.csv"))

    X_tensor = torch.tensor(X_client.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_client.values.flatten(), dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Try to connect with retries
    connected = False
    while not connected:
        try:
            client_socket.connect((host, port))
            connected = True
            print(f"Client {client_id} connected to server at {host}:{port}")
        except socket.error as e:
            print(f"Connection attempt failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            continue

    # Send ready signal to server
    print(f"Client {client_id} waiting for all clients to be ready...")
    ready_message = struct.pack(">I", 1)  # 1 means ready
    client_socket.sendall(ready_message)
    
    # Wait for server to signal all clients are ready
    buffer = b""
    while len(buffer) < 4:
        buffer += client_socket.recv(4 - len(buffer))
    start_signal = struct.unpack(">I", buffer)[0]
    
    if start_signal == 1:
        print(f"All clients are ready. Starting training synchronously.")
    else:
        print(f"Received unexpected start signal: {start_signal}")
        return

    # Receive number of global iterations
    buffer = b""
    while len(buffer) < 4:
        buffer += client_socket.recv(4 - len(buffer))
    global_iterations = struct.unpack(">I", buffer)[0]

    # Setup model
    input_size = X_client.shape[1]
    num_classes = len(y_client["Sleep Disorder"].unique())
    model = SleepModel(input_size, num_classes)

    # Main training loop
    for iteration in range(global_iterations):
        print(f"\nClient {client_id} - Starting iteration {iteration + 1}/{global_iterations}...")

        # Receive global model from server (except for first iteration)
        if iteration > 0:
            print(f"Receiving global model from server...")
            
            # Receive model size
            buffer = b""
            while len(buffer) < 4:
                buffer += client_socket.recv(4 - len(buffer))
            msg_size = struct.unpack(">I", buffer)[0]
            
            # Receive model weights
            buffer = b""
            while len(buffer) < msg_size:
                buffer += client_socket.recv(msg_size - len(buffer))
            
            # Load received weights
            buffer_io = io.BytesIO(buffer)
            global_state = torch.load(buffer_io)
            model.load_state_dict(global_state)
            print(f"Client {client_id} received aggregated weights.")

        # Train local model
        print(f"Training local model...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model.train()
        
        for epoch in range(5):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Evaluate local model
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y_tensor).sum().item() / len(y_tensor)
        
        print(f"Client {client_id} local accuracy after iteration {iteration + 1}: {accuracy:.2f}")
        
        # Send updated weights to server
        client_data = {"state_dict": model.state_dict(), "accuracy": accuracy}
        buffer = io.BytesIO()
        torch.save(client_data, buffer)
        message = buffer.getvalue()
        
        # Send message size
        client_socket.sendall(struct.pack(">I", len(message)))
        # Send message
        client_socket.sendall(message)
        
        print(f"Client {client_id} sent updated weights to server.")
        print(f"Iteration {iteration + 1} complete.")

    # Close the socket
    client_socket.close()
    print(f"Client {client_id} training complete.")

if __name__ == "__main__":
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    data_dir = "client_data"
    
    # Don't forget to import time for the retry logic
    import time
    
    client_program(client_id, data_dir)