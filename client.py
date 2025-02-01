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
def client_program(client_id, data_dir, host="6.tcp.ngrok.io", port=17926):  # Use ngrok's TCP address
    print(f"Client {client_id} attempting to connect to server at {host}:{port}...")

    X_client = pd.read_csv(os.path.join(data_dir, f"client_{client_id}", "X_client.csv"))
    y_client = pd.read_csv(os.path.join(data_dir, f"client_{client_id}", "y_client.csv"))

    X_tensor = torch.tensor(X_client.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_client.values.flatten(), dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print(f"Client {client_id} connected to server at {host}:{port}")

    buffer = b""
    while len(buffer) < 4:
        buffer += client_socket.recv(4 - len(buffer))

    global_iterations = struct.unpack(">I", buffer)[0]

    input_size = X_client.shape[1]
    num_classes = len(y_client["Sleep Disorder"].unique())

    model = SleepModel(input_size, num_classes)

    for iteration in range(global_iterations):
        print(f"\nClient {client_id} - Starting local training (Iteration {iteration + 1}/{global_iterations})...")

        if iteration > 0:
            buffer = b""
            while len(buffer) < 4:
                buffer += client_socket.recv(4 - len(buffer))

            msg_size = struct.unpack(">I", buffer)[0]

            buffer = b""
            while len(buffer) < msg_size:
                buffer += client_socket.recv(msg_size - len(buffer))

            buffer_io = io.BytesIO(buffer)
            global_state = torch.load(buffer_io)
            model.load_state_dict(global_state)

            print(f"Client {client_id} received aggregated weights. Beginning next round of training.")

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

        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y_tensor).sum().item() / len(y_tensor)

        print(f"Client {client_id} local accuracy after iteration {iteration + 1}: {accuracy:.2f}")

        client_data = {"state_dict": model.state_dict(), "accuracy": accuracy}
        buffer = io.BytesIO()
        torch.save(client_data, buffer)
        message = buffer.getvalue()

        client_socket.sendall(struct.pack(">I", len(message)))
        client_socket.sendall(message)
        print(f"Client {client_id} sent updated weights to server.")

    client_socket.close()
    print(f"Client {client_id} training complete.")

if __name__ == "__main__":
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    data_dir = "client_data"
    client_program(client_id, data_dir)
