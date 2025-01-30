import socket
import threading
import torch
import torch.nn as nn
import io
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

# Server Program
def server_program():
    host = "0.0.0.0"
    port = 5001
    num_clients = 2
    global_iterations = 10  

    input_size = 11  
    num_classes = 3  
    global_model = SleepModel(input_size, num_classes)
    global_state = None  

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(num_clients)
    print(f"Server is listening on {host}:{port}")

    clients = []
    for _ in range(num_clients):
        conn, addr = server_socket.accept()
        clients.append(conn)
        print(f"Client {addr} connected.")
        # Send the number of global iterations to the client
        conn.sendall(struct.pack(">I", global_iterations))

    for iteration in range(global_iterations):
        print(f"\n--- Global Iteration {iteration + 1}/{global_iterations} ---")

        if iteration > 0:
            buffer = io.BytesIO()
            torch.save(global_state, buffer)
            model_data = buffer.getvalue()

            for client in clients:
                client.sendall(struct.pack(">I", len(model_data)))  # Send message size
                client.sendall(model_data)  # Send actual model
                print(f"Sent {len(model_data)} bytes to client {client.getpeername()}")

        client_models = []
        client_accuracies = []

        def handle_client(conn):
            try:
                print(f"Waiting for model update from {conn.getpeername()}...")

                buffer = b""
                while len(buffer) < 4:
                    part = conn.recv(4 - len(buffer))
                    if not part:
                        print(f"Client {conn.getpeername()} disconnected.")
                        return
                    buffer += part

                msg_size = struct.unpack(">I", buffer)[0]

                buffer = b""
                while len(buffer) < msg_size:
                    part = conn.recv(msg_size - len(buffer))
                    if not part:
                        print(f"Client {conn.getpeername()} disconnected mid-transfer.")
                        return
                    buffer += part

                print(f"Received {len(buffer)} bytes from {conn.getpeername()}")
                buffer_io = io.BytesIO(buffer)
                client_data = torch.load(buffer_io)
                client_models.append(client_data["state_dict"])
                client_accuracies.append(client_data["accuracy"])

            except Exception as e:
                print(f"Error receiving data from client: {e}")

        threads = []
        for client in clients:
            thread = threading.Thread(target=handle_client, args=(client,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if len(client_models) == num_clients:
            print("Aggregating client models...")
            global_state = {key: torch.stack([cm[key] for cm in client_models]).mean(dim=0) for key in client_models[0].keys()}
            global_model.load_state_dict(global_state)

            global_accuracy = sum(client_accuracies) / len(client_accuracies)
            print(f"Global Model Accuracy after iteration {iteration + 1}: {global_accuracy:.2f}")

    torch.save(global_model.state_dict(), "final_global_model.pth")
    print("Training complete. Final global model saved.")

    for client in clients:
        client.close()
    server_socket.close()

if __name__ == "__main__":
    server_program()
