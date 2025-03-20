import socket
import threading
import torch
import torch.nn as nn
import io
import struct
import matplotlib.pyplot as plt
import time

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
    num_clients = 3 
    global_iterations = 100

    input_size = 11  
    num_classes = 3  
    global_model = SleepModel(input_size, num_classes)
    global_state = None  

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(num_clients)
    print(f"Server is waiting for {num_clients} clients to connect...")

    clients = []
    client_ids = {}
    global_accuracies = []  # Store accuracy at each iteration

    # Accept client connections
    for i in range(num_clients):
        conn, addr = server_socket.accept()
        clients.append(conn)
        client_ids[conn] = i + 1  # Assign a client ID
        print(f"Client {client_ids[conn]} ({addr}) connected. ({len(clients)}/{num_clients})")

    print("All clients connected. Waiting for all clients to be ready...")
    
    # Wait for all clients to signal they're ready
    ready_clients = 0
    ready_lock = threading.Lock()
    
    def wait_for_ready(conn):
        nonlocal ready_clients
        try:
            buffer = b""
            while len(buffer) < 4:
                chunk = conn.recv(4 - len(buffer))
                if not chunk:  # Connection closed
                    print(f"Client {client_ids[conn]} disconnected while waiting for ready signal")
                    return
                buffer += chunk
                
            ready_signal = struct.unpack(">I", buffer)[0]
            if ready_signal == 1:
                with ready_lock:
                    ready_clients += 1
                    print(f"Client {client_ids[conn]} is ready. ({ready_clients}/{num_clients})")
        except Exception as e:
            print(f"Error waiting for ready signal from Client {client_ids[conn]}: {e}")
    
    # Create threads to wait for ready signals
    ready_threads = [threading.Thread(target=wait_for_ready, args=(client,)) for client in clients]
    for thread in ready_threads:
        thread.start()
    for thread in ready_threads:
        thread.join()
    
    if ready_clients == num_clients:
        print(f"All {ready_clients}/{num_clients} clients are ready. Starting training...")
        
        # Send start signal to all clients
        for client in clients:
            try:
                start_message = struct.pack(">I", 1)  # 1 means start
                client.sendall(start_message)
            except Exception as e:
                print(f"Error sending start signal to Client {client_ids[client]}: {e}")
    else:
        print(f"Only {ready_clients}/{num_clients} clients are ready. Cannot start training.")
        # Close connections and exit
        for client in clients:
            client.close()
        server_socket.close()
        return

    # Start timing
    start_time = time.time()

    # Send global iterations count
    for client in clients:
        client.sendall(struct.pack(">I", global_iterations))

    for iteration in range(global_iterations):
        print(f"\n--- Global Iteration {iteration + 1}/{global_iterations} ---")

        if iteration > 0:
            buffer = io.BytesIO()
            torch.save(global_state, buffer)
            model_data = buffer.getvalue()

            # Send to all clients simultaneously
            for client in clients:
                client.sendall(struct.pack(">I", len(model_data)))  
                client.sendall(model_data)  
                print(f"Sent aggregated weights to Client {client_ids[client]}")

        client_models = []
        client_accuracies = []
        client_received_times = {}  # Track when each client's model is received

        def handle_client(conn):
            try:
                print(f"Waiting for model update from Client {client_ids[conn]}...")

                buffer = b""
                while len(buffer) < 4:
                    buffer += conn.recv(4 - len(buffer))

                msg_size = struct.unpack(">I", buffer)[0]

                buffer = b""
                while len(buffer) < msg_size:
                    buffer += conn.recv(msg_size - len(buffer))

                # Record the time this client's update was received
                client_received_times[conn] = time.time()
                
                print(f"Received {len(buffer)} bytes from Client {client_ids[conn]}")
                buffer_io = io.BytesIO(buffer)
                client_data = torch.load(buffer_io)
                client_models.append(client_data["state_dict"])
                client_accuracies.append(client_data["accuracy"])

            except Exception as e:
                print(f"Error receiving data from Client {client_ids[conn]}: {e}")

        threads = [threading.Thread(target=handle_client, args=(client,)) for client in clients]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if len(client_models) == num_clients:
            # Record time when all models are received
            aggregation_start_time = time.time()
            print("Received all client weights. Beginning aggregation...")

            # Simulate some time for aggregation (if aggregation is very quick)
            # time.sleep(1.0)  # Uncomment if you want a minimum aggregation time

            global_state = {key: torch.stack([cm[key] for cm in client_models]).mean(dim=0) for key in client_models[0].keys()}
            global_model.load_state_dict(global_state)

            global_accuracy = sum(client_accuracies) / len(client_accuracies)
            global_accuracies.append(global_accuracy)  # Store accuracy for graph
            print(f"Global Model Accuracy after iteration {iteration + 1}: {global_accuracy:.2f}")

    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    # Save the final model
    torch.save(global_model.state_dict(), "final_global_model.pth")
    print("Training complete. Final global model saved.")

    # Plot Accuracy Graph
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, global_iterations + 1), global_accuracies, marker='o', markersize=4, linestyle='-', color='b')
    plt.xlabel("Global Iteration")
    plt.ylabel("Global Accuracy")
    plt.title("Global Model Accuracy over Global Iterations")
    plt.grid(True)
    
    # Show the plot before the program ends
    print("Displaying accuracy graph. Close the graph window to end the program.")
    plt.show()  # This ensures the graph is displayed before exiting

    # Close connections
    for client in clients:
        client.close()
    server_socket.close()

if __name__ == "__main__":
    server_program()