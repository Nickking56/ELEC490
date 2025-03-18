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
import time
import subprocess
import signal
import atexit

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

# Files for inter-process communication
DISPLAY_FILE = "display_status.txt"
LED_FILE = "led_status.txt"
RUNNING_FILE = "running_status.txt"

# Function to start display and LED controller processes
def start_controllers():
    # Start the display controller
    display_process = subprocess.Popen(["python3", "display_controller.py"])
    
    # Start the LED controller
    led_process = subprocess.Popen(["python3", "led_controller.py"])
    
    return display_process, led_process

# Function to update the display
def update_display(number):
    with open(DISPLAY_FILE, 'w') as f:
        f.write(str(number))

# Function to trigger LED sequences
def trigger_communication():
    """Trigger the LED communication cycle"""
    with open(LED_FILE, 'w') as f:
        f.write("communicate")
    
    # Wait for LED animation to complete (approx 1.2 seconds)
    # This ensures the animation finishes before continuing
    time.sleep(1.2)

# Function to set idle mode
def set_idle_mode():
    # Trigger idle mode for LEDs
    with open(LED_FILE, 'w') as f:
        f.write("idle")
    
    # Display stays showing the last iteration number
    # (No need to update DISPLAY_FILE)

# Function to reset the displays
def reset_displays():
    # Reset the iteration display to 000
    with open(DISPLAY_FILE, 'w') as f:
        f.write("0")
    
    # Reset the LEDs
    with open(LED_FILE, 'w') as f:
        f.write("reset")
    
    # Wait for controllers to process the commands
    time.sleep(0.2)

# Function to stop controllers on exit
def stop_controllers(display_process, led_process):
    # Signal to stop running
    with open(RUNNING_FILE, 'w') as f:
        f.write("0")
    
    # Wait a bit for processes to read the file
    time.sleep(0.5)
    
    # Terminate processes if they're still running
    if display_process.poll() is None:
        display_process.terminate()
    
    if led_process.poll() is None:
        led_process.terminate()

def client_program(client_id, data_dir, host="6.tcp.ngrok.io", port=17926):
    print(f"Client {client_id} attempting to connect to server at {host}:{port}...")
    
    # Load client data
    X_client = pd.read_csv(os.path.join(data_dir, f"client_{client_id}", "X_client.csv"))
    y_client = pd.read_csv(os.path.join(data_dir, f"client_{client_id}", "y_client.csv"))
    
    X_tensor = torch.tensor(X_client.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_client.values.flatten(), dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Start display and LED controllers
    display_process, led_process = start_controllers()
    
    # Wait a moment for controllers to start
    time.sleep(0.5)
    
    # Reset displays to initial state
    reset_displays()
    
    # Register function to stop controllers on exit
    atexit.register(lambda: stop_controllers(display_process, led_process))
    
    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print(f"Client {client_id} connected to server at {host}:{port}")
    
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
        # Update display with current iteration
        update_display(iteration)
        
        print(f"\nClient {client_id} - Starting local training (Iteration {iteration + 1}/{global_iterations})...")
        
        # Receive global model from server (except for first iteration)
        if iteration > 0:
            # Visualize server-to-client communication
            print(f"Receiving global model from server...")
            trigger_communication()
            
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
            print(f"Client {client_id} received aggregated weights. Beginning next round of training.")
        
        # Train local model
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
        
        # Visualize client-to-server communication
        print(f"Sending updated weights to server...")
        trigger_communication()
        
        # Add a pause to simulate server aggregation time
        print(f"Server aggregating weights... (visualized by LED 6 pause)")
        time.sleep(0.5)  # Aggregation pause
        
        print(f"Client {client_id} sent updated weights to server.")
    
    # Training complete
    client_socket.close()
    print(f"Client {client_id} training complete.")
    
    # Set to idle mode instead of stopping controllers
    set_idle_mode()
    print("System set to idle mode. Display shows final iteration count with first and last LEDs on.")
    print("Run the client again to reset and start a new training session.")

if __name__ == "__main__":
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    data_dir = "client_data"
    client_program(client_id, data_dir)
    while True:
        set_idle_mode()