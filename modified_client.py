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

# Function to start loading animation on display
def start_loading_animation():
    """Start loading animation on display"""
    with open(DISPLAY_FILE, 'w') as f:
        f.write("loading")
    
    # Start LED loading animation
    with open(LED_FILE, 'w') as f:
        f.write("loading")
    
    # Wait for animation to initialize
    time.sleep(0.2)

# Function to flash all LEDs once
def flash_all_leds():
    """Flash all LEDs once to indicate connection status"""
    with open(LED_FILE, 'w') as f:
        f.write("flash")
    time.sleep(0.6)  # Wait for flash to complete

# Function to trigger LED sequences
def trigger_communication(server_wait_time=0.5):
    """Trigger the LED communication cycle with a variable server wait time"""
    with open(LED_FILE, 'w') as f:
        f.write(f"communicate:{server_wait_time}")
    
    # Wait for LED animation to complete - dynamic wait time based on:
    # - Time for LED chain to reach server (~0.1s)
    # - Server wait time (variable)
    # - Time for LED chain to return (~0.1s)
    # - Small buffer (0.1s)
    wait_time = 0.1 + server_wait_time + 0.1 + 0.1
    time.sleep(wait_time)  # Adjusted wait time for the animation

# Function to set node LED on during training
def set_node_training():
    """Turn on the node LED to indicate local training"""
    with open(LED_FILE, 'w') as f:
        f.write("node_training")

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
    print(f"Client {client_id} starting up...")
    
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
    
    # Start loading animation
    start_loading_animation()
    
    # Register function to stop controllers on exit
    atexit.register(lambda: stop_controllers(display_process, led_process))
    
    print(f"Client {client_id} attempting to connect to server at {host}:{port}...")
    
    # Connect to server
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((host, port))
    except Exception as e:
        print(f"Connection error: {e}")
        # Stop the loading animation and show error (000)
        reset_displays()
        sys.exit(1)
    
    print(f"Client {client_id} connected to server at {host}:{port}")
    
    # Flash all LEDs to indicate successful connection
    flash_all_leds()
    
    # Reset displays after connection
    reset_displays()
    
    # Receive number of global iterations
    buffer = b""
    while len(buffer) < 4:
        buffer += client_socket.recv(4 - len(buffer))
    global_iterations = struct.unpack(">I", buffer)[0]
    
    # Setup model
    input_size = X_client.shape[1]
    num_classes = len(y_client["Sleep Disorder"].unique())
    model = SleepModel(input_size, num_classes)
    
    # Initialize timing variables
    send_time = 0
    
    # Main training loop
    for iteration in range(global_iterations):
        # Update display with current iteration + 1
        update_display(iteration + 1)
        
        print(f"\nClient {client_id} - Starting iteration {iteration + 1}/{global_iterations}...")
        
        # Receive global model from server (except for first iteration)
        if iteration > 0:
            print(f"Receiving global model from server...")
            
            # Record time when we receive the global model
            receive_time = time.time()
            
            # Calculate how long we were waiting (since sending our update)
            wait_time = receive_time - send_time
            print(f"Client {client_id} waited {wait_time:.2f} seconds for aggregated weights.")
            
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
        
        # Turn on node LED during training
        set_node_training()
        
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
        
        # Record time before sending weights
        send_time = time.time()
        
        # Send message size
        client_socket.sendall(struct.pack(">I", len(message)))
        # Send message
        client_socket.sendall(message)
        
        # Visualize communication cycle once per iteration
        print(f"Communicating with server...")
        
        # Use a minimum of 0.5s for server pause, but it may be longer in reality
        # depending on how long it takes for the server to aggregate weights
        trigger_communication(0.5)
        
        print(f"Iteration {iteration + 1} complete.")
    
    # Training complete - flash all LEDs to indicate completion
    flash_all_leds()
    
    # Close connection
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