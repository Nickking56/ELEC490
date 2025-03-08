from gpiozero import LED
import time
import os
import signal

# Create a file to communicate between processes
LED_FILE = "led_status.txt"
RUNNING_FILE = "running_status.txt"

# Define LEDs
leds = [
    LED(3),   # LED 1
    LED(14),  # LED 2
    LED(4),   # LED 3
    LED(15),  # LED 4
    LED(17),  # LED 5
    LED(27)   # LED 6
]

def client_to_server():
    """LED sequence to visualize data transfer from client to server"""
    for led in leds:
        led.on()
        time.sleep(0.1)
        led.off()

def server_to_client():
    """LED sequence to visualize data transfer from server to client"""
    for led in reversed(leds):
        led.on()
        time.sleep(0.1)
        led.off()

def led_controller():
    """Main loop for the LED controller"""
    # Initialize LED status file
    with open(LED_FILE, 'w') as f:
        f.write("none")
    
    try:
        while True:
            # Check if we should continue running
            try:
                with open(RUNNING_FILE, 'r') as f:
                    if f.read().strip() != "1":
                        break
            except:
                # If file not found, continue running
                pass
                
            # Read the LED status
            try:
                with open(LED_FILE, 'r') as f:
                    status = f.read().strip()
                    
                    if status == "client_to_server":
                        client_to_server()
                        # Reset status after performing action
                        with open(LED_FILE, 'w') as f:
                            f.write("none")
                            
                    elif status == "server_to_client":
                        server_to_client()
                        # Reset status after performing action
                        with open(LED_FILE, 'w') as f:
                            f.write("none")
            except:
                # If file not found or error reading, continue
                pass
                
            # Small delay to prevent high CPU usage
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        pass
    finally:
        # Turn off all LEDs
        for led in leds:
            led.off()
        print("LED controller stopped")

if __name__ == "__main__":
    led_controller()