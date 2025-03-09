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

def communication_cycle():
    """Visualize complete communication cycle between client and server"""
    # Turn off all LEDs first
    for led in leds:
        led.off()
    
    # Client sending to server (left to right)
    for led in leds:
        led.on()
        time.sleep(0.02)
    
    # Brief pause at server end
    time.sleep(0.1)
    
    # Server sending back to client (right to left)
    for i in range(len(leds)-1, -1, -1):
        # Turn off current LED
        leds[i].off()
        # Turn on previous LED (if not at the beginning)
        if i > 0:
            leds[i-1].on()
        time.sleep(0.02)
    
    # Final LED turns off after a short delay
    time.sleep(0.05)
    leds[0].off()

# Keep these for backward compatibility but have them call the new function
def client_to_server():
    """Legacy function that now uses the complete cycle"""
    communication_cycle()

def server_to_client():
    """Legacy function that now uses the complete cycle"""
    communication_cycle()
        
def idle_mode():
    """Set idle mode - first and last LED on"""
    # Turn off all LEDs first
    for led in leds:
        led.off()
    # Turn on first and last LED
    leds[0].on()
    leds[-1].on()
    
def reset_leds():
    """Reset all LEDs to off"""
    for led in leds:
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
                    
                    if status == "client_to_server" or status == "server_to_client" or status == "communicate":
                        communication_cycle()
                        # Reset status after performing action
                        with open(LED_FILE, 'w') as f:
                            f.write("none")
                            
                    elif status == "idle":
                        idle_mode()
                        # Don't reset status for idle mode
                        
                    elif status == "reset":
                        reset_leds()
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