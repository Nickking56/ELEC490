from gpiozero import PWMLED
import time
import os
import signal

# Create a file to communicate between processes
LED_FILE = "led_status.txt"
RUNNING_FILE = "running_status.txt"

# Define LEDs using PWM for brightness control
leds = [
    PWMLED(3),   # LED 1
    PWMLED(14),  # LED 2
    PWMLED(4),   # LED 3
    PWMLED(15),  # LED 4
    PWMLED(17),  # LED 5
    PWMLED(27)   # LED 6 (halfway point)
]

def communication_cycle():
    """
    Visualize communication with direction-dependent trailing glow
    and pausing at the halfway point until trailing LEDs fade away
    """
    # Turn off all LEDs first
    for led in leds:
        led.value = 0
    
    # Define the halfway point (index 5 - the 6th LED)
    halfway_point = 5
    
    # ---- PHASE 1: Client to Server (Left to Right) ----
    # Move from left to right until reaching the halfway point
    for i in range(halfway_point + 1):
        # Reset all LEDs
        for led in leds:
            led.value = 0
            
        # Main LED at full brightness
        leds[i].value = 1.0
        
        # Trailing LED one behind at medium brightness (if valid position)
        if i-1 >= 0:
            leds[i-1].value = 0.4
        
        # Trailing LED two behind at faint brightness (if valid position)
        if i-2 >= 0:
            leds[i-2].value = 0.1
            
        time.sleep(0.01)  # Fast animation
    
    # ---- PHASE 2: Pause at Halfway Point (Server Aggregation) ----
    # Add significant pause at server to simulate aggregation
    time.sleep(0.5)  # Pause for aggregation
    
    # Keep the main LED lit at the halfway point
    # and let the trailing LEDs gradually fade out
    
    # First trailing LED still visible
    for led in leds:
        led.value = 0
    leds[halfway_point].value = 1.0
    leds[halfway_point-1].value = 0.4
    leds[halfway_point-2].value = 0.1
    time.sleep(0.01)  # Fast animation
    
    # Only the closest trailing LED visible
    for led in leds:
        led.value = 0
    leds[halfway_point].value = 1.0
    leds[halfway_point-1].value = 0.4
    time.sleep(0.01)  # Fast animation
    
    # Only the main LED at the halfway point
    for led in leds:
        led.value = 0
    leds[halfway_point].value = 1.0
    time.sleep(0.01)  # Fast animation
    
    # ---- PHASE 3: Server to Client (Right to Left) ----
    # Move from halfway point back to the left
    for i in range(halfway_point, -1, -1):
        # Reset all LEDs
        for led in leds:
            led.value = 0
            
        # Main LED at full brightness
        leds[i].value = 1.0
        
        # Trailing LED (now on the right) at medium brightness
        if i+1 < len(leds):
            leds[i+1].value = 0.4
        
        # Second trailing LED at faint brightness
        if i+2 < len(leds):
            leds[i+2].value = 0.1
            
        time.sleep(0.01)  # Fast animation
    
    # ---- PHASE 4: End Cycle ----
    # Final cleanup - turn all LEDs off
    time.sleep(0.05)  # Small pause at the end
    for led in leds:
        led.value = 0

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
        led.value = 0
    # Turn on first and last LED
    leds[0].value = 1.0
    leds[-1].value = 1.0
    
def reset_leds():
    """Reset all LEDs to off"""
    for led in leds:
        led.value = 0

def led_controller():
    """Main loop for the LED controller"""
    # Initialize LED status file
    with open(LED_FILE, 'w') as f:
        f.write("none")
    
    # Initialize running status file if it doesn't exist
    if not os.path.exists(RUNNING_FILE):
        with open(RUNNING_FILE, 'w') as f:
            f.write("1")  # Default to running
    
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
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        pass
    finally:
        # Turn off all LEDs
        for led in leds:
            led.value = 0
        print("LED controller stopped")

if __name__ == "__main__":
    led_controller()