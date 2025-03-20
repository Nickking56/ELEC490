from gpiozero import PWMLED
import time
import os
import signal

# Create a file to communicate between processes
LED_FILE = "led_status.txt"
RUNNING_FILE = "running_status.txt"

# Define LEDs using PWM for brightness control
leds = [
    PWMLED(3),   # LED 1 (node)
    PWMLED(14),  # LED 2
    PWMLED(4),   # LED 3
    PWMLED(15),  # LED 4
    PWMLED(17),  # LED 5
    PWMLED(27)   # LED 6 (server)
]

def flash_all_leds():
    """Flash all LEDs once"""
    # Turn off all LEDs first
    for led in leds:
        led.value = 0
    
    # Flash all LEDs on
    for led in leds:
        led.value = 1.0
    time.sleep(0.3)
    
    # Turn all LEDs off
    for led in leds:
        led.value = 0
    time.sleep(0.1)

def snake_loading_animation(cycles=5):
    """
    Create a snake-like loading animation that circles through the LEDs
    cycles: number of full cycles to perform
    """
    # Define snake length
    snake_length = 3
    
    # Turn off all LEDs first
    for led in leds:
        led.value = 0
    
    # Create the snake animation
    for _ in range(cycles):
        # Forward movement (LED 1 to LED 6)
        for pos in range(len(leds) + snake_length - 1):
            # Turn off all LEDs
            for led in leds:
                led.value = 0
            
            # Draw the snake (brightest LED at the head, fading to tail)
            for i in range(snake_length):
                snake_pos = pos - i
                if 0 <= snake_pos < len(leds):
                    brightness = 1.0 - (i * 0.3)  # Fade brightness for tail
                    leds[snake_pos].value = max(0.1, brightness)
            
            time.sleep(0.05)  # Speed of snake movement
        
        # Backward movement (LED 6 to LED 1)
        for pos in range(len(leds) + snake_length - 1, -1, -1):
            # Turn off all LEDs
            for led in leds:
                led.value = 0
            
            # Draw the snake (brightest LED at the head, fading to tail)
            for i in range(snake_length):
                snake_pos = pos - i
                if 0 <= snake_pos < len(leds):
                    brightness = 1.0 - (i * 0.3)  # Fade brightness for tail
                    leds[snake_pos].value = max(0.1, brightness)
            
            time.sleep(0.05)  # Speed of snake movement

def communication_cycle(server_wait_time=0.5):
    """
    Visualize communication with direction-dependent trailing glow
    and pausing at the halfway point until trailing LEDs fade away
    
    server_wait_time: How long to pause at the server LED (minimum 0.5s)
    """
    # Turn off all LEDs first
    for led in leds:
        led.value = 0
    
    # Define the node and server positions
    node_led = 0       # LED 1
    server_led = 5     # LED 6
    
    # ---- PHASE 1: Client to Server (Left to Right) ----
    # Move from left to right until reaching the server
    for i in range(server_led + 1):
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
    
    # ---- PHASE 2: Pause at Server (Aggregation) ----
    # Keep the main LED lit at the server
    # and let the trailing LEDs gradually fade out
    
    # First trailing LED still visible
    for led in leds:
        led.value = 0
    leds[server_led].value = 1.0
    leds[server_led-1].value = 0.4
    leds[server_led-2].value = 0.1
    time.sleep(0.01)  # Fast animation
    
    # Only the closest trailing LED visible
    for led in leds:
        led.value = 0
    leds[server_led].value = 1.0
    leds[server_led-1].value = 0.4
    time.sleep(0.01)  # Fast animation
    
    # Only the server LED at full brightness
    for led in leds:
        led.value = 0
    leds[server_led].value = 1.0
    
    # Pause at server for aggregation - use provided wait time (min 0.5s)
    time.sleep(max(0.5, server_wait_time))
    
    # ---- PHASE 3: Server to Client (Right to Left) ----
    # Move from server back to the node
    for i in range(server_led, -1, -1):
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

def node_training():
    """Turn on node LED during local training"""
    # Turn off all LEDs
    for led in leds:
        led.value = 0
    
    # Turn on the node LED (LED 1)
    leds[0].value = 1.0

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
                    
                    if status.startswith("communicate:"):
                        # Extract server wait time if specified
                        parts = status.split(":")
                        if len(parts) > 1:
                            try:
                                wait_time = float(parts[1])
                                communication_cycle(wait_time)
                            except ValueError:
                                # If conversion fails, use default
                                communication_cycle()
                        else:
                            communication_cycle()
                        # Reset status after performing action
                        with open(LED_FILE, 'w') as f:
                            f.write("none")
                    
                    elif status in ["client_to_server", "server_to_client", "communicate"]:
                        communication_cycle()
                        # Reset status after performing action
                        with open(LED_FILE, 'w') as f:
                            f.write("none")
                    
                    elif status == "node_training":
                        node_training()
                        # Don't reset status for node training - keeps LED on
                    
                    elif status == "loading":
                        snake_loading_animation()
                        # After running the animation, switch to continuous mode
                        with open(LED_FILE, 'w') as f:
                            f.write("continuous_loading")
                    
                    elif status == "continuous_loading":
                        # Run a single cycle of the snake animation and keep the status
                        snake_loading_animation(cycles=1)
                    
                    elif status == "flash":
                        flash_all_leds()
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