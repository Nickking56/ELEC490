import RPi.GPIO as GPIO
import time
import threading
import os
import signal

# Create a file to communicate between processes
DISPLAY_FILE = "display_status.txt"
RUNNING_FILE = "running_status.txt"

# Define segment pins (A-G, DP) using BOARD numbering
segments = [16, 29, 33, 36, 37, 18, 31]  # A, B, C, D, E, F, G
# Define digit control pins
digits = [15, 21, 22]  # Digit 1, Digit 2, Digit 3

def setup_display():
    """Initialize the GPIO pins for the seven-segment display"""
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(segments, GPIO.OUT, initial=GPIO.HIGH)  # Set all segments OFF
    GPIO.setup(digits, GPIO.OUT, initial=GPIO.HIGH)    # Turn off all digits

def turn_on_segment(DigitPos, Number):
    """Turn on a specific digit position with a specific number"""
    # Digit selection mapping
    digitsMap = {
        1: (1, 0, 0),  # First digit (hundreds)
        2: (0, 1, 0),  # Second digit (tens)
        3: (0, 0, 1)   # Third digit (ones)
    }
    
    # Select the correct digit
    for idx, digit in enumerate(digitsMap[DigitPos]):
        if digit == 1:
            GPIO.output(digits[idx], GPIO.HIGH)
        else:
            GPIO.output(digits[idx], GPIO.LOW)
    
    # Number to segment mapping
    nums = {
        -1: (0, 0, 0, 0, 0, 0, 0),
        0: (1, 1, 1, 1, 1, 1, 0),  # 0
        1: (0, 1, 1, 0, 0, 0, 0),  # 1
        2: (1, 1, 0, 1, 1, 0, 1),  # 2
        3: (1, 1, 1, 1, 0, 0, 1),  # 3
        4: (0, 1, 1, 0, 0, 1, 1),  # 4
        5: (1, 0, 1, 1, 0, 1, 1),  # 5
        6: (1, 0, 1, 1, 1, 1, 1),  # 6
        7: (1, 1, 1, 0, 0, 0, 0),  # 7
        8: (1, 1, 1, 1, 1, 1, 1),  # 8
        9: (1, 1, 1, 1, 0, 1, 1)   # 9
    }
    
    # Special segment patterns
    patterns = {
        # Line pattern for loading
        'line1': (1, 0, 0, 0, 0, 0, 0),  # Top
        'line2': (0, 1, 0, 0, 0, 0, 0),  # Top right
        'line3': (0, 0, 1, 0, 0, 0, 0),  # Bottom right
        'line4': (0, 0, 0, 1, 0, 0, 0),  # Bottom
        'line5': (0, 0, 0, 0, 1, 0, 0),  # Bottom left
        'line6': (0, 0, 0, 0, 0, 1, 0),  # Top left
        'dash':  (0, 0, 0, 0, 0, 0, 1)   # Middle dash
    }
    
    # If Number is a string, it might be a special pattern
    if isinstance(Number, str) and Number in patterns:
        pattern = patterns[Number]
    else:
        # Otherwise, get the standard number pattern
        pattern = nums[Number]
    
    # Turn on/off segments based on the pattern
    for idx, seg in enumerate(pattern):
        if seg == 1:
            GPIO.output(segments[idx], GPIO.LOW)
        else:
            GPIO.output(segments[idx], GPIO.HIGH)

def get_digit(number, n):
    """Get the nth digit of a number"""
    return number // 10**n % 10

def display_number(number):
    """Display a 3-digit number on the seven-segment display"""
    # Ensure the number is between 0-999
    number = max(0, min(999, number))
    
    # Extract individual digits
    digit1 = get_digit(number, 2)  # Hundreds
    digit2 = get_digit(number, 1)  # Tens
    digit3 = get_digit(number, 0)  # Ones
    
    # Display each digit
    turn_on_segment(1, digit1)
    time.sleep(0.005)
    turn_on_segment(2, digit2)
    time.sleep(0.005)
    turn_on_segment(3, digit3)
    time.sleep(0.005)

def display_loading_animation(frame):
    """Display a snake-like loading animation on the 7-segment display"""
    # Loading animation pattern - circles around the display
    patterns = ['line1', 'line2', 'line3', 'line4', 'line5', 'line6']
    
    # Calculate current position in animation
    pos1 = frame % 6
    pos2 = (frame + 2) % 6
    pos3 = (frame + 4) % 6
    
    # Display each segment of the animation on different digits
    turn_on_segment(1, patterns[pos1])
    time.sleep(0.005)
    turn_on_segment(2, patterns[pos2])
    time.sleep(0.005)
    turn_on_segment(3, patterns[pos3])
    time.sleep(0.005)

def display_controller():
    """Main loop for the display controller"""
    setup_display()
    
    # Initialize display status file
    with open(DISPLAY_FILE, 'w') as f:
        f.write("0")
    
    # Initialize running status file
    with open(RUNNING_FILE, 'w') as f:
        f.write("1")
    
    try:
        current_number = 0
        loading_frame = 0
        in_loading_mode = False
        
        while True:
            # Check if we should continue running
            try:
                with open(RUNNING_FILE, 'r') as f:
                    if f.read().strip() != "1":
                        break
            except:
                # If file not found, continue running
                pass
                
            # Read the current display mode
            try:
                with open(DISPLAY_FILE, 'r') as f:
                    status = f.read().strip()
                    
                    if status == "loading":
                        in_loading_mode = True
                    elif in_loading_mode and status != "loading":
                        # Exit loading mode if status changed
                        in_loading_mode = False
                        try:
                            current_number = int(status)
                        except:
                            current_number = 0
                    elif not in_loading_mode:
                        try:
                            new_number = int(status)
                            if new_number != current_number:
                                current_number = new_number
                        except:
                            # If not a number, ignore
                            pass
            except:
                # If file not found or error reading, continue with current mode
                pass
                
            # Display based on current mode
            if in_loading_mode:
                display_loading_animation(loading_frame)
                loading_frame = (loading_frame + 1) % 6  # Cycle through animation frames
            else:
                display_number(current_number)
            
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
        print("Display controller stopped")

if __name__ == "__main__":
    display_controller()