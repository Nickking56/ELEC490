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
    
    # Turn on/off segments based on the number
    for idx, num in enumerate(nums[Number]):
        if num == 1:
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
        while True:
            # Check if we should continue running
            try:
                with open(RUNNING_FILE, 'r') as f:
                    if f.read().strip() != "1":
                        break
            except:
                # If file not found, continue running
                pass
                
            # Read the current number to display
            try:
                with open(DISPLAY_FILE, 'r') as f:
                    new_number = int(f.read().strip())
                    if new_number != current_number:
                        current_number = new_number
            except:
                # If file not found or error reading, continue with current number
                pass
                
            # Display the number
            display_number(current_number)
            
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
        print("Display controller stopped")

if __name__ == "__main__":
    display_controller()