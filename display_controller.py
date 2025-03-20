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

def turn_on_custom_segments(DigitPos, segment_pattern):
    """Turn on specific segments based on a pattern"""
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
    
    # Turn on/off segments based on the pattern
    for idx, segment_state in enumerate(segment_pattern):
        if segment_state == 1:
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

def wait_display():
    """Display 'UAIt' for waiting"""
    # Custom segment patterns for "UAIt"
    patterns = [
        # 'u' pattern for first digit
        (0, 0, 1, 1, 1, 0, 0),
        # 'A' pattern for second digit
        (1, 1, 1, 0, 1, 1, 1),
        # 'I' pattern for third digit (with 't' in lowercase)
        (0, 1, 1, 0, 0, 0, 0),
    ]
    
    # Display each custom digit
    turn_on_custom_segments(1, patterns[0])
    time.sleep(0.005)
    turn_on_custom_segments(2, patterns[1])
    time.sleep(0.005)
    turn_on_custom_segments(3, patterns[2])
    time.sleep(0.005)

def snake_animation_frame(frame_num):
    """Generate a snake animation frame"""
    # Define the snake animation sequence
    # Each tuple has patterns for each digit (1, 2, 3)
    snake_frames = [
        # Frame 0: Top segment of all digits
        [(1,0,0,0,0,0,0), (1,0,0,0,0,0,0), (1,0,0,0,0,0,0)],
        # Frame 1: Top-right segment
        [(0,1,0,0,0,0,0), (0,1,0,0,0,0,0), (0,1,0,0,0,0,0)],
        # Frame 2: Bottom-right segment
        [(0,0,1,0,0,0,0), (0,0,1,0,0,0,0), (0,0,1,0,0,0,0)],
        # Frame 3: Bottom segment
        [(0,0,0,1,0,0,0), (0,0,0,1,0,0,0), (0,0,0,1,0,0,0)],
        # Frame 4: Bottom-left segment
        [(0,0,0,0,1,0,0), (0,0,0,0,1,0,0), (0,0,0,0,1,0,0)],
        # Frame 5: Top-left segment
        [(0,0,0,0,0,1,0), (0,0,0,0,0,1,0), (0,0,0,0,0,1,0)],
        # Frame 6: Middle segment
        [(0,0,0,0,0,0,1), (0,0,0,0,0,0,1), (0,0,0,0,0,0,1)],
        # Additional frames for smoother animation
        [(1,1,0,0,0,0,0), (0,0,0,0,0,1,1), (0,0,0,1,1,0,0)],
        [(0,1,1,0,0,0,0), (1,0,0,0,0,0,1), (0,0,1,1,0,0,0)],
        [(0,0,1,1,0,0,0), (1,1,0,0,0,0,0), (0,1,1,0,0,0,0)],
        [(0,0,0,1,1,0,0), (0,1,1,0,0,0,0), (1,1,0,0,0,0,0)],
        [(0,0,0,0,1,1,0), (0,0,1,1,0,0,0), (1,0,0,0,0,0,1)]
    ]
    
    frame_idx = frame_num % len(snake_frames)
    return snake_frames[frame_idx]

def display_snake_animation_frame(frame_num):
    """Display a frame of the snake animation"""
    patterns = snake_animation_frame(frame_num)
    
    # Display each digit with its pattern