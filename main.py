import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import os
from datetime import datetime

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variables for smoothing cursor movement
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0
smoothing_factor = 0.5

# Click thresholds
click_threshold = 0.05  # Distance between finger tip and MCP joint
click_cooldown = 0.3  # seconds
last_click_time = 0

# Screenshot path
screenshot_dir = "virtual_mouse_screenshots"
os.makedirs(screenshot_dir, exist_ok=True)

# Fist detection variables
fist_threshold = 0.1  # Maximum average distance for fist detection
screenshot_cooldown = 1.0  # seconds
last_screenshot_time = 0

# Thumb detection variables
thumb_open_threshold = 0.3  # Distance threshold for considering thumb open
thumb_history = []
history_length = 5

# Create a resizable window
cv2.namedWindow('Virtual Mouse', cv2.WINDOW_NORMAL)

def is_fist(landmarks):
    # Calculate average distance from palm center to finger tips
    palm_center = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    distances = []
    for finger in [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]:
        tip = landmarks[finger]
        distance = math.hypot(tip.x - palm_center.x, tip.y - palm_center.y)
        distances.append(distance)
    return np.mean(distances) < fist_threshold

def is_thumb_open(landmarks):
    """Returns True if thumb is extended away from hand (cursor should pause)"""
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    
    # Calculate distance between thumb tip and wrist
    distance = math.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y)
    
    # Add to history for more stable detection
    thumb_history.append(distance)
    if len(thumb_history) > history_length:
        thumb_history.pop(0)
    
    # Use average of recent measurements
    avg_distance = np.mean(thumb_history)
    
    return avg_distance > thumb_open_threshold

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Flip the frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize default values
        cursor_active = False
        thumb_open = True  # Default to paused state when no hand is detected

        # Add elegant header with your name
        header_height = 60
        cv2.rectangle(frame, (0, 0), (frame.shape[1], header_height), (50, 50, 50), -1)
        cv2.putText(frame, "Gesture Driven Virtual Mouse Control System", (20, 30), 
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (255, 215, 0), 2)
        cv2.putText(frame, "Prepared by: Syed Ahmed Ali", (frame.shape[1]-400, 30), 
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Process the frame with mediapipe hands
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get landmark positions
                landmarks = hand_landmarks.landmark
                
                # Check thumb position (open = cursor inactive)
                thumb_open = is_thumb_open(landmarks)
                
                # Check for fist (screenshot)
                current_time = time.time()
                if is_fist(landmarks) and (current_time - last_screenshot_time) > screenshot_cooldown:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")
                    pyautogui.screenshot(screenshot_path)
                    last_screenshot_time = current_time
                    cv2.putText(frame, "Screenshot Taken!", (10, header_height + 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Only proceed if thumb is closed (not open)
                if not thumb_open:
                    cursor_active = True
                    # Get finger positions
                    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                    
                    # Calculate cursor position based on index finger
                    cursor_x = int(index_tip.x * screen_width)
                    cursor_y = int(index_tip.y * screen_height)
                    
                    # Smooth the cursor movement
                    curr_x = prev_x + smoothing_factor * (cursor_x - prev_x)
                    curr_y = prev_y + smoothing_factor * (cursor_y - prev_y)
                    
                    # Move the mouse
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y
                    
                    # Calculate distances for click detection
                    index_dist = math.hypot(index_tip.x - index_mcp.x, index_tip.y - index_mcp.y)
                    middle_dist = math.hypot(middle_tip.x - middle_mcp.x, middle_tip.y - middle_mcp.y)
                    
                    # Left click (index finger down)
                    if index_dist < click_threshold and (current_time - last_click_time) > click_cooldown:
                        pyautogui.click()
                        last_click_time = current_time
                        cv2.putText(frame, "Left Click", (10, header_height + 80), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Right click (middle finger down)
                    if middle_dist < click_threshold and (current_time - last_click_time) > click_cooldown:
                        pyautogui.rightClick()
                        last_click_time = current_time
                        cv2.putText(frame, "Right Click", (10, header_height + 120), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display cursor status
        status_text = "Cursor: ACTIVE" if cursor_active else "Cursor: PAUSED"
        pause_reason = "(no hand detected)" if not results.multi_hand_landmarks else "(thumb extended)" if thumb_open else ""
        cv2.putText(frame, f"{status_text} {pause_reason}", (10, header_height + 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Visual indicator for thumb state if hand is detected
        if results.multi_hand_landmarks:
            thumb_state = "THUMB: OPEN" if thumb_open else "THUMB: CLOSED"
            cv2.putText(frame, thumb_state, (10, header_height + 190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if thumb_open else (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Virtual Mouse', frame)
        
        # Exit on 'q' key press (works even when window not focused)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()