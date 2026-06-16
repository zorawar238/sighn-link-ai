# ==============================================================================
# LANDMARK VISUALIZER SCRIPT - Indian Sign Language Translator AI
# ==============================================================================
# Purpose: This script opens the webcam, runs our HandTracker class, draws the
# skeletal joints (landmarks) in real time, and prints landmark coordinate data.
# We use this to verify MediaPipe is tracking hands successfully.
#
# How to run:
#   python scripts/test_landmarks.py
#
# How to exit:
#   Press 'q' (or 'Q') on your keyboard while the window is active.
# ==============================================================================

import cv2
import sys
import os
import time
import numpy as np

# Add the parent directory to the system path so Python can find the 'app' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from app.hand_tracker import HandTracker

def test_landmarks():
    print("[INFO] Starting MediaPipe hand landmark visualizer...")
    print(f"[INFO] Initializing webcam at index: {config.CAMERA_INDEX}")
    
    # Initialize the camera capture object
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)
        
    # Configure frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    # Initialize our HandTracker class
    print("[INFO] Initializing MediaPipe Hands model...")
    tracker = HandTracker()
    print("[INFO] Initialization complete! Track your hands in front of the camera.")
    print("[INFO] Press 'q' on the keyboard to exit.")
    
    # Frame rate calculation variables
    prev_time = 0
    
    # Loop to capture and process frames continuously
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam. Exiting...")
            break
            
        # Horizontally mirror the frame (selfie view) so tracking feels natural
        frame = cv2.flip(frame, 1)
        
        # 1. Detect hands and draw the visual skeleton overlay directly on the frame
        frame = tracker.find_hands(frame, draw=True)
        
        # 2. Extract landmark coordinates
        # landmarks is a flat NumPy array of shape (126,)
        landmarks = tracker.extract_landmarks()
        
        # Calculate how many hands are currently in view
        # We do this by checking if the coordinate values in each hand slot are non-zero.
        # Left hand is index 0:63, Right hand is index 63:126.
        left_hand_visible = not np.all(landmarks[0:63] == 0)
        right_hand_visible = not np.all(landmarks[63:126] == 0)
        hands_count = int(left_hand_visible) + int(right_hand_visible)
        
        # If hands are visible, let's print the coordinate array shape to the console
        if hands_count > 0:
            # We also print the first landmark of the tracked hand (Wrist coordinate x, y)
            # as a live debug log. Left hand wrist is landmark 0 (index 0, 1, 2 for X, Y, Z).
            wrist_x = landmarks[0] if left_hand_visible else landmarks[63]
            wrist_y = landmarks[1] if left_hand_visible else landmarks[64]
            print(f"[DEBUG] Hand(s) in view: {hands_count} | Landmark Shape: {landmarks.shape} | Wrist: ({wrist_x:.2f}, {wrist_y:.2f})")
            
        # Calculate FPS (Frames Per Second)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # Add premium graphical status overlays on the frame
        # 1. Overlay a semi-transparent dark status bar at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (config.FRAME_WIDTH, 40), (20, 20, 20), -1) # Sleek charcoal
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 2. Render App Title
        cv2.putText(frame, "ISL Translator - Hand Tracking Test", (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 3. Render Status: Hands detected
        hands_text = f"Hands Tracked: {hands_count}"
        color = (0, 255, 0) if hands_count > 0 else (0, 165, 255) # Green if active, orange if not
        cv2.putText(frame, hands_text, (config.FRAME_WIDTH - 280, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # 4. Render FPS counter
        cv2.putText(frame, f"FPS: {int(fps)}", (config.FRAME_WIDTH - 90, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # 5. Render active instructions
        cv2.putText(frame, "Press 'Q' to Exit", (15, config.FRAME_HEIGHT - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display the live processed video frame
        cv2.imshow("ISL Translator - MediaPipe Test", frame)
        
        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Stopping hand tracking test...")
            break
            
    # Clean up and release webcam
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("[INFO] Resources released successfully. Visualizer stopped.")

if __name__ == "__main__":
    test_landmarks()
