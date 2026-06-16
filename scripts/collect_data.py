# ==============================================================================
# DATASET RECORDER SCRIPT - Indian Sign Language Translator AI
# ==============================================================================
# Purpose: This script guides you through recording 3D hand landmark coordinate
# data to build your own custom sign language dataset.
# It records sequences of 30 frames (~1 sec of motion) for each target sign.
#
# How to run:
#   python scripts/collect_data.py
#
# Keyboard controls:
#   Press 'q' (or 'Q') to quit the program early.
# ==============================================================================

import cv2
import numpy as np
import os
import sys
import time

# Add the parent folder to the system path so Python can find the config & app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from app.hand_tracker import HandTracker

def setup_directories():
    """
    Creates the complete nested directory structure under 'collected_data/'.
    Directories: collected_data/<gesture_name>/<sequence_number>/
    """
    print("[INFO] Setting up dataset folder directories...")
    
    # Iterate through each word we want to recognize (e.g. 'hello', 'yes')
    for gesture in config.GESTURES:
        # For each word, we want to record 30 separate sequences
        for sequence in range(config.NO_SEQUENCES):
            # Generate the path: collected_data/<gesture>/<sequence>
            dir_path = os.path.join(config.DATA_PATH, gesture, str(sequence))
            
            # Create the folder and all its parent folders if they don't exist
            os.makedirs(dir_path, exist_ok=True)
            
    print(f"[INFO] Folders successfully set up under: {config.DATA_PATH}")

def collect_data():
    # 1. Create all directory folders first
    setup_directories()
    
    print("[INFO] Initializing webcam capture...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Ensure no other app is using it.")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    # Initialize the HandTracker class
    tracker = HandTracker()
    
    print("\n" + "="*80)
    print("                      DATASET RECORDING PROTOCOL")
    print("="*80)
    print("You will record 30 sequences of 30 frames (~1 second of motion) for each sign.")
    print("A 2-second countdown will occur on the screen before each sequence begins.")
    print("Please perform the gesture dynamically during the active recording phase.")
    print("Press 'q' at any time to exit the recording early.")
    print("="*80 + "\n")
    
    time.sleep(2) # Give the user a moment to read the console
    
    # 2. Iterate through each target sign in config.GESTURES
    for gesture in config.GESTURES:
        print(f"\n[RECORDING SIGN]: '{gesture.upper()}'")
        
        # 3. Iterate through 30 sequences for this sign
        for sequence in range(config.NO_SEQUENCES):
            
            # --- PHASE A: GET READY COUNTDOWN ---
            # We want a 2-second countdown on-screen so the user can position their hand.
            countdown_start = time.time()
            countdown_duration = 2.0  # seconds
            
            while time.time() - countdown_start < countdown_duration:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                
                # Still draw landmarks during countdown so the user can verify positioning
                frame = tracker.find_hands(frame, draw=True)
                
                # Calculate remaining seconds for countdown (2... 1...)
                time_elapsed = time.time() - countdown_start
                time_remaining = max(0.0, countdown_duration - time_elapsed)
                
                # Draw standard translucent overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (config.FRAME_WIDTH, config.FRAME_HEIGHT), (0, 0, 0), -1)
                alpha = 0.45  # Dim the screen slightly
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                # Add text: SIGN TITLE
                cv2.putText(frame, f"GET READY TO SIGN: '{gesture.upper()}'", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Add text: SEQUENCE PROGRESS
                cv2.putText(frame, f"Sequence: {sequence + 1}/{config.NO_SEQUENCES}", (50, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                
                # Add text: COUNTDOWN NUMBER
                cv2.putText(frame, f"Starting in: {int(time_remaining) + 1}...", (50, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3, cv2.LINE_AA)
                
                cv2.imshow("ISL Dataset Collector", frame)
                
                # Allow user to quit during countdown by pressing 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Early exit triggered by user during countdown.")
                    cap.release()
                    tracker.close()
                    cv2.destroyAllWindows()
                    sys.exit(0)
            
            # --- PHASE B: ACTIVE DATA RECORDING ---
            print(f"  -> Recording sequence {sequence}...")
            
            # Record 30 consecutive frames
            for frame_num in range(config.SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                
                # Run hand tracking
                frame = tracker.find_hands(frame, draw=True)
                
                # Extract landmarks coordinate array shape (126,)
                landmarks = tracker.extract_landmarks()
                
                # Generate save path: collected_data/<gesture>/<sequence>/<frame_num>.npy
                npy_path = os.path.join(config.DATA_PATH, gesture, str(sequence), f"{frame_num}.npy")
                
                # Save landmarks as raw NumPy binary files
                np.save(npy_path, landmarks)
                
                # Overlay dynamic "ACTIVE RECORDING" HUD
                # Draw high-visibility red bar at the top representing active recording
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (config.FRAME_WIDTH, 40), (0, 0, 200), -1) # Red status bar
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                cv2.putText(frame, f"RECORDING SIGN: '{gesture.upper()}'", (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                cv2.putText(frame, f"Seq: {sequence + 1}/{config.NO_SEQUENCES} | Frame: {frame_num + 1}/{config.SEQUENCE_LENGTH}",
                            (config.FRAME_WIDTH - 250, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                cv2.putText(frame, "Press 'Q' to Quit", (15, config.FRAME_HEIGHT - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Display active capture frame
                cv2.imshow("ISL Dataset Collector", frame)
                
                # Ensure a small delay for OpenCV window processing
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Early exit triggered by user during active collection.")
                    cap.release()
                    tracker.close()
                    cv2.destroyAllWindows()
                    sys.exit(0)
                    
            print(f"  -> Completed sequence {sequence}!")
            
    # Clean up operations when entire collection is completed
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("\n" + "="*80)
    print("                    DATASET RECORDING SUCCESSFULLY COMPLETED!")
    print("="*80)
    print(f"All landmark coordinate sequences have been saved successfully under: {config.DATA_PATH}")
    print("You can verify files exist inside your directory.")
    print("="*80 + "\n")

if __name__ == "__main__":
    collect_data()
