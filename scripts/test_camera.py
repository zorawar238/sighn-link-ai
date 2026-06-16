# ==============================================================================
# CAMERA TEST SCRIPT - Indian Sign Language Translator AI
# ==============================================================================
# Purpose: This script opens your webcam and displays the live video stream.
# We use this to verify OpenCV is working correctly and your camera is accessible.
#
# How to run:
#   python scripts/test_camera.py
#
# How to exit:
#   Press 'q' (or 'Q') on your keyboard while the window is active.
# ==============================================================================

import cv2
import sys
import os
import time

# We need to import our config file. To make sure Python can find it,
# we add the parent folder to the system path.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def test_camera():
    print("[INFO] Starting webcam test...")
    print(f"[INFO] Attempting to open camera at index: {config.CAMERA_INDEX}")
    
    # Initialize the camera capture object.
    # config.CAMERA_INDEX is usually 0 for the default webcam.
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    
    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        print("[ERROR] Suggestions to fix:")
        print("  1. Make sure no other application (like Zoom, Teams, or Discord) is using your webcam.")
        print("  2. If using an external webcam, try changing CAMERA_INDEX to 1 or 2 in 'config.py'.")
        print("  3. Make sure your system has given terminal/VS Code permission to access the webcam.")
        sys.exit(1)
        
    # Configure webcam frame dimensions using values from config.py
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    print("[INFO] Webcam opened successfully!")
    print("[INFO] Live feed running. Press 'q' key to quit the program.")
    
    # Variable to calculate and display Frames Per Second (FPS)
    prev_time = 0
    
    # Start the continuous loop to grab and display frames
    while True:
        # Read a frame from the webcam.
        # ret: A boolean (True/False) showing if the frame was successfully read.
        # frame: The image array containing the captured frame.
        ret, frame = cap.read()
        
        # If OpenCV failed to read a frame, break the loop
        if not ret:
            print("[ERROR] Failed to grab frame from webcam. Exiting...")
            break
            
        # Optional: Flip the frame horizontally (mirror effect) so it feels natural.
        # Sign language translation feels much more intuitive when you see yourself mirrored.
        frame = cv2.flip(frame, 1)
        
        # Calculate FPS (Frames Per Second)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # Add styled overlays onto the frame to make the app feel premium.
        # 1. Overlay a semi-transparent status bar at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (config.FRAME_WIDTH, 40), (43, 29, 14), -1) # Dark brown bar
        alpha = 0.6  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 2. Add text: App Title
        cv2.putText(frame, "ISL Translator AI - Camera Test", (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 3. Add text: FPS Indicator
        cv2.putText(frame, f"FPS: {int(fps)}", (config.FRAME_WIDTH - 100, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # 4. Add text: Instruction helper at the bottom left
        cv2.putText(frame, "Press 'Q' to Exit", (15, config.FRAME_HEIGHT - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display the processed frame in a graphical window
        cv2.imshow("ISL Translator - Camera Test", frame)
        
        # cv2.waitKey(1) waits for 1 millisecond for a keyboard event.
        # 0xFF extracts the last 8 bits of the key code to match character values.
        # ord('q') gets the ASCII value of the character 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting webcam test...")
            break
            
    # Clean up operations after leaving the loop:
    # 1. Release the webcam hardware so other apps can use it.
    cap.release()
    # 2. Close all OpenCV graphical windows.
    cv2.destroyAllWindows()
    print("[INFO] Webcam test completed successfully and resources released.")

if __name__ == "__main__":
    test_camera()
