# ==============================================================================
# REAL-TIME PREDICTION VISUALIZER - Indian Sign Language Translator AI
# ==============================================================================
# Purpose: This script is the ultimate real-time visualizer for our ISL translator!
# It loads the trained LSTM model, reads your live webcam stream, processes hand
# landmarks, buffers coordinates, and prints dynamic smoothed classifications
# on the video frame with a custom glowing HUD.
#
# How to run:
#   python scripts/test_predictions.py
#
# Keyboard controls:
#   Press 'q' (or 'Q') to quit the program.
# ==============================================================================

import cv2
import numpy as np
import os
import sys
import time

# Disable annoying TensorFlow startup warning logs (keeps console clean)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Add the parent folder to the system path so Python can find the config & app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from app.hand_tracker import HandTracker

def run_realtime_prediction():
    print("[INFO] Starting Real-time ISL Gesture Predictor...")
    
    # 1. Load the trained LSTM model
    MODEL_FILE_PATH = os.path.join(config.MODEL_PATH, "isl_lstm_model.keras")
    print(f"[INFO] Loading trained Keras model from: {MODEL_FILE_PATH}...")
    
    if not os.path.exists(MODEL_FILE_PATH):
        print(f"[ERROR] Could not find the model file at: {MODEL_FILE_PATH}")
        print("[ERROR] Please make sure you completed Phase 5 by running 'python scripts/train_model.py' first!")
        sys.exit(1)
        
    model = tf.keras.models.load_model(MODEL_FILE_PATH)
    print("[INFO] Deep learning model loaded successfully!")
    
    # 2. Initialize webcam capture
    print(f"[INFO] Opening webcam at index: {config.CAMERA_INDEX}")
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Make sure no other application is using it.")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    # Initialize the HandTracker class
    tracker = HandTracker()
    
    # 3. Buffer Variables
    # rolling_sequence: Accumulates coordinates for the last 30 frames
    rolling_sequence = []
    
    # predictions_history: Remembers the last 10 prediction indices for smoothing
    predictions_history = []
    
    # The final smoothed gesture and confidence score to render on screen
    display_gesture = ""
    display_confidence = 0.0
    
    # The active confidence threshold (80%)
    CONFIDENCE_THRESHOLD = 0.8
    
    # Frame rate calculation variables
    prev_time = 0
    
    print("\n" + "="*80)
    print("                      REAL-TIME TRANSLATOR RUNNING!")
    print("="*80)
    print("Place your hands in the webcam frame and perform one of the trained signs:")
    print("  -> 'hello' | 'thank_you' | 'yes' | 'no' | 'i_love_you'")
    print("Perform the movement continuously. The AI needs a 1-second sequence.")
    print("Press 'q' on your keyboard to exit the program.")
    print("="*80 + "\n")
    
    # 4. Main capturing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam. Exiting...")
            break
            
        # Horizontally mirror the frame so visual coordinates feel natural
        frame = cv2.flip(frame, 1)
        
        # Run hand landmarks detection and draw overlay skeletons
        frame = tracker.find_hands(frame, draw=True)
        
        # Extract flat landmark array of shape (126,)
        landmarks = tracker.extract_landmarks()
        
        # Calculate live FPS (Frames Per Second)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # Check if any hands are visible.
        # If no hand is present, the array is entirely zeros.
        hand_visible = not np.all(landmarks == 0)
        
        if hand_visible:
            # Append current landmarks vector to our rolling frame sequence list
            rolling_sequence.append(landmarks)
            # Cap the rolling list size to the required SEQUENCE_LENGTH (30 frames)
            rolling_sequence = rolling_sequence[-config.SEQUENCE_LENGTH:]
            
            # 5. Run inference once our rolling buffer is fully loaded with 30 frames
            if len(rolling_sequence) == config.SEQUENCE_LENGTH:
                # Reshape from (30, 126) to (1, 30, 126) to create the 3D batch shape Keras expects
                input_data = np.expand_dims(rolling_sequence, axis=0)
                
                # Run the model prediction (verbose=0 disables standard tensorboard logs)
                prediction_probabilities = model.predict(input_data, verbose=0)[0]
                
                # Get the class index with the highest probability score
                predicted_class_idx = np.argmax(prediction_probabilities)
                raw_confidence = prediction_probabilities[predicted_class_idx]
                
                # Append predicted index to the history buffer (capped at 10)
                predictions_history.append(predicted_class_idx)
                predictions_history = predictions_history[-10:]
                
                # 6. Prediction Temporal Smoothing (Anti-Jitter)
                # Find the most common index in the last 10 predictions history
                most_frequent_idx = np.bincount(predictions_history).argmax()
                most_frequent_count = np.bincount(predictions_history)[most_frequent_idx]
                
                # We only trigger a translation if:
                #   1. The model is consistent (predicted at least 6 out of 10 recent frames)
                #   2. The prediction probability is above our 80% confidence threshold
                smoothed_confidence = prediction_probabilities[most_frequent_idx]
                
                if most_frequent_count >= 6 and smoothed_confidence > CONFIDENCE_THRESHOLD:
                    display_gesture = config.GESTURES[most_frequent_idx]
                    display_confidence = smoothed_confidence
                else:
                    # If model is uncertain, fade/clear the display gesture text
                    display_gesture = "Analyzing..."
                    display_confidence = smoothed_confidence
        else:
            # If no hands are visible, immediately clear the buffers and text
            # This prevents old gestures from "sticking" on the screen when you drop your hands!
            rolling_sequence = []
            predictions_history = []
            display_gesture = "No Hand Detected"
            display_confidence = 0.0
            
        # 7. Render Premium HUD Overlays
        # A. Top Charcoal Status Panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (config.FRAME_WIDTH, 40), (20, 20, 20), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        cv2.putText(frame, "ISL Translator AI - Real-time Inference", (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {int(fps)}", (config.FRAME_WIDTH - 90, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # B. Bottom Dashboard Card
        # Draw a beautiful dark card at the bottom to display the translation results
        cv2.rectangle(frame, (0, config.FRAME_HEIGHT - 80), (config.FRAME_WIDTH, config.FRAME_HEIGHT), (26, 21, 18), -1)
        
        # Display large, elegant translation word
        # Glow green if active gesture found, grey if no hand, orange if analyzing
        if display_gesture in config.GESTURES:
            text_color = (0, 255, 0) # Green for active prediction
            gesture_label = display_gesture.replace("_", " ").upper()
        elif display_gesture == "Analyzing...":
            text_color = (0, 165, 255) # Orange for transitioning
            gesture_label = display_gesture
        else:
            text_color = (150, 150, 150) # Grey for inactive/no hand
            gesture_label = display_gesture
            
        cv2.putText(frame, "Translation:", (20, config.FRAME_HEIGHT - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, gesture_label, (20, config.FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
        
        # C. Glowing Confidence Progress Bar
        # Glow color represents active states
        bar_color = (0, 255, 0) if display_confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
        bar_width = int(display_confidence * 250) # Max bar width is 250 pixels
        
        # Draw background progress bar track (dark grey)
        cv2.rectangle(frame, (config.FRAME_WIDTH - 280, config.FRAME_HEIGHT - 35), 
                      (config.FRAME_WIDTH - 30, config.FRAME_HEIGHT - 20), (50, 50, 50), -1)
        # Draw filled progress bar representing confidence percentage
        if bar_width > 0:
            cv2.rectangle(frame, (config.FRAME_WIDTH - 280, config.FRAME_HEIGHT - 35), 
                          (config.FRAME_WIDTH - 280 + bar_width, config.FRAME_HEIGHT - 20), bar_color, -1)
            
        # Draw confidence numeric text above progress bar
        confidence_text = f"Confidence: {int(display_confidence * 100)}%"
        cv2.putText(frame, confidence_text, (config.FRAME_WIDTH - 280, config.FRAME_HEIGHT - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Draw active escape instructions
        cv2.putText(frame, "Press 'Q' to Quit", (15, config.FRAME_HEIGHT - 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Show frame in graphical window
        cv2.imshow("ISL Real-time Translator", frame)
        
        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Shutting down real-time translation...")
            break
            
    # Clean up and release camera
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("[INFO] Real-time translation stopped. Camera released successfully.")

if __name__ == "__main__":
    run_realtime_prediction()
