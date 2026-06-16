# ==============================================================================
# SENTENCE TRANSLATOR & SPEECH ENGINE - Indian Sign Language Translator AI
# ==============================================================================
# Purpose: This script is the complete multi-word translator and voice output pipeline!
# It aggregates your gestures into full sentences, implements duplicate-prevention
# locks with hand-drop resets, and uses background threading to speak sentences
# out loud via pyttsx3 without freezing the live camera stream.
#
# How to run:
#   python scripts/translate_speech.py
#
# Keyboard controls:
#   [Spacebar]  - Speaks the entire compiled sentence out loud.
#   [Backspace] - Deletes the last word from the sentence.
#   ['c' / 'C'] - Clears the entire sentence.
#   ['q' / 'Q'] - Quits the program.
# ==============================================================================

import cv2
import numpy as np
import os
import sys
import threading
import time

# Disable annoying TensorFlow startup warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pyttsx3

# Add the parent folder to the system path so Python can find the config & app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def normalize_sequence(sequence):
    """
    Applies mathematical translation and scale normalization to a 30-frame sequence.
    Subtracts the active hand's wrist coordinate on frame 0 from all frames (translation-invariant displacement)
    and scales the hand coordinates relative to the starting wrist-to-knuckle distance (scale-invariant).
    """
    seq = np.array(sequence).copy()
    seq_reshaped = seq.reshape((30, 2, 21, 3))
    
    for h in range(2):
        hand_data = seq_reshaped[:, h]
        is_active = np.sum(np.abs(hand_data)) > 1e-5
        
        if is_active:
            # wrist coordinate on frame 0
            wrist_0 = hand_data[0, 0].copy()
            # wrist-to-middle-MCP knuckle distance on frame 0
            scale = np.linalg.norm(hand_data[0, 9] - hand_data[0, 0])
            
            if scale > 0:
                for t in range(30):
                    seq_reshaped[t, h] = seq_reshaped[t, h] - wrist_0
                    seq_reshaped[t, h] = seq_reshaped[t, h] / scale
            else:
                for t in range(30):
                    seq_reshaped[t, h] = seq_reshaped[t, h] - wrist_0
                    
    return seq_reshaped.reshape((30, 126))
from app.hand_tracker import HandTracker

def speak_text_async(text):
    """
    Speaks the given text out loud asynchronously using a background thread.
    This prevents the main OpenCV camera loop from freezing while speaking!
    """
    if not text.strip():
        return
        
    def speak():
        try:
            # We initialize pyttsx3 INSIDE the background thread.
            # This is critical on Windows to prevent COM multi-threading conflicts.
            engine = pyttsx3.init()
            # Set speaking rate (160 is a comfortable, standard speed)
            engine.setProperty('rate', 160)
            
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[WARNING] Async Text-To-Speech engine encountered an error: {e}")
            
    # Spawn a background daemon thread to run the speech engine.
    # daemon=True ensures the thread terminates instantly when the main window closes.
    speech_thread = threading.Thread(target=speak, daemon=True)
    speech_thread.start()

def run_translation_speech():
    print("[INFO] Starting ISL Sentence Translator & Voice Engine...")
    
    # 1. Load the trained LSTM model
    MODEL_FILE_PATH = os.path.join(config.MODEL_PATH, "isl_lstm_model.keras")
    if not os.path.exists(MODEL_FILE_PATH):
        print(f"[ERROR] Could not find the model file at: {MODEL_FILE_PATH}")
        sys.exit(1)
        
    model = tf.keras.models.load_model(MODEL_FILE_PATH)
    print("[INFO] Trained Keras model loaded successfully!")
    
    # 2. Initialize camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    # Initialize the HandTracker
    tracker = HandTracker()
    
    # 3. State Buffers
    rolling_sequence = []   # Buffers last 30 frames for LSTM shape (30, 126)
    predictions_history = []# Buffers last 10 predictions for smoothing
    
    # Sentence compilation variables
    sentence = []           # List of compiled words in the sentence
    last_word = None        # Lock variable to prevent duplicate double-appends
    
    # Real-time HUD states
    display_gesture = ""
    display_confidence = 0.0
    CONFIDENCE_THRESHOLD = 0.8
    prev_time = 0
    
    print("\n" + "="*80)
    print("                 SENTENCE TRANSLATOR & VOICE ENGINE RUNNING!")
    print("="*80)
    print("Perform sign gestures dynamically in front of the camera.")
    print("  -> 'hello' | 'thank_you' | 'yes' | 'no' | 'i_love_you'")
    print("\nKeyboard Hotkeys:")
    print("  [Spacebar]  - Speaks the entire compiled sentence out loud.")
    print("  [Backspace] - Deletes the last word from your sentence.")
    print("  ['C' Key]   - Clears the entire sentence.")
    print("  ['Q' Key]   - Safely exits the program.")
    print("="*80 + "\n")
    
    # 4. Main capturing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break
            
        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame, draw=True)
        landmarks = tracker.extract_landmarks()
        
        # Calculate live FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # Check if hands are present in frame
        hand_visible = not np.all(landmarks == 0)
        
        if hand_visible:
            rolling_sequence.append(landmarks)
            rolling_sequence = rolling_sequence[-config.SEQUENCE_LENGTH:]
            
            # Dynamic Sequence Padding:
            # If the buffer is not yet full, pad it by repeating the earliest frame.
            # This enables INSTANT, zero-latency predictions the millisecond your hand appears!
            pad_count = config.SEQUENCE_LENGTH - len(rolling_sequence)
            prediction_sequence = [rolling_sequence[0]] * pad_count + rolling_sequence
            
            # Apply Translation and Scale Invariance Normalization to the sequence
            normalized_seq = normalize_sequence(prediction_sequence)
            
            input_data = np.expand_dims(normalized_seq, axis=0)
            prediction_probabilities = model.predict(input_data, verbose=0)[0]
            
            predicted_class_idx = np.argmax(prediction_probabilities)
            raw_confidence = prediction_probabilities[predicted_class_idx]
            
            # Append predicted index to predictions history buffer for temporal smoothing
            predictions_history.append(predicted_class_idx)
            predictions_history = predictions_history[-10:]
            
            # Retrieve most frequent class in predictions history
            most_frequent_idx = np.bincount(predictions_history).argmax()
            most_frequent_count = np.bincount(predictions_history)[most_frequent_idx]
            
            smoothed_confidence = prediction_probabilities[most_frequent_idx]
            
            # 5. Sentence compilation and lock checks
            if most_frequent_count >= 6 and smoothed_confidence > CONFIDENCE_THRESHOLD:
                predicted_word = config.GESTURES[most_frequent_idx]
                
                # Smart Duplicate-Prevention: Only append if it's different from the last word signed
                if predicted_word != last_word:
                    sentence.append(predicted_word)
                    last_word = predicted_word
                    
                    # Micro-audio feedback: immediately speak the newly appended word in the background
                    spoken_word = predicted_word.replace("_", " ")
                    speak_text_async(spoken_word)
                    print(f"[SENTENCE LOG] Appended: '{spoken_word}' | Current Sentence: {' '.join(sentence)}")
                    
                display_gesture = predicted_word
                display_confidence = smoothed_confidence
            else:
                display_gesture = "Analyzing..."
                display_confidence = smoothed_confidence
        else:
            # 6. Hand-Drop Reset Trigger
            # When hands are dropped, we clear local frame buffers.
            # Crucially, we set last_word = None which immediately unlocks the duplicate prevention.
            # This allows the user to sign the same word twice in a row by simply dropping their hands in between!
            rolling_sequence = []
            predictions_history = []
            display_gesture = "No Hand Detected"
            display_confidence = 0.0
            last_word = None
            
        # 7. Listen for keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        
        # Exit on 'q' or 'Q'
        if key == ord('q'):
            print("[INFO] Shutting down translator...")
            break
            
        # Spacebar: Speaks the entire compiled sentence out loud
        elif key == ord(' '):
            if sentence:
                sentence_text = " ".join(sentence).replace("_", " ")
                print(f"[VOICE TRIGGER] Speaking sentence: '{sentence_text}'")
                speak_text_async(sentence_text)
            else:
                print("[WARNING] Sentence is empty. Nothing to speak.")
                speak_text_async("Sentence is empty")
                
        # Backspace: Deletes the last word from the sentence list
        elif key == 8: # ASCII key code for backspace in Windows OpenCV
            if sentence:
                removed = sentence.pop()
                print(f"[SENTENCE LOG] Deleted last word: '{removed.replace('_', ' ')}'")
                
        # 'c' or 'C': Clears the entire sentence
        elif key == ord('c') or key == ord('C'):
            sentence = []
            last_word = None
            print("[SENTENCE LOG] Sentence cleared.")
            speak_text_async("Cleared")
            
        # 8. Render Premium HUD Overlays
        # A. Top Charcoal Bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (config.FRAME_WIDTH, 40), (20, 20, 20), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        cv2.putText(frame, "ISL Sentence Translator & Voice Engine", (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {int(fps)}", (config.FRAME_WIDTH - 90, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # B. Bottom Dashboard Card
        # Draw a beautiful dark card at the bottom to display the sentence results
        cv2.rectangle(frame, (0, config.FRAME_HEIGHT - 120), (config.FRAME_WIDTH, config.FRAME_HEIGHT), (26, 21, 18), -1)
        
        # Display large, elegant translation word
        if display_gesture in config.GESTURES:
            text_color = (0, 255, 0) # Green for active prediction
            gesture_label = display_gesture.replace("_", " ").upper()
        elif display_gesture == "Analyzing...":
            text_color = (0, 165, 255) # Orange for transitioning
            gesture_label = display_gesture
        else:
            text_color = (150, 150, 150) # Grey for inactive/no hand
            gesture_label = display_gesture
            
        cv2.putText(frame, "Current Sign:", (20, config.FRAME_HEIGHT - 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, gesture_label, (20, config.FRAME_HEIGHT - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
        
        # Render confidence progress bar
        bar_color = (0, 255, 0) if display_confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
        bar_width = int(display_confidence * 150) # Max bar width is 150 pixels
        
        cv2.rectangle(frame, (200, config.FRAME_HEIGHT - 85), (350, config.FRAME_HEIGHT - 75), (50, 50, 50), -1)
        if bar_width > 0:
            cv2.rectangle(frame, (200, config.FRAME_HEIGHT - 85), (200 + bar_width, config.FRAME_HEIGHT - 75), bar_color, -1)
        
        confidence_text = f"{int(display_confidence * 100)}%"
        cv2.putText(frame, confidence_text, (360, config.FRAME_HEIGHT - 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Display the compiled sentence
        sentence_label = " ".join(sentence).replace("_", " ").upper()
        if not sentence_label:
            sentence_label = "Waiting for gestures..."
            sentence_color = (100, 100, 100)
        else:
            sentence_color = (255, 255, 255)
            
        cv2.putText(frame, "Compiled Sentence:", (20, config.FRAME_HEIGHT - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, sentence_label, (20, config.FRAME_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, sentence_color, 2, cv2.LINE_AA)
        
        # Draw active hotkey guides in the bottom right corner
        cv2.putText(frame, "[Space: Speak]", (config.FRAME_WIDTH - 230, config.FRAME_HEIGHT - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "[Backspace: Del]", (config.FRAME_WIDTH - 230, config.FRAME_HEIGHT - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "[C Key: Clear]", (config.FRAME_WIDTH - 230, config.FRAME_HEIGHT - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(frame, "Press 'Q' to Quit", (15, config.FRAME_HEIGHT - 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow("ISL Translator & Speech Engine", frame)
        
    # Clean up
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    print("[INFO] Resources released successfully. Translator stopped.")

if __name__ == "__main__":
    run_translation_speech()
