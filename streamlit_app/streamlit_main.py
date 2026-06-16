# ==============================================================================
# STREAMLIT WEB APP MAIN - Indian Sign Language Translator AI
# ==============================================================================
# Purpose: This is the premium, web-based graphical dashboard for our ISL app.
# It packages the hand tracker, trained Keras LSTM model, and async speech engine
# into a gorgeous, highly interactive web application.
#
# How to run:
#   streamlit run streamlit_app/streamlit_main.py
# ==============================================================================

import streamlit as st
import cv2
import numpy as np
import os
import sys
import time

# Disable annoying TensorFlow startup warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Resolve parent paths so we can import our modules
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
from scripts.translate_speech import speak_text_async

# ------------------------------------------------------------------------------
# 1. PREMIUM PAGE AND STYLING CONFIGURATIONS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="ISL Translator AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection to deliver a state-of-the-art Dark Mode HUD dashboard
# We use modern typography, smooth gradients, and glassmorphism styling card containers.
st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        * {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Dashboard Title Styling with neat linear gradient */
        .title-container {
            text-align: center;
            padding: 10px 0px 25px 0px;
        }
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00FFCC 0%, #0099FF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0px;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #AAAAAA;
            font-weight: 300;
            margin-top: 5px;
        }
        
        /* Glassmorphic Panel Cards */
        .dashboard-card {
            background-color: #1e1a17;
            border: 1px solid #332924;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        }
        
        .card-header {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: #AAAAAA;
            margin-bottom: 12px;
            font-weight: 600;
        }
        
        /* Glowing neon labels for predicted gestures */
        .gesture-active {
            font-size: 2.2rem;
            font-weight: 800;
            color: #00FFCC;
            text-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
            margin: 0;
        }
        .gesture-transition {
            font-size: 2.2rem;
            font-weight: 800;
            color: #FF9900;
            text-shadow: 0 0 10px rgba(255, 153, 0, 0.3);
            margin: 0;
        }
        .gesture-inactive {
            font-size: 2.2rem;
            font-weight: 800;
            color: #888888;
            margin: 0;
        }
        
        /* Large high-contrast compiled sentence box */
        .sentence-box {
            background-color: #2c2420;
            border-left: 5px solid #0099FF;
            padding: 15px;
            border-radius: 8px;
            font-size: 1.5rem;
            font-weight: 600;
            color: #FFFFFF;
            min-height: 60px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }
        .sentence-placeholder {
            color: #666666;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. MODEL AND RESOURCE CACHING
# ------------------------------------------------------------------------------
# We cache the TensorFlow model load. Caching ensures the massive neural network
# is only read from the disk ONCE on start-up. Without caching, Streamlit would
# reload the 200KB model file on every single button click, freezing the GUI!
@st.cache_resource
def load_isl_deep_model():
    model_path = os.path.join(config.MODEL_PATH, "isl_lstm_model.keras")
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)


# ------------------------------------------------------------------------------
# 3. SESSION STATE INITIALIZATION
# ------------------------------------------------------------------------------
# Streamlit runs from top to bottom on every user action. To prevent variable
# resets (e.g. wiping your sentence list when clicking buttons), we must store
# persistent variables in Streamlit's Session State object.
if 'sentence' not in st.session_state:
    st.session_state.sentence = []
if 'last_word' not in st.session_state:
    st.session_state.last_word = None
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False
if 'last_word_time' not in st.session_state:
    st.session_state.last_word_time = 0.0


# ------------------------------------------------------------------------------
# 4. SIDEBAR CONFIGURATIONS PANEL
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤟 ISL Control Room")
    st.markdown("---")
    
    # Check if the Keras model exists on disk
    model = load_isl_deep_model()
    if model is None:
        st.error("⚠️ Trained Model Not Found!")
        st.info("Please make sure you have run 'python scripts/train_model.py' to compile and save your LSTM network before launching the web app.")
        st.stop()
    else:
        st.success("🤖 LSTM Brain Loaded Successfully!")
        
    st.markdown("### ⚙️ System Settings")
    
    # 1. Camera Input Selector (0 for built-in, 1 or 2 for external webcams)
    camera_idx = st.slider("Camera Input Index", min_value=0, max_value=2, value=config.CAMERA_INDEX, step=1)
    
    # 2. Confidence Threshold Slider (sensitivity of prediction filter)
    confidence_threshold = st.slider("AI Confidence Threshold", min_value=0.30, max_value=0.99, value=0.40, step=0.01)
    
    st.markdown("---")
    
    # 3. Dynamic Stop/Start Controller Buttons
    if not st.session_state.run_camera:
        if st.button("▶️ Start Translator", use_container_width=True, type="primary"):
            st.session_state.run_camera = True
            st.rerun()
    else:
        if st.button("⏹️ Stop Translator", use_container_width=True):
            st.session_state.run_camera = False
            st.rerun()
            
    st.markdown("---")
    st.markdown("### 📝 Keyboard Reference Guide")
    st.markdown("""
    If camera active:
    * Click camera screen and press **`[Space]`** to speak sentence.
    * Press **`[Backspace]`** to delete last word.
    * Press **`[C]`** to clear sentence.
    * Press **`[Q]`** to quit camera feed.
    """)


# ------------------------------------------------------------------------------
# 5. MAIN APP LAYOUT DASHBOARD
# ------------------------------------------------------------------------------
# Page title header block
st.markdown("""
    <div class="title-container">
        <h1 class="main-title">Indian Sign Language Translator AI</h1>
        <p class="subtitle">Real-time Recurrent Deep Neural Network Gesture Translation Dashboard</p>
    </div>
""", unsafe_allow_html=True)

# Build a neat 2-Column responsive dashboard
col_feed, col_dashboard = st.columns([1.1, 0.9])

# A. COLUMN 1: Camera stream viewer
with col_feed:
    st.markdown("### 📹 Live Camera View")
    
    # Empty placeholder card where frames will be rendered dynamically
    camera_placeholder = st.empty()
    
    # If the camera stream is inactive, display a gorgeous, helpful greeting card
    if not st.session_state.run_camera:
        camera_placeholder.markdown("""
            <div style="background-color: #1e1a17; border: 2px dashed #44362f; border-radius: 15px; height: 380px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; color: #888888;">
                <span style="font-size: 5rem; margin-bottom: 10px;">🤟</span>
                <h4 style="color: #DDDDDD; font-weight: 600; margin-bottom: 5px;">Ready to Translate!</h4>
                <p style="font-size: 0.9rem; max-width: 320px; color: #AAAAAA;">Click the green <b>Start Translator</b> button in the sidebar panel to activate your webcam stream.</p>
            </div>
        """, unsafe_allow_html=True)

# B. COLUMN 2: Translation Control Center Dashboard
with col_dashboard:
    st.markdown("### 📊 Translation Dashboard")
    
    # 1. Current gesture card container placeholders
    gesture_card = st.container()
    with gesture_card:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Current Gesture</div>', unsafe_allow_html=True)
        current_gesture_placeholder = st.empty()
        
        # Initial render when camera is off
        current_gesture_placeholder.markdown('<p class="gesture-inactive">Webcam Stopped</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # 2. AI Confidence card container placeholders
    confidence_card = st.container()
    with confidence_card:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">AI Confidence Score</div>', unsafe_allow_html=True)
        
        confidence_text_placeholder = st.empty()
        confidence_bar_placeholder = st.empty()
        
        # Initial render
        confidence_text_placeholder.markdown("<small style='color: #666;'>0%</small>", unsafe_allow_html=True)
        confidence_bar_placeholder.progress(0.0)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # 3. Compiled sentence box placeholders
    sentence_card = st.container()
    with sentence_card:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Compiled Translation Sentence</div>', unsafe_allow_html=True)
        
        sentence_placeholder = st.empty()
        
        # Initial render
        sentence_placeholder.markdown('<div class="sentence-box"><span class="sentence-placeholder">Waiting for gestures...</span></div>', unsafe_allow_html=True)
        
        # 4. Web interaction buttons (Speak, Delete, Clear)
        col_speak, col_del, col_clear = st.columns(3)
        
        with col_speak:
            if st.button("🔊 Speak Sentence", use_container_width=True, type="secondary"):
                if st.session_state.sentence:
                    sentence_text = " ".join(st.session_state.sentence).replace("_", " ")
                    speak_text_async(sentence_text)
                else:
                    speak_text_async("Sentence is empty")
                    
        with col_del:
            if st.button("🔙 Delete Word", use_container_width=True):
                if st.session_state.sentence:
                    removed = st.session_state.sentence.pop()
                    st.rerun()
                    
        with col_clear:
            if st.button("❌ Clear All", use_container_width=True):
                st.session_state.sentence = []
                st.session_state.last_word = None
                speak_text_async("Cleared")
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 6. ACTIVE STREAM AND PREDICTION LOOP
# ------------------------------------------------------------------------------
# If the user toggles "Start Translator" in the sidebar
if st.session_state.run_camera:
    
    # Initialize the camera capture
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        st.sidebar.error("❌ Could not open webcam at select index!")
        st.session_state.run_camera = False
        st.rerun()
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    # Initialize hand tracker object
    tracker = HandTracker()
    
    # Coordinate and prediction sequence buffers
    rolling_sequence = []
    predictions_history = []       # stores (class_idx, confidence) per frame
    HISTORY_LEN      = 20         # how many frames to look back
    CONSENSUS_NEEDED = 13         # how many must agree (13/20 = 65%)
    WORD_COOLDOWN    = 1.5        # seconds between accepted words
    
    # We display a brief spinner in the sidebar during webcam loading
    st.sidebar.info("📹 Warm-up camera running...")
    
    # Main loop that drives the live stream and predictions
    while st.session_state.run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam. Loop stopped.")
            break
            
        # Initialize default HUD values for this frame to prevent NameError
        display_gesture = "No Hand Detected"
        display_confidence = 0.0
        
        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame, draw=True)
        landmarks = tracker.extract_landmarks()
        
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
            normalized_seq = normalize_sequence(prediction_sequence)  # (30, 126)

            # Add velocity features: frame-to-frame differences (must match preprocess.py)
            velocity = np.zeros_like(normalized_seq)
            velocity[1:] = normalized_seq[1:] - normalized_seq[:-1]
            seq_with_velocity = np.concatenate([normalized_seq, velocity], axis=1)  # (30, 252)

            # Shape batch size 1: (1, 30, 252)
            input_data = np.expand_dims(seq_with_velocity, axis=0)
            prediction_probabilities = model.predict(input_data, verbose=0)[0]
            
            predicted_class_idx = np.argmax(prediction_probabilities)
            raw_confidence = prediction_probabilities[predicted_class_idx]
            
            predictions_history.append((predicted_class_idx, float(raw_confidence)))
            predictions_history = predictions_history[-HISTORY_LEN:]

            # --- Robust smoothing ---
            # 1. Find the most frequent class in history
            class_votes = [c for c, _ in predictions_history]
            counts = np.bincount(class_votes, minlength=len(config.GESTURES))
            most_frequent_idx   = int(np.argmax(counts))
            most_frequent_count = int(counts[most_frequent_idx])

            # 2. Average confidence for THAT class across history (not just current frame)
            class_confidences = [conf for cls, conf in predictions_history
                                  if cls == most_frequent_idx]
            smoothed_confidence = float(np.mean(class_confidences)) if class_confidences else 0.0

            # 3. Accept only when consensus + confidence + cooldown all pass
            now = time.time()
            cooldown_ok = (now - st.session_state.last_word_time) >= WORD_COOLDOWN

            if (most_frequent_count >= CONSENSUS_NEEDED
                    and smoothed_confidence > confidence_threshold
                    and cooldown_ok):
                predicted_word = config.GESTURES[most_frequent_idx]

                if predicted_word != st.session_state.last_word:
                    st.session_state.sentence.append(predicted_word)
                    st.session_state.last_word = predicted_word
                    st.session_state.last_word_time = now
                    spoken_word = predicted_word.replace("_", " ")
                    speak_text_async(spoken_word)

                display_gesture = predicted_word
                display_confidence = smoothed_confidence
            else:
                display_gesture = "Analyzing..."
                display_confidence = smoothed_confidence
        else:
            # Drop Hand Reset Trigger
            rolling_sequence = []
            predictions_history = []
            display_gesture = "No Hand Detected"
            display_confidence = 0.0
            st.session_state.last_word = None
            
        # --- RENDER TO STREAMLIT UI COMPONENTS ---
        # 1. Render camera feed frame
        # Convert BGR (OpenCV) to RGB (Streamlit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(frame_rgb, use_container_width=True)
        
        # 2. Render current gesture panel card
        if display_gesture in config.GESTURES:
            card_class = "gesture-active"
            gesture_label = display_gesture.replace("_", " ").upper()
        elif display_gesture == "Analyzing...":
            card_class = "gesture-transition"
            gesture_label = display_gesture
        else:
            card_class = "gesture-inactive"
            gesture_label = display_gesture
            
        current_gesture_placeholder.markdown(f'<p class="{card_class}">{gesture_label}</p>', unsafe_allow_html=True)
        
        # 3. Render AI confidence dashboard
        confidence_text_placeholder.markdown(f"<small style='color: #DDDDDD;'>Confidence Score: <b>{int(display_confidence * 100)}%</b></small>", unsafe_allow_html=True)
        confidence_bar_placeholder.progress(float(display_confidence))
        
        # 4. Render compiled translation sentence box
        sentence_label = " ".join(st.session_state.sentence).replace("_", " ").upper()
        if not sentence_label:
            sentence_placeholder.markdown('<div class="sentence-box"><span class="sentence-placeholder">Waiting for gestures...</span></div>', unsafe_allow_html=True)
        else:
            sentence_placeholder.markdown(f'<div class="sentence-box">{sentence_label}</div>', unsafe_allow_html=True)
            
        # Keyboard inputs listener (Spacebar, Backspace, C, Q keys)
        # Note: Streamlit runs inside browser. Local keypresses inside OpenCV window (if active) are also handled.
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            st.session_state.run_camera = False
            st.rerun()
        elif key == ord(' '):
            if st.session_state.sentence:
                sentence_text = " ".join(st.session_state.sentence).replace("_", " ")
                speak_text_async(sentence_text)
        elif key == 8: # Backspace
            if st.session_state.sentence:
                st.session_state.sentence.pop()
                st.rerun()
        elif key == ord('c') or key == ord('C'):
            st.session_state.sentence = []
            st.session_state.last_word = None
            speak_text_async("Cleared")
            st.rerun()
            
        # Yield brief execution slice (keeps the browser highly responsive and stops CPU pegging!)
        time.sleep(0.01)
        
    # --- CLEAN UP PROCEDURES ---
    cap.release()
    tracker.close()
    cv2.destroyAllWindows()
    st.sidebar.success("⏹️ Camera released successfully.")
