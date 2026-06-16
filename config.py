# ==============================================================================
# CONFIGURATION - Indian Sign Language Translator AI
# ==============================================================================
# This file stores all global constants, settings, and hyperparameters.
# Putting them here prevents "magic numbers" in our code and makes it easy
# to fine-tune the system (e.g., changing frame size or sequence length).

import os

# ------------------------------------------------------------------------------
# 1. PATH CONFIGURATIONS
# ------------------------------------------------------------------------------
# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where we will save recorded sign language sequence landmarks (for training)
# IMPORTANT: Stored OUTSIDE OneDrive to prevent sync from deleting .npy files!
DATA_PATH = r"C:\Users\nisha\AppData\Local\isl_gesture_data"

# Where we will save our trained machine learning models
MODEL_PATH = os.path.join(BASE_DIR, "models")

# Ensure directories exist
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)


# ------------------------------------------------------------------------------
# 2. CAMERA AND VIDEO CONFIGURATIONS
# ------------------------------------------------------------------------------
# Default camera index. On most laptops, '0' is the built-in webcam.
# If you have an external webcam plugged in, you might need to change this to '1' or '2'.
CAMERA_INDEX = 0

# Width and height of the video frame to capture.
# 640x480 is standard, fast to process, and works well on most computers.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ------------------------------------------------------------------------------
# 3. MEDIAPIPE HAND TRACKING CONFIGURATIONS
# ------------------------------------------------------------------------------
# Minimum confidence value ([0.0, 1.0]) from the hand detection model
# for the detection to be considered successful.
MIN_DETECTION_CONFIDENCE = 0.5

# Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model
# to be considered tracked. High confidence reduces jitter.
MIN_TRACKING_CONFIDENCE = 0.5

# Maximum number of hands to detect. We set this to 2 since we use both hands
# for many Indian Sign Language (ISL) gestures.
MAX_HANDS = 2


# ------------------------------------------------------------------------------
# 4. GESTURE RECOGNITION & LSTM MODEL PARAMETERS
# ------------------------------------------------------------------------------
# The list of gestures our AI will recognize.
# You can add more words to this list as you train them!
GESTURES = [
    "hello", "yes", "no", "thank_you", "i_love_you",
]

# The number of consecutive frames we record for a single gesture.
# Since video runs at ~30 frames per second, 30 frames represent ~1 second of movement.
# Our LSTM model will look at a sequence of 30 frames to predict the gesture.
SEQUENCE_LENGTH = 30

# The number of data points per frame.
# MediaPipe detects 21 landmarks per hand. Each landmark has 3 coordinates (X, Y, Z).
# For 2 hands: 21 landmarks * 3 coordinates * 2 hands = 126 coordinates in total.
NUM_FEATURES = 21 * 3 * 2  # 126 raw position features per frame

# Total features fed to the model = position + velocity (frame-to-frame diff)
# This makes motion patterns explicit and easier to learn with limited data.
MODEL_FEATURES = NUM_FEATURES * 2  # 252

# The number of data sequences we will record per gesture during data collection.
# 120 sequences per gesture provides a robust training set of 600 sequences.
NO_SEQUENCES = 200
