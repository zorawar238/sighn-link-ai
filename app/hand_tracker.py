# ==============================================================================
# HAND TRACKER MODULE - Indian Sign Language Translator AI
# ==============================================================================
# Purpose: This class encapsulates the Google MediaPipe Hands solution.
# It handles reading frames, detecting hand land marks, drawing skeletons,
# and extracting landmark coordinates into a structured array for our LSTM model.
# ==============================================================================

import cv2
import mediapipe as mp
import numpy as np
import sys
import os

# Ensure config can be loaded by resolving parent directory path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class HandTracker:
    def __init__(self):
        """
        Initializes the MediaPipe Hands model using configurations from config.py.
        """
        # Initialize MediaPipe Hands solution modules
        self.mp_hands = mp.solutions.hands
        
        # mp_draw is a helper module that draws dots (landmarks) and lines (connections)
        # on our camera frames.
        self.mp_draw = mp.solutions.drawing_utils
        
        # Configure drawing styles to look clean and professional
        self.hand_draw_style = self.mp_draw.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=3 # Green nodes
        )
        self.conn_draw_style = self.mp_draw.DrawingSpec(
            color=(255, 255, 255), thickness=2, circle_radius=1 # White connection lines
        )
        
        # Load the Hands model with settings from our config file
        self.hands = self.mp_hands.Hands(
            max_num_hands=config.MAX_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        # Variable to hold the results of our most recent frame processing
        self.results = None

    def find_hands(self, frame, draw=True):
        """
        Processes a BGR camera frame to detect hands and optionally draws the skeleton.
        
        Parameters:
            frame (numpy.ndarray): The current BGR image frame from the webcam.
            draw (bool): If True, draws landmarks and connections directly on the frame.
            
        Returns:
            frame (numpy.ndarray): The processed frame with overlays (if draw=True).
        """
        # OpenCV captures images in BGR format, but MediaPipe requires RGB.
        # So we convert the frame color spacing first.
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect hand landmarks
        self.results = self.hands.process(img_rgb)
        
        # If any hands are detected in the frame
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw connection lines and hand joints (landmarks)
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_lms, 
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.hand_draw_style,
                        connection_drawing_spec=self.conn_draw_style
                    )
        
        return frame

    def extract_landmarks(self):
        """
        Extracts 21 3D coordinate points (X, Y, Z) for both the Left and Right hand.
        
        Returns:
            np.ndarray: A flattened 1D array of shape (126,) containing the landmarks.
                       Structure: [Left Hand landmarks (63 coords), Right Hand landmarks (63 coords)]
                       If a hand is missing, its 63 slots are padded with zeros.
        """
        # We initialize a zero matrix of shape (2, 21, 3)
        # Dimension 0: 2 hands (Index 0 = Left Hand, Index 1 = Right Hand)
        # Dimension 1: 21 landmarks per hand
        # Dimension 2: 3 coordinates (x, y, z) per landmark
        hands_data = np.zeros((2, 21, 3))
        
        # If no hands are detected, return the zero matrix flattened (126 zeros)
        if not self.results or not self.results.multi_hand_landmarks:
            return hands_data.flatten()
            
        # Iterate through all detected hands
        for idx, hand_lms in enumerate(self.results.multi_hand_landmarks):
            # MediaPipe classifies which hand is which (Left/Right)
            # It provides a list of classifications mirroring the order of landmarks.
            handedness = self.results.multi_handedness[idx].classification[0].label
            
            # MediaPipe's handedness refers to the real hand, but because our video
            # is horizontally mirrored in Phase 1, the hands appear flipped to the camera.
            # Index mapping:
            #   If MediaPipe says 'Left' -> It corresponds to Left hand in camera
            #   If MediaPipe says 'Right' -> It corresponds to Right hand in camera
            hand_idx = 0 if handedness == "Left" else 1
            
            # Extract the coordinates of the 21 landmarks for this hand
            for lm_idx, landmark in enumerate(hand_lms.landmark):
                # Save normalized x, y, and z coordinates
                hands_data[hand_idx, lm_idx] = [landmark.x, landmark.y, landmark.z]
                
        # Flatten our (2, 21, 3) matrix into a 1D array of shape (126,)
        return hands_data.flatten()
        
    def close(self):
        """
        Closes the MediaPipe Hands object and releases internal resources.
        """
        self.hands.close()
