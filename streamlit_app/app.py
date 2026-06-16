import streamlit as st
import numpy as np
import mediapipe as mp
import tensorflow as tf
import threading
import os
import sys
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sign Link AI",
    page_icon="🤟",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.main { background: #0a0a1a; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin-bottom: 24px;
}
.hero h1 { color: #ffd700; font-size: 2.4rem; font-weight: 800; margin:0; }
.hero p  { color: #a0b8d8; font-size: 1rem; margin-top: 8px; }

.result-box {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    border: 2px solid #ffd700;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 16px 0;
}
.result-word { color: #ffd700; font-size: 2.8rem; font-weight: 800; }
.result-conf { color: #a0c0e8; font-size: 1rem; margin-top: 4px; }

.gesture-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
    margin: 16px 0;
}
.gesture-chip {
    background: #1a1a2e;
    border: 1px solid #0f3460;
    border-radius: 8px;
    padding: 10px 4px;
    text-align: center;
    font-size: 0.75rem;
    color: #a0b8d8;
}
.gesture-chip span { display: block; font-size: 1.4rem; }

.tip-box {
    background: #0f2040;
    border-left: 4px solid #ffd700;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    color: #c0d8f0;
    font-size: 0.9rem;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
GESTURES    = ["hello", "yes", "no", "thank_you", "i_love_you"]
EMOJI       = {"hello":"🤚", "yes":"👍", "no":"👉", "thank_you":"🙏", "i_love_you":"🤟"}
SEQUENCE_LEN = 30
FEATURES     = 252
CONFIDENCE_THRESHOLD = 0.40
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "..", "models", "isl_lstm_model.keras")

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(os.path.abspath(MODEL_PATH))
    except Exception as e:
        st.error(f"Model not found: {e}")
        return None

model = load_model()

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

# ── Helper: extract landmarks ─────────────────────────────────────────────────
def extract_landmarks(image_rgb, hands_detector):
    result = hands_detector.process(image_rgb)
    frame  = np.zeros(126)
    if result.multi_hand_landmarks:
        for i, hand_lm in enumerate(result.multi_hand_landmarks[:2]):
            base = i * 63
            for j, lm in enumerate(hand_lm.landmark):
                frame[base + j*3:base + j*3+3] = [lm.x, lm.y, lm.z]
    return frame, result

# ── Helper: normalize sequence ────────────────────────────────────────────────
def normalize_sequence(seq):
    arr    = np.array(seq)
    origin = arr[0, :3].copy()
    arr[:, 0::3] -= origin[0]
    arr[:, 1::3] -= origin[1]
    arr[:, 2::3] -= origin[2]
    scale  = np.linalg.norm(arr[0, 3:6]) + 1e-6
    arr   /= scale
    return arr

# ── Helper: build velocity ────────────────────────────────────────────────────
def build_features(positions):
    pos      = np.array(positions)
    vel      = np.zeros_like(pos)
    vel[1:]  = pos[1:] - pos[:-1]
    return np.concatenate([pos, vel], axis=1)

# ── Video processor for WebRTC ────────────────────────────────────────────────
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands       = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=0,
        )
        self.sequence    = []
        self.result_text = ""
        self.confidence  = 0.0
        self.cooldown    = 0
        self.consensus   = []
        self.lock        = threading.Lock()

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = img_bgr[:, :, ::-1]  # BGR → RGB

        landmarks, detection = extract_landmarks(img_rgb, self.hands)

        # Draw hand landmarks
        if detection.multi_hand_landmarks:
            for hand_lm in detection.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img_bgr, hand_lm,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

        # Accumulate sequence
        if np.any(landmarks):
            self.sequence.append(landmarks)
        else:
            self.sequence = []

        self.sequence = self.sequence[-SEQUENCE_LEN:]

        # Predict when buffer full
        if len(self.sequence) == SEQUENCE_LEN and model is not None:
            try:
                norm = normalize_sequence(self.sequence)
                feat = build_features(norm)
                inp  = np.expand_dims(feat, axis=0).astype(np.float32)
                pred = model.predict(inp, verbose=0)[0]
                conf = float(np.max(pred))
                idx  = int(np.argmax(pred))

                if conf >= CONFIDENCE_THRESHOLD:
                    gesture = GESTURES[idx]
                    self.consensus.append(gesture)
                    self.consensus = self.consensus[-5:]

                    if self.consensus.count(gesture) >= 3 and self.cooldown == 0:
                        with self.lock:
                            self.result_text = gesture
                            self.confidence  = conf
                            self.cooldown    = 20
                else:
                    self.consensus = []

            except Exception:
                pass

        if self.cooldown > 0:
            self.cooldown -= 1

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# ── RTC config (STUN servers) ─────────────────────────────────────────────────
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1>🤟 Sign Link AI</h1>
  <p>Real-Time Indian Sign Language Translator · No hardware needed</p>
</div>
""", unsafe_allow_html=True)

# Gesture chips
st.markdown("""
<div class="gesture-grid">
  <div class="gesture-chip"><span>🤚</span>Hello</div>
  <div class="gesture-chip"><span>👍</span>Yes</div>
  <div class="gesture-chip"><span>👉</span>No</div>
  <div class="gesture-chip"><span>🙏</span>Thank You</div>
  <div class="gesture-chip"><span>🤟</span>I Love You</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="tip-box">
  💡 Allow camera access when prompted · Hold gesture clearly for 1–2 seconds
</div>
""", unsafe_allow_html=True)

# Result placeholder
result_placeholder = st.empty()

# WebRTC streamer
ctx = webrtc_streamer(
    key="sign-link-ai",
    video_processor_factory=GestureProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Show live result
if ctx.video_processor:
    with ctx.video_processor.lock:
        gesture = ctx.video_processor.result_text
        conf    = ctx.video_processor.confidence

    if gesture:
        emoji = EMOJI.get(gesture, "🤟")
        label = gesture.replace("_", " ").title()
        result_placeholder.markdown(f"""
        <div class="result-box">
          <div class="result-word">{emoji} {label}</div>
          <div class="result-conf">Confidence: {conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        result_placeholder.markdown("""
        <div class="result-box">
          <div class="result-word" style="color:#4a6080;">Waiting for gesture...</div>
          <div class="result-conf">Show your hand clearly in the camera</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.markdown("""
<p style="text-align:center; color:#4a6080; font-size:0.85rem;">
  Sign Link AI · Built with MediaPipe + LSTM · 
  <a href="https://github.com/zorawar238/sighn-link-ai" style="color:#ffd700;">GitHub</a>
</p>
""", unsafe_allow_html=True)
