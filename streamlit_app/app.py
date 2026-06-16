import streamlit as st
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PIL import Image
import os
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sign Link AI",
    page_icon="🤟",
    layout="centered",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin-bottom: 20px;
}
.hero h1 { color: #ffd700; font-size: 2.2rem; font-weight: 800; margin: 0; }
.hero p  { color: #a0b8d8; font-size: 1rem; margin-top: 8px; margin-bottom: 0; }

.result-box {
    background: linear-gradient(135deg, #1a1a2e, #0f3460);
    border: 2px solid #ffd700;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}
.result-word { color: #ffd700; font-size: 3rem; font-weight: 800; }
.result-conf { color: #a0c0e8; font-size: 0.95rem; margin-top: 6px; }

.waiting-box {
    background: #0f1a2e;
    border: 2px dashed #2a4060;
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
    color: #4a6080;
    font-size: 1.1rem;
}

.gesture-grid {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 14px 0;
}
.gchip {
    background: #1a1a2e;
    border: 1px solid #2a3a5e;
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
    color: #a0b8d8;
    font-size: 0.8rem;
    min-width: 80px;
}
.gchip span { display: block; font-size: 1.5rem; margin-bottom: 3px; }

.tip {
    background: #0a1628;
    border-left: 4px solid #ffd700;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    color: #a0b8d8;
    font-size: 0.88rem;
    margin: 10px 0;
}

.prob-bar-wrap { margin: 4px 0; }
.prob-label { color: #a0b8d8; font-size: 0.85rem; margin-bottom: 2px; }

div[data-testid="stButton"] button {
    background: linear-gradient(90deg, #1a1a2e, #0f3460) !important;
    color: #ffd700 !important;
    border: 1px solid #ffd700 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    width: 100% !important;
}
div[data-testid="stButton"] button:hover {
    background: #ffd700 !important;
    color: #1a1a2e !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
GESTURES  = ["hello", "yes", "no", "thank_you", "i_love_you"]
EMOJI     = {"hello":"🤚","yes":"👍","no":"👉","thank_you":"🙏","i_love_you":"🤟"}
INSTRUCTIONS = {
    "hello":     "Salute → sweep hand outward",
    "yes":       "Thumbs-up → nod up and down",
    "no":        "Index finger → wag side to side",
    "thank_you": "Flat hand at chin → push forward",
    "i_love_you":"Thumb + index + pinky → hold still",
}
SEQ_LEN   = 30
THRESHOLD = 0.40
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "isl_lstm_model.keras")

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.abspath(MODEL_PATH)
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        model_complexity=0,
    )

model = load_model()
hands = load_hands()
mp_drawing = mp.solutions.drawing_utils

# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_landmarks(img_array):
    rgb = img_array[:, :, :3]
    result = hands.process(rgb)
    frame  = np.zeros(126)
    found  = False
    if result.multi_hand_landmarks:
        found = True
        for i, lm in enumerate(result.multi_hand_landmarks[:2]):
            base = i * 63
            for j, pt in enumerate(lm.landmark):
                frame[base + j*3 : base + j*3+3] = [pt.x, pt.y, pt.z]
    return frame, found

def normalize_seq(seq):
    arr    = np.array(seq, dtype=np.float32)
    origin = arr[0, :3].copy()
    arr[:, 0::3] -= origin[0]
    arr[:, 1::3] -= origin[1]
    arr[:, 2::3] -= origin[2]
    scale  = max(np.linalg.norm(arr[0, 3:6]), 1e-6)
    arr   /= scale
    return arr

def build_features(positions):
    pos     = np.array(positions, dtype=np.float32)
    vel     = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    return np.concatenate([pos, vel], axis=1)

def predict_gesture(img_array):
    lm, found = extract_landmarks(img_array)
    if not found:
        return None, 0.0, None

    # Build 30-frame sequence from single capture
    # (static shape works perfectly; motion gestures predict by shape)
    seq = [lm] * SEQ_LEN
    norm = normalize_seq(seq)
    feat = build_features(norm)
    inp  = np.expand_dims(feat, 0).astype(np.float32)
    pred = model.predict(inp, verbose=0)[0]
    idx  = int(np.argmax(pred))
    conf = float(pred[idx])
    return GESTURES[idx] if conf >= THRESHOLD else None, conf, pred

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1>🤟 Sign Link AI</h1>
  <p>Indian Sign Language Translator &nbsp;·&nbsp; Real-Time AI &nbsp;·&nbsp; No hardware needed</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file not found. Please check `models/isl_lstm_model.keras` is in the repo.")
    st.stop()

# Gesture reference
st.markdown("""
<div class="gesture-grid">
  <div class="gchip"><span>🤚</span>Hello</div>
  <div class="gchip"><span>👍</span>Yes</div>
  <div class="gchip"><span>👉</span>No</div>
  <div class="gchip"><span>🙏</span>Thank You</div>
  <div class="gchip"><span>🤟</span>I Love You</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="tip">💡 <b>How to use:</b> Click <b>"Take Photo"</b> below → show your hand gesture clearly → click the shutter → see the result instantly!</div>', unsafe_allow_html=True)

# Camera input
img_file = st.camera_input("📷 Show your gesture here", label_visibility="collapsed")

if img_file is not None:
    img = Image.open(img_file).convert("RGB")
    img_array = np.array(img)

    with st.spinner("🧠 Analysing gesture..."):
        gesture, conf, probs = predict_gesture(img_array)

    if gesture:
        emoji = EMOJI[gesture]
        label = gesture.replace("_", " ").title()
        instr = INSTRUCTIONS[gesture]
        st.markdown(f"""
        <div class="result-box">
          <div class="result-word">{emoji} {label}</div>
          <div class="result-conf">Confidence: {conf*100:.1f}% &nbsp;·&nbsp; {instr}</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown("**All gesture probabilities:**")
        for i, g in enumerate(GESTURES):
            p = float(probs[i]) * 100
            em = EMOJI[g]
            lbl = g.replace("_", " ").title()
            st.markdown(f"<div class='prob-label'>{em} {lbl}</div>", unsafe_allow_html=True)
            st.progress(min(p / 100, 1.0))
    else:
        st.markdown("""
        <div class="waiting-box">
          🖐️ No hand detected or confidence too low<br>
          <small>Make sure your hand is clearly visible and well-lit</small>
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="waiting-box">
      📷 Click the camera above to take a photo of your gesture
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("""
<p style="text-align:center; color:#4a6080; font-size:0.82rem;">
Sign Link AI &nbsp;·&nbsp; Built with MediaPipe + LSTM &nbsp;·&nbsp;
<a href="https://github.com/zorawar238/sighn-link-ai" style="color:#ffd700; text-decoration:none;">⭐ GitHub</a>
</p>
""", unsafe_allow_html=True)
