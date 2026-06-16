# 🤟 Sign Link AI

**Real-Time Indian Sign Language Translator** — using a webcam, MediaPipe, and a deep learning LSTM model. No special hardware required.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sighn-link-ai.streamlit.app)

---

## ✨ Features

- 🎥 **Real-time** gesture recognition via webcam
- 🧠 **LSTM Neural Network** trained on real hand data
- ✋ **MediaPipe** for hand landmark detection (21 points per hand)
- 📱 Works in any browser — no installation needed
- 🌐 No internet required for inference (runs locally)

## 🤚 Recognised Gestures

| Gesture | Sign |
|---------|------|
| 🤚 Hello | Salute → sweep outward |
| 👍 Yes | Thumbs-up → nod up/down |
| 👉 No | Index finger → wag sideways |
| 🙏 Thank You | Flat hand at chin → push forward |
| 🤟 I Love You | Thumb + index + pinky → hold still |

---

## 🏗️ Architecture

```
Webcam
  └─► MediaPipe → 21 landmarks × 3 axes × 2 hands = 126 values/frame
        └─► Normalize (wrist anchor + scale) + Velocity features → 252 values/frame
              └─► 30-frame rolling buffer → LSTM → Softmax → Gesture label
```

### Model
- **Input:** `(30, 252)` — 30 frames × 252 features (position + velocity)
- **LSTM Layer 1:** 64 units
- **LSTM Layer 2:** 64 units
- **Dense:** 128 neurons (ReLU) + BatchNorm
- **Output:** 5 classes (Softmax)
- **Accuracy:** ~80% on real-world test data

---

## 🚀 Run Locally

```bash
# Clone
git clone https://github.com/zorawar238/sighn-link-ai.git
cd sighn-link-ai

# Install
pip install -r requirements.txt

# Run
streamlit run streamlit_app/app.py
```

## 🗂️ Project Structure

```
sighn-link-ai/
├── streamlit_app/
│   ├── app.py              ← Deployment web app (WebRTC)
│   └── streamlit_main.py   ← Local app (OpenCV)
├── scripts/
│   ├── collect_real_data.py ← Record gestures via webcam
│   ├── preprocess.py        ← Build dataset from recordings
│   └── train_model.py       ← Train the LSTM model
├── app/
│   └── hand_tracker.py      ← MediaPipe wrapper
├── models/
│   └── isl_lstm_model.keras ← Trained model
├── config.py                ← All settings
├── requirements.txt
└── RECORD_GESTURES.bat      ← Windows quick-launcher
```

---

## 🛠️ Tech Stack

| Technology | Role |
|-----------|------|
| Python | Core language |
| MediaPipe (Google) | Hand landmark detection |
| TensorFlow / Keras | LSTM model training & inference |
| OpenCV | Local webcam capture |
| Streamlit | Web UI |
| streamlit-webrtc | Browser webcam for deployment |

---

## 📈 Training Your Own Model

1. Record gestures: double-click `RECORD_GESTURES.bat`
2. Preprocess: `python scripts/preprocess.py`
3. Train: `python scripts/train_model.py`

Data is saved to `AppData\Local\isl_gesture_data\` (avoids cloud sync issues).

---

## 🔮 Future Work

- Expand to full ISL alphabet (A–Z)
- Text-to-speech output
- Mobile app (Android)
- Bidirectional translation (speech → sign animation)
- Multi-language support (ASL, BSL)

---

## 👤 Author

**zorawar238** · [GitHub](https://github.com/zorawar238/sighn-link-ai)

> *Built to bridge communication for 63 million deaf/mute people in India.*
