# ==============================================================================
# REAL DATA COLLECTOR - ISL Translator AI
# ==============================================================================
# Records 30 real sequences per gesture from YOUR camera.
# Total time: ~25-30 minutes for all 32 gestures.
#
# How to run:
#   python scripts/collect_real_data.py
#
# Controls:
#   SPACE = skip current gesture (if you want to redo it later)
#   Q     = quit early (progress is saved, you can resume)
# ==============================================================================

import cv2
import numpy as np
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from app.hand_tracker import HandTracker

# --- Settings ---
REAL_SEQUENCES   = 30    # sequences per gesture (vs 200 for synthetic)
SEQUENCE_LENGTH  = config.SEQUENCE_LENGTH   # 30 frames
COUNTDOWN_SECS   = 1.5   # seconds between sequences
DATA_PATH        = config.DATA_PATH

# Gesture instructions shown on screen
GESTURE_HINTS = {
    "hello":        "Raise hand to forehead (salute), open outward",
    "thank_you":    "Flat open hand at chin, push FORWARD",
    "yes":          "Thumbs-up fist, NOD UP AND DOWN",
    "no":           "Index pointing SIDEWAYS, wag UP-DOWN",
    "i_love_you":   "Thumb+Index+Pinky out (ILY sign), hold still",
    "please":       "Flat hand, circle CLOCKWISE on chest",
    "sorry":        "Fist, circle COUNTER-CLOCKWISE on chest",
    "good_morning": "Open hand, sweep UPWARD (sun rising)",
    "good_night":   "Open hand, sweep DOWNWARD (sun setting)",
    "help":         "Thumbs-up fist, LIFT UPWARD strongly",
    "water":        "3 fingers (W shape), bring toward mouth",
    "food":         "Fingertips together (O shape), push FORWARD toward mouth",
    "one":          "Index finger ONLY, point up -- hold still",
    "two":          "Index + Middle (V/peace) -- hold still",
    "three":        "3 fingers extended -- hold still",
    "four":         "4 fingers (no thumb) -- hold still",
    "five":         "All 5 fingers SPREAD wide -- hold still",
    "six":          "Thumb + Pinky only (hang-loose shape) -- hold still",
    "seven":        "Thumb + Ring + Pinky extended -- hold still",
    "eight":        "Thumb pinches Middle finger -- hold still",
    "nine":         "Thumb + Index circle (OK sign shape) -- hold still",
    "ten":          "Thumbs-up fist, SHAKE LEFT AND RIGHT",
    "happy":        "Open hand, circle UPWARD on chest",
    "sad":          "Open hand, slide DOWN from eye level",
    "angry":        "Claw fingers, pull TOWARD body + shake",
    "scared":       "All 5 fingers spread, PULSE hands outward",
    "surprised":    "Open hand, shoot UP + FORWARD quickly",
    "mother":       "Open hand, TAP CHIN level twice",
    "father":       "Open hand, TAP FOREHEAD level twice",
    "sister":       "V-hand (2 fingers), slide DOWN along cheek",
    "brother":      "L-shape (thumb+index), sweep DOWN from forehead",
    "friend":       "Hooked index finger, shake SIDE TO SIDE",
}

def setup_real_directories():
    for gesture in config.GESTURES:
        for seq in range(REAL_SEQUENCES):
            os.makedirs(os.path.join(DATA_PATH, gesture, str(seq)), exist_ok=True)

def draw_hud(frame, gesture, hint, seq, total_seqs, frame_num, total_frames, phase):
    h, w = frame.shape[:2]

    # Top bar
    if phase == "recording":
        cv2.rectangle(frame, (0,0), (w,55), (0,0,180), -1)
        status_text = f"RECORDING  Seq {seq+1}/{total_seqs}  Frame {frame_num+1}/{total_frames}"
        cv2.putText(frame, status_text, (12,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (0,0), (w,55), (0,140,0), -1)
        cv2.putText(frame, f"GET READY  Seq {seq+1}/{total_seqs}", (12,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    # Gesture name — big
    cv2.putText(frame, gesture.upper().replace("_"," "), (12,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,255,200), 3, cv2.LINE_AA)

    # Hint box
    box_y = 115
    cv2.rectangle(frame, (0, box_y), (w, box_y+45), (30,30,30), -1)
    cv2.putText(frame, hint, (12, box_y+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,255,200), 1, cv2.LINE_AA)

    # Progress bar (bottom)
    if phase == "recording" and total_frames > 0:
        prog = int(w * frame_num / total_frames)
        cv2.rectangle(frame, (0, h-8), (w, h), (60,60,60), -1)
        cv2.rectangle(frame, (0, h-8), (prog, h), (0,220,120), -1)

    # Controls reminder
    cv2.putText(frame, "SPACE=skip gesture  Q=quit", (12, h-18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1, cv2.LINE_AA)

    return frame

def collect_real_data():
    print("\n" + "="*70)
    print("   ISL REAL DATA COLLECTOR")
    print("="*70)
    print(f"  Gestures : {len(config.GESTURES)}")
    print(f"  Sequences: {REAL_SEQUENCES} per gesture")
    print(f"  Frames   : {SEQUENCE_LENGTH} per sequence (~1 second)")
    print(f"  Est. time: {len(config.GESTURES) * REAL_SEQUENCES * (COUNTDOWN_SECS + 1.2) / 60:.0f} minutes")
    print("="*70)
    print("\n  A camera window will open. For each gesture:")
    print("  1. READ the instruction on screen")
    print("  2. Position your hand")
    print("  3. Perform the gesture when the bar turns RED")
    print("  4. Repeat 30 times — then move to next gesture")
    print("\n  Press SPACE to skip a gesture, Q to quit (progress saved)")
    print("="*70 + "\n")

    setup_real_directories()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    tracker = HandTracker()

    for gesture in config.GESTURES:
        hint = GESTURE_HINTS.get(gesture, "Perform the gesture")

        # --- RESUME: skip if already fully collected ---
        last_seq_path = os.path.join(DATA_PATH, gesture, str(REAL_SEQUENCES - 1))
        last_frame_path = os.path.join(last_seq_path, f"{SEQUENCE_LENGTH - 1}.npy")
        if os.path.exists(last_frame_path):
            print(f"  [SKIP - already done] {gesture}")
            continue

        print(f"\n[GESTURE]: {gesture.upper()}")
        print(f"  Hint: {hint}")
        skip_gesture = False

        for seq in range(REAL_SEQUENCES):
            if skip_gesture:
                break

            # ---------- COUNTDOWN ----------
            countdown_start = time.time()
            while time.time() - countdown_start < COUNTDOWN_SECS:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                frame = tracker.find_hands(frame, draw=True)
                frame = draw_hud(frame, gesture, hint, seq, REAL_SEQUENCES, 0, SEQUENCE_LENGTH, "ready")

                # Big countdown number
                remaining = COUNTDOWN_SECS - (time.time() - countdown_start)
                cv2.putText(frame, f"{remaining:.1f}", (config.FRAME_WIDTH//2-40, config.FRAME_HEIGHT//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,165,255), 5, cv2.LINE_AA)

                cv2.imshow("ISL Data Collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("[INFO] Quit by user — all progress saved.")
                    cap.release(); tracker.close(); cv2.destroyAllWindows(); sys.exit(0)
                if key == ord(' '):
                    print(f"  [SKIP] Gesture '{gesture}' skipped.")
                    skip_gesture = True; break

            if skip_gesture: break

            # ---------- RECORD ----------
            print(f"  Recording seq {seq+1}/{REAL_SEQUENCES}...", end="\r")
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                frame = tracker.find_hands(frame, draw=True)
                landmarks = tracker.extract_landmarks()

                npy_path = os.path.join(DATA_PATH, gesture, str(seq), f"{frame_num}.npy")
                np.save(npy_path, landmarks)

                frame = draw_hud(frame, gesture, hint, seq, REAL_SEQUENCES,
                                 frame_num, SEQUENCE_LENGTH, "recording")
                cv2.imshow("ISL Data Collector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n[INFO] Quit by user — all progress saved.")
                    cap.release(); tracker.close(); cv2.destroyAllWindows(); sys.exit(0)
                if key == ord(' '):
                    skip_gesture = True; break

        if not skip_gesture:
            print(f"  [DONE] {gesture} -- {REAL_SEQUENCES} sequences recorded")
        else:
            print(f"  [SKIP] {gesture} -- using synthetic data for this gesture")

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()

    print("\n" + "="*70)
    print("   REAL DATA COLLECTION COMPLETE!")
    print("="*70)
    print("  Now run:")
    print("    python scripts/preprocess.py")
    print("    python scripts/train_model.py")
    print("="*70 + "\n")

if __name__ == "__main__":
    collect_real_data()
