# ==============================================================================
# SYNTHETIC DATA GENERATOR - ISL Translator AI (Redesigned v3)
# ==============================================================================
# Core Design Principle:
#   Every gesture has a UNIQUE hand shape (different finger count/config).
#   This means the LSTM can distinguish gestures primarily by SHAPE, not just
#   by motion trajectory — making it far more robust to real-world variation.
#
# Shape uniqueness strategy:
#   - 0 fingers:  closed fist                 → yes, sorry, angry
#   - 1 finger:   index only                  → no, one, help
#   - 2 fingers:  index+middle (V)            → two, sister
#   - 3 fingers:  index+middle+ring           → three, water
#   - 4 fingers:  all except thumb            → four
#   - 5 spread:   all wide                    → five, scared
#   - ILY:        thumb+index+pinky           → i_love_you  ← most unique
#   - L-shape:    thumb+index                 → brother, six
#   - thumb only: thumbs up                   → help(base), ten
#   - ok shape:   thumb+index circle          → nine
#   - pinch 2:    thumb+middle                → eight
#   - pinch 3:    thumb+ring                  → seven
#   - hang-loose: thumb+pinky                 → six
#   - open:       all fingers                 → hello, thank_you, please,
#                                               good_morning, good_night,
#                                               happy, sad, surprised,
#                                               mother, father, food, friend
#     (open-hand group: distinguished by MOTION AMPLITUDE + DIRECTION)
# ==============================================================================

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from scripts.collect_data import setup_directories

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def _zeros():
    return np.zeros((21, 3))

def _wrist(x=0.50, y=0.72, z=0.0):
    h = np.zeros((21, 3))
    h[0]  = [x,        y,        z      ]
    h[1]  = [x-0.05,  y-0.05,  -0.02   ]  # Thumb CMC
    h[5]  = [x-0.05,  y-0.20,  -0.03   ]  # Index  MCP
    h[9]  = [x+0.00,  y-0.22,  -0.03   ]  # Middle MCP
    h[13] = [x+0.05,  y-0.20,  -0.03   ]  # Ring   MCP
    h[17] = [x+0.09,  y-0.17,  -0.03   ]  # Pinky  MCP
    return h

def _ext(h, mcp, d=(0,-1,0), L=0.28):
    d = np.array(d, float); d /= np.linalg.norm(d)
    b = h[mcp]
    h[mcp+1]=b+d*L*0.33; h[mcp+2]=b+d*L*0.66; h[mcp+3]=b+d*L
    return h

def _fold(h, mcp):
    b = h[mcp]
    h[mcp+1]=b+[0.01, 0.05, 0.02]
    h[mcp+2]=b+[0.01, 0.09, 0.04]
    h[mcp+3]=b+[0.00, 0.10, 0.05]
    return h

def _thumb_up(h):   # thumb extends upward
    h[2]=h[1]+[-0.05,-0.07,-0.02]; h[3]=h[2]+[-0.03,-0.06,-0.01]; h[4]=h[3]+[-0.02,-0.05,-0.01]
    return h

def _thumb_side(h): # thumb extends sideways (out)
    h[2]=h[1]+[-0.07,-0.02,-0.02]; h[3]=h[2]+[-0.06,-0.01,-0.01]; h[4]=h[3]+[-0.05,-0.01,-0.01]
    return h

def _thumb_fold(h):
    h[2]=h[1]+[0.02, 0.03, 0.02]; h[3]=h[2]+[0.02, 0.03, 0.02]; h[4]=h[3]+[0.02, 0.02, 0.01]
    return h

# ==============================================================================
# HAND SHAPE LIBRARY  (each returns a (21,3) array)
# ==============================================================================

def sh_open():
    """All 5 fingers fully extended — most common ISL base."""
    h=_wrist(); h=_thumb_up(h)
    h=_ext(h, 5,( 0.00,-1,-0.10),0.30); h=_ext(h, 9,( 0.00,-1,-0.10),0.32)
    h=_ext(h,13,( 0.00,-1,-0.10),0.30); h=_ext(h,17,( 0.00,-1,-0.10),0.26)
    return h

def sh_fist():
    """Closed fist, thumb folded ACROSS palm (NOT thumbs-up) — yes, sorry, angry.
    Key: thumb is HIDDEN inside fist, clearly different from sh_thumbs_up."""
    h=_wrist(); h=_thumb_fold(h)   # thumb tucked in, NOT sticking up
    h=_fold(h,5); h=_fold(h,9); h=_fold(h,13); h=_fold(h,17)
    return h

def sh_index():
    """Index pointing UPWARD — one, help (static)."""
    h=_wrist(); h=_thumb_fold(h)
    h=_ext(h,5,(0,-1,-0.10),0.30); h=_fold(h,9); h=_fold(h,13); h=_fold(h,17)
    return h

def sh_index_sideways():
    """Index pointing SIDEWAYS (horizontal) — no.
    Visually very different from sh_index (which points UP).
    ISL 'no' = index wagging side-to-side while pointing outward."""
    h=_wrist(); h=_thumb_fold(h)
    # Index extends to the side (-X direction = toward camera-left)
    h=_ext(h,5,(-1, 0.10,-0.10),0.28)
    h=_fold(h,9); h=_fold(h,13); h=_fold(h,17)
    return h

def sh_two():
    """Index + middle (V/peace) — two, sister."""
    h=_wrist(); h=_thumb_fold(h)
    h=_ext(h,5,(-0.04,-1,-0.10),0.30); h=_ext(h,9,(0.04,-1,-0.10),0.30)
    h=_fold(h,13); h=_fold(h,17)
    return h

def sh_three():
    """Index + middle + ring — three, water."""
    h=_wrist(); h=_thumb_fold(h)
    h=_ext(h,5,(-0.02,-1,-0.10),0.30); h=_ext(h,9,(0,-1,-0.10),0.30)
    h=_ext(h,13,(0.02,-1,-0.10),0.28); h=_fold(h,17)
    return h

def sh_four():
    """Four fingers (no thumb) — four."""
    h=_wrist(); h=_thumb_fold(h)
    h=_ext(h,5,(-0.02,-1,-0.10),0.30); h=_ext(h,9,(0,-1,-0.10),0.32)
    h=_ext(h,13,(0.02,-1,-0.10),0.30); h=_ext(h,17,(0.04,-1,-0.10),0.26)
    return h

def sh_five():
    """All 5 fingers spread wide — five, scared."""
    h=_wrist(); h=_thumb_up(h)
    h=_ext(h,5,(-0.14,-1,-0.10),0.30); h=_ext(h,9,(-0.05,-1,-0.10),0.32)
    h=_ext(h,13,(0.05,-1,-0.10),0.30); h=_ext(h,17,(0.14,-1,-0.10),0.26)
    return h

def sh_ily():
    """Thumb+index+pinky (ILY 🤟) — i_love_you."""
    h=_wrist(); h=_thumb_up(h)
    h=_ext(h,5,(0,-1,-0.10),0.30); h=_fold(h,9); h=_fold(h,13)
    h=_ext(h,17,(0,-1,-0.10),0.24)
    return h

def sh_hang_loose():
    """Thumb+pinky only (🤙) — six."""
    h=_wrist(); h=_thumb_side(h)
    h=_fold(h,5); h=_fold(h,9); h=_fold(h,13)
    h=_ext(h,17,(0,-1,-0.10),0.24)
    return h

def sh_thumb_ring_pinky():
    """Thumb+ring+pinky — seven."""
    h=_wrist(); h=_thumb_up(h)
    h=_fold(h,5); h=_fold(h,9)
    h=_ext(h,13,(0.02,-1,-0.10),0.28); h=_ext(h,17,(0.05,-1,-0.10),0.24)
    return h

def sh_thumb_middle_pinch():
    """Thumb tip pinches middle finger, index+ring+pinky out — eight."""
    h=_wrist()
    # Thumb bends toward middle
    h[2]=h[1]+[-0.02,-0.02, 0.00]; h[3]=h[2]+[0.01, 0.01, 0.02]; h[4]=h[3]+[0.02, 0.02, 0.03]
    h=_ext(h,5,(0,-1,-0.10),0.30)
    # Middle bent to thumb
    h[10]=h[9]+[0.00,-0.06,-0.01]; h[11]=h[10]+[0.01, 0.03, 0.03]; h[12]=h[4].copy()
    h=_ext(h,13,(0.02,-1,-0.10),0.28); h=_ext(h,17,(0.05,-1,-0.10),0.24)
    return h

def sh_ok():
    """Index bent forming circle with thumb (OK/9) — nine."""
    h=_wrist()
    h[2]=h[1]+[-0.03,-0.05,-0.01]; h[3]=h[2]+[-0.01,-0.01, 0.01]; h[4]=h[3]+[0.01, 0.01, 0.02]
    # Index hooks to touch thumb
    h[6]=h[5]+[0.00,-0.08,-0.01]; h[7]=h[6]+[0.02, 0.04, 0.02]; h[8]=h[4]+[0.01, 0.00, 0.01]
    h=_ext(h,9,(0,-1,-0.10),0.30); h=_ext(h,13,(0.02,-1,-0.10),0.28)
    h=_ext(h,17,(0.05,-1,-0.10),0.24)
    return h

def sh_thumbs_up():
    """Fist + thumb up — ten, help base."""
    h=_wrist(); h=_thumb_up(h)
    h=_fold(h,5); h=_fold(h,9); h=_fold(h,13); h=_fold(h,17)
    return h

def sh_l():
    """Thumb+index L-shape — brother."""
    h=_wrist(); h=_thumb_side(h)
    h=_ext(h,5,(0,-1,-0.10),0.30); h=_fold(h,9); h=_fold(h,13); h=_fold(h,17)
    return h

def sh_o():
    """All fingertips meet thumb (O-shape) — food."""
    h=_wrist()
    cx,cy=0.50,0.62
    for tip,ang in zip([4,8,12,16,20],[210,290,310,330,350]):
        r=np.deg2rad(ang); h[tip]=[cx+0.045*np.cos(r), cy+0.045*np.sin(r),-0.03]
    h[2]=h[1]+[-0.02,-0.04,-0.01]; h[3]=h[2]+[-0.01,-0.03,-0.01]
    h[6]=h[5]+[0.00,-0.07,-0.01];  h[7]=h[6]+[0.00,-0.05,-0.01]
    h[10]=h[9]+[0.00,-0.07,-0.01]; h[11]=h[10]+[0.00,-0.05,-0.01]
    h[14]=h[13]+[0.00,-0.07,-0.01];h[15]=h[14]+[0.00,-0.05,-0.01]
    h[18]=h[17]+[0.00,-0.06,-0.01];h[19]=h[18]+[0.00,-0.04,-0.01]
    return h

def sh_hook():
    """Index hooked/bent — friend."""
    h=_wrist(); h=_thumb_fold(h)
    h[6]=h[5]+[0.00,-0.10,-0.01]; h[7]=h[6]+[0.03, 0.04, 0.02]; h[8]=h[7]+[0.03, 0.03, 0.02]
    h=_fold(h,9); h=_fold(h,13); h=_fold(h,17)
    return h

def sh_claw():
    """All fingers clawed/bent — angry."""
    h=_wrist()
    h[2]=h[1]+[-0.03,-0.04,-0.01]; h[3]=h[2]+[-0.01,-0.02, 0.01]; h[4]=h[3]+[0.01, 0.01, 0.02]
    for mcp in [5,9,13,17]:
        b=h[mcp]
        h[mcp+1]=b+[0.00,-0.09,-0.01]; h[mcp+2]=b+[0.01,-0.05, 0.04]; h[mcp+3]=b+[0.02,-0.01, 0.06]
    return h

# ==============================================================================
# SHAPE MAP
# ==============================================================================
SHAPE_MAP = {
    # Existing
    "hello":        sh_open,      # open  + BIG X wave
    "thank_you":    sh_open,      # open  + strong forward-down arc
    "yes":          sh_thumbs_up, # thumbs-up + Y nod (different motion from ten's X shake)
    "no":           sh_index_sideways,  # index sideways + Y wag (distinct from 'one')
    "i_love_you":   sh_ily,       # ILY   + static (most unique)
    # Common words
    "please":       sh_open,      # open  + CLOCKWISE chest circle
    "sorry":        sh_fist,      # fist  + CCW chest circle
    "good_morning": sh_open,      # open  + LARGE upward sweep
    "good_night":   sh_open,      # open  + LARGE downward sweep
    "help":         sh_thumbs_up, # thumbs-up fist + lift up
    "water":        sh_three,     # 3-finger W + tap chin
    "food":         sh_o,         # O-hand + tap mouth
    # Numbers (all STATIC — shape is the signal, not motion)
    "one":          sh_index,     # index only
    "two":          sh_two,       # V/peace
    "three":        sh_three,     # 3 fingers
    "four":         sh_four,      # 4 fingers
    "five":         sh_five,      # spread 5
    "six":          sh_hang_loose,# thumb+pinky 🤙
    "seven":        sh_thumb_ring_pinky,
    "eight":        sh_thumb_middle_pinch,
    "nine":         sh_ok,        # OK circle
    "ten":          sh_thumbs_up, # thumbs-up + shake
    # Emotions
    "happy":        sh_open,      # open  + upward circles
    "sad":          sh_open,      # open  + downward slide (tears)
    "angry":        sh_claw,      # claw  + Z-pull
    "scared":       sh_five,      # spread-5 + pulse out
    "surprised":    sh_open,      # open  + fast rise + Z push
    # Family
    "mother":       sh_open,      # open  + tap CHIN
    "father":       sh_open,      # open  + tap FOREHEAD
    "sister":       sh_two,       # V-hand + slide cheek down
    "brother":      sh_l,         # L-hand + forehead to chin sweep
    "friend":       sh_hook,      # hooked-index + side shake
}

# ==============================================================================
# MOTION PHYSICS
# Open-hand gestures are separated by: direction, amplitude, frequency, axis
# ==============================================================================
def apply_motion(gesture, hand, t, seq):
    h = hand.copy()
    T = t / 29.0
    ph = seq * 0.20   # phase variation per sequence

    # ---- EXISTING -------------------------------------------------------
    if gesture == "hello":
        # ISL: large wave at face level
        h[:,0] += 0.18 * np.sin(t * 2*np.pi/8 + ph)

    elif gesture == "thank_you":
        # ISL: chin → forward-outward — DISTINCTIVE forward Z motion
        h[:,1] -= 0.08           # chin level
        h[:,2] -= 0.28 * T       # strong forward push
        h[:,1] += 0.06 * T       # slight drop

    elif gesture == "yes":
        # ISL: thumbs-up fist nods strongly UP-DOWN (Y axis)
        # Motion axis: Y only — very different from ten (X axis)
        h[:,1] += 0.18 * np.sin(t * 2*np.pi/8 + ph)

    elif gesture == "no":
        # ISL: index points sideways and wags UP-DOWN (Y axis)
        # Large amplitude, fast — unmistakeable even with noise
        h[:,1] += 0.16 * np.sin(t * 2*np.pi/5 + ph)   # Y wag
        h[:,0] += 0.04 * np.sin(t * 2*np.pi/7 + ph)   # slight X drift

    elif gesture == "i_love_you":
        # Static ILY — tiny Z sway only
        h[:,2] += 0.03 * np.sin(t * 2*np.pi/22 + ph)

    # ---- COMMON WORDS ---------------------------------------------------
    elif gesture == "please":
        # CW circle — LARGER than hello's wave
        h[:,0] += 0.09 * np.sin(t * 2*np.pi/13 + ph)
        h[:,1] += 0.09 * np.cos(t * 2*np.pi/13 + ph)

    elif gesture == "sorry":
        # CCW circle (fist) — opposite direction to please
        h[:,0] -= 0.07 * np.sin(t * 2*np.pi/13 + ph)
        h[:,1] += 0.07 * np.cos(t * 2*np.pi/13 + ph)

    elif gesture == "good_morning":
        # LARGE upward sweep (sun rising) — dominant Y motion
        h[:,1] -= 0.35 * T
        h[:,0] += 0.05 * T

    elif gesture == "good_night":
        # LARGE downward sweep (sun setting) — opposite of morning
        h[:,1] += 0.35 * T
        h[:,0] -= 0.05 * T

    elif gesture == "help":
        # Thumbs-up lifts upward strongly
        h[:,1] -= 0.28 * T
        h[:,2] -= 0.08 * T

    elif gesture == "water":
        # W-hand (3 fingers) taps chin — fast double tap (2 cycles)
        h[:,1] -= 0.10           # chin level
        h[:,1] += 0.08 * np.sin(t * 2*np.pi/6 + ph)

    elif gesture == "food":
        # O-hand moves STRONGLY toward mouth (forward Z + upward Y)
        # Very distinct from yes (which stays in place and nods)
        h[:,1] -= 0.22 * T       # strong rise toward mouth
        h[:,2] -= 0.20 * T       # strong forward Z push

    # ---- NUMBERS (static — tiny micro-motions only) ---------------------
    elif gesture == "one":
        pass   # Pure static index

    elif gesture == "two":
        h[:,0] += 0.02 * np.sin(t * 2*np.pi/28 + ph)   # tiny sway

    elif gesture == "three":
        h[:,2] -= 0.02 * T   # tiny forward nudge

    elif gesture == "four":
        h[:,1] -= 0.02 * T   # tiny upward

    elif gesture == "five":
        pass   # Static spread hand

    elif gesture == "six":
        pass   # Static hang-loose

    elif gesture == "seven":
        pass   # Static thumb+ring+pinky

    elif gesture == "eight":
        pass   # Static thumb-middle pinch

    elif gesture == "nine":
        pass   # Static OK shape

    elif gesture == "ten":
        # Thumbs-up shakes strongly LEFT-RIGHT (X axis)
        # Motion axis: X only — very different from yes (Y axis)
        h[:,0] += 0.18 * np.sin(t * 2*np.pi/5 + ph)

    # ---- EMOTIONS -------------------------------------------------------
    elif gesture == "happy":
        # Open hand — circles upward on chest, net upward drift
        h[:,0] += 0.06 * np.sin(t * 2*np.pi/10 + ph)
        h[:,1] += 0.06 * np.cos(t * 2*np.pi/10 + ph)
        h[:,1] -= 0.14 * T      # upward drift separates from please (no drift)

    elif gesture == "sad":
        # Open hand — steady downward (tears), plus Z pull-back
        h[:,1] -= 0.10          # start near eye level
        h[:,1] += 0.32 * T      # strong downward (most distinctive)
        h[:,2] += 0.08 * T

    elif gesture == "angry":
        # Claw hands — Z punch toward camera
        h[:,2] -= 0.18 * np.sin(t * 2*np.pi/6 + ph)
        h[:,1] += 0.05 * np.sin(t * 2*np.pi/4 + ph)

    elif gesture == "scared":
        # Spread hands — pulse outward
        scale = 1.0 + 0.20 * np.abs(np.sin(t * 2*np.pi/11 + ph))
        h[:,0] = (h[:,0]-0.5)*scale+0.5
        h[:,1] = (h[:,1]-0.65)*scale+0.65

    elif gesture == "surprised":
        # Open hands — shoot upward + forward quickly
        h[:,1] -= 0.28 * T
        h[:,2] -= 0.15 * T

    # ---- FAMILY ---------------------------------------------------------
    elif gesture == "mother":
        # Open hand at CHIN level — tap twice (Y oscillate, lower position)
        h[:,1] -= 0.20          # chin — clearly below neutral
        h[:,1] += 0.07 * np.sin(t * 2*np.pi/7 + ph)

    elif gesture == "father":
        # Open hand at FOREHEAD level — tap twice (same rhythm, HIGH position)
        h[:,1] -= 0.40          # forehead — clearly above chin
        h[:,1] += 0.07 * np.sin(t * 2*np.pi/7 + ph)

    elif gesture == "sister":
        # V-hand slides DOWN cheek
        h[:,1] += 0.22 * T
        h[:,0] += 0.05 * T

    elif gesture == "brother":
        # L-hand starts HIGH, sweeps DOWN
        h[:,1] -= 0.30*(1.0-T)

    elif gesture == "friend":
        # Hooked index — side-to-side shake + Y undulation
        h[:,0] += 0.12 * np.sin(t * 2*np.pi/8 + ph)
        h[:,1] += 0.04 * np.sin(t * 2*np.pi/5 + ph)

    return h

# ==============================================================================
# MAIN GENERATOR
# ==============================================================================
def generate_synthetic_dataset():
    print("[INFO] Generating ISL dataset (v3 — shape-first design)...")
    print(f"[INFO] Gestures: {len(config.GESTURES)}")
    setup_directories()

    for gesture in config.GESTURES:
        if gesture not in SHAPE_MAP:
            print(f"[WARNING] No shape for '{gesture}', skipping."); continue
        print(f"[GENERATING]: '{gesture.upper()}'")
        shape_fn = SHAPE_MAP[gesture]

        for seq in range(config.NO_SEQUENCES):
            x_off   = np.random.uniform(-0.20, 0.20)
            y_off   = np.random.uniform(-0.18, 0.12)
            sc      = np.random.uniform(0.82, 1.18)  # hand-size variation

            for f in range(config.SEQUENCE_LENGTH):
                left  = _zeros()
                right = shape_fn()
                right = apply_motion(gesture, right, f, seq)
                right[:,0] += x_off
                right[:,1] += y_off
                right[:,0]  = (right[:,0]-0.5)*sc+0.5
                right[:,1]  = (right[:,1]-0.5)*sc+0.5

                # Left/right hand augmentation
                if seq % 2 == 0:
                    left = right.copy(); left[:,0] = 1.0-left[:,0]
                    right = _zeros()

                flat  = np.stack([left, right], axis=0).flatten()
                noise = np.random.normal(0, 0.016, flat.shape)
                flat[flat != 0] += noise[flat != 0]

                path = os.path.join(config.DATA_PATH, gesture, str(seq), f"{f}.npy")
                np.save(path, flat)

        print(f"  -> {config.NO_SEQUENCES} sequences for '{gesture}'")

    print("\n" + "="*70)
    print("   ISL DATASET (v3) GENERATED SUCCESSFULLY!")
    print("="*70)
    print(f"  Gestures:  {len(config.GESTURES)}")
    print(f"  Sequences: {len(config.GESTURES)*config.NO_SEQUENCES}")
    print(f"  Frames:    {len(config.GESTURES)*config.NO_SEQUENCES*config.SEQUENCE_LENGTH}")
    print("="*70+"\n")

if __name__ == "__main__":
    generate_synthetic_dataset()
