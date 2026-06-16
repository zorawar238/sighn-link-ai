# ==============================================================================
# PREPROCESSING SCRIPT - Indian Sign Language Translator AI
# ==============================================================================
# Purpose: This script loads the 4,500 individual saved hand landmark files,
# combines them into 30-frame sequence blocks (shape: 150, 30, 126), converts
# labels into one-hot encoded binary vectors, shuffles the data, splits it
# into 90% Training and 10% Testing sets, and saves the final arrays to disk.
#
# How to run:
#   python scripts/preprocess.py
# ==============================================================================

import numpy as np
import os
import sys

# Add the parent folder to the system path so Python can find the config
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

def preprocess_data():
    print("[INFO] Starting gesture dataset preprocessing...")
    
    # 1. Define target folder for preprocessed dataset
    DATASET_DIR = os.path.join(config.BASE_DIR, "dataset")
    os.makedirs(DATASET_DIR, exist_ok=True)
    
    # 2. Build a label mapping dictionary
    # Maps text labels (e.g. 'hello') to integers (e.g. 0)
    # result: {'hello': 0, 'thank_you': 1, 'yes': 2, 'no': 3, 'i_love_you': 4}
    label_map = {label: num for num, label in enumerate(config.GESTURES)}
    print(f"[INFO] Mapping labels to integers: {label_map}")
    
    # Lists to hold compiled sequences and their corresponding numerical labels
    sequences = []
    labels = []
    
    # 3. Compile all 4,500 frames into sequences
    print("[INFO] Compiling individual frame coordinate files...")
    
    # Loop through each word (gesture) — auto-detects real vs synthetic data
    loaded = 0
    skipped = 0
    for gesture in config.GESTURES:
        # Auto-detect which sequences actually have data (handles 30 real OR 200 synthetic)
        gesture_dir = os.path.join(config.DATA_PATH, gesture)
        available_seqs = [
            seq_id for seq_id in range(config.NO_SEQUENCES)
            if os.path.exists(os.path.join(gesture_dir, str(seq_id), "0.npy"))
        ]

        if not available_seqs:
            print(f"[WARNING] No sequences found for '{gesture}', skipping.")
            skipped += 1
            continue

        for sequence in available_seqs:
            window = []
            for frame_num in range(config.SEQUENCE_LENGTH):
                npy_path = os.path.join(config.DATA_PATH, gesture, str(sequence), f"{frame_num}.npy")
                if os.path.exists(npy_path):
                    res = np.load(npy_path)
                    window.append(res)
                else:
                    print(f"[WARNING] Missing frame: {npy_path}. Padding with zeros.")
                    window.append(np.zeros(config.NUM_FEATURES))

            window_normalized = normalize_sequence(window)
            sequences.append(window_normalized)
            labels.append(label_map[gesture])
            loaded += 1

        print(f"  [OK] '{gesture}': {len(available_seqs)} sequences")

    print(f"[INFO] Total sequences loaded: {loaded} ({skipped} gestures skipped)")

    # Convert lists to NumPy arrays
    # X holds features: Shape will be (150, 30, 126)
    #   150: Total sequences (5 gestures * 30 sequences)
    #    30: Frame sequence length
    #   126: Number of 3D hand landmarks coordinates (21 landmarks * 3 coords * 2 hands)
    X = np.array(sequences)  # shape: (N, 30, 126)

    # --- Add Velocity Features (frame-to-frame difference) ---
    # velocity[t] = position[t] - position[t-1]; velocity[0] = zeros
    # This makes hand MOTION explicit so the LSTM can learn dynamic gestures.
    velocity = np.zeros_like(X)
    velocity[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]  # shape: (N, 30, 126)
    X = np.concatenate([X, velocity], axis=2)          # shape: (N, 30, 252)

    # y_raw holds raw labels: Shape will be (N,)
    y_raw = np.array(labels)

    print(f"[INFO] Successfully loaded and stacked raw data.")
    print(f"[INFO] Feature Array (X) shape: {X.shape}  [position + velocity]")
    print(f"[INFO] Raw Label Array (y_raw) shape: {y_raw.shape}")
    
    # 4. One-Hot Encoding
    # We convert raw integers like 0, 1, 2 into binary vectors:
    # 0 -> [1, 0, 0, 0, 0] | 1 -> [0, 1, 0, 0, 0] etc.
    print("[INFO] Performing One-Hot Encoding on labels...")
    num_classes = len(config.GESTURES)
    y = np.zeros((y_raw.size, num_classes))
    y[np.arange(y_raw.size), y_raw] = 1
    print(f"[INFO] One-Hot Label Array (y) shape: {y.shape}")
    
    # 5. Shuffling the Dataset
    # Since our raw data is sequentially loaded (all 'hello' first, then all 'yes'),
    # we MUST shuffle it so the neural network gets a mix during training.
    print("[INFO] Shuffling dataset randomly...")
    # Set a fixed random seed so that every time you run this script,
    # the shuffle is identical (reproducible research).
    np.random.seed(42)
    
    # Generate random indices from 0 to 149
    shuffled_indices = np.arange(X.shape[0])
    np.random.shuffle(shuffled_indices)
    
    # Re-order our arrays using the shuffled indices
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    
    # 6. Split into Training (90%) and Testing (10%) sets
    # Training set = 135 sequences | Testing set = 15 sequences
    print("[INFO] Splitting dataset (90% Train / 10% Test)...")
    split_index = int(0.9 * X.shape[0])
    
    X_train, X_test = X_shuffled[:split_index], X_shuffled[split_index:]
    y_train, y_test = y_shuffled[:split_index], y_shuffled[split_index:]
    
    # 7. Save the resulting consolidated arrays to the 'dataset/' folder
    print(f"[INFO] Saving preprocessed arrays to folder: {DATASET_DIR}...")
    np.save(os.path.join(DATASET_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATASET_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATASET_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATASET_DIR, "y_test.npy"), y_test)
    
    print("\n" + "="*80)
    print("                 PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Training features (X_train) shape: {X_train.shape} -> Used for model fitting.")
    print(f"Training labels   (y_train) shape: {y_train.shape} -> Match targets for training.")
    print(f"Testing features  (X_test)  shape: {X_test.shape}  -> Used for unseen evaluation.")
    print(f"Testing labels    (y_test)  shape: {y_test.shape}  -> Used to score final accuracy.")
    print("="*80 + "\n")

if __name__ == "__main__":
    preprocess_data()
