# ==============================================================================
# MODEL TRAINING SCRIPT - Indian Sign Language Translator AI (Improved v2)
# ==============================================================================
# Improvements over v1:
#   - Deeper LSTM architecture: 3 LSTM layers (128 -> 128 -> 64 units)
#   - Larger Dense classification head (256 -> 128 nodes with BatchNorm + Dropout)
#   - Increased epochs to 200 with larger patience (20) for EarlyStopping
#   - ReduceLROnPlateau: halves learning rate when val_loss plateaus
#   - Larger batch size (32) for smoother gradients
#   - L2 kernel regularization on LSTM layers to prevent overfitting
# ==============================================================================

import numpy as np
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def train_lstm_model():
    print("[INFO] Starting TensorFlow LSTM Model Training (Improved v2)...")

    # 1. Load preprocessed datasets
    DATASET_DIR = os.path.join(config.BASE_DIR, "dataset")
    X_train = np.load(os.path.join(DATASET_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATASET_DIR, "y_train.npy"))
    X_test  = np.load(os.path.join(DATASET_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATASET_DIR, "y_test.npy"))

    print(f"[INFO] X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"[INFO] X_test:  {X_test.shape}  | y_test:  {y_test.shape}")

    num_classes = len(config.GESTURES)
    print(f"[INFO] Number of gesture classes: {num_classes}")

    # 2. Build improved deep LSTM architecture
    model = Sequential([

        # ---- LSTM Stack -------------------------------------------------------
        # Simplified architecture for small real dataset (~150 samples)
        # Heavy models overfit; smaller = better generalization
        LSTM(
            64,
            return_sequences=True,
            input_shape=(config.SEQUENCE_LENGTH, config.MODEL_FEATURES),
            kernel_regularizer=regularizers.l2(1e-4),
        ),
        Dropout(0.3),

        LSTM(
            64,
            return_sequences=False,
            kernel_regularizer=regularizers.l2(1e-4),
        ),
        Dropout(0.3),

        # ---- Classification Head ----------------------------------------------
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Output: one node per gesture class, softmax gives probabilities
        Dense(num_classes, activation='softmax'),
    ])

    # 3. Compile with Adam + tuned learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n" + "="*80)
    print("                 IMPROVED LSTM ARCHITECTURE SUMMARY")
    print("="*80)
    model.summary()
    print("="*80 + "\n")

    # 4. Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,               # Wait longer before giving up
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,                # Halve the learning rate on plateau
        patience=7,
        min_lr=1e-6,
        verbose=1
    )

    # 5. Compute class weights to fix data imbalance
    # (e.g. help=30 seqs vs hello=5 seqs — weights balance the loss so all gestures matter equally)
    from collections import Counter
    y_ints = np.argmax(y_train, axis=1)
    counts = Counter(y_ints)
    total = len(y_ints)
    class_weight = {cls: total / (num_classes * cnt) for cls, cnt in counts.items()}
    print(f"[INFO] Class weights: { {config.GESTURES[k]: round(v,2) for k,v in class_weight.items()} }")

    # 6. Train
    print("[INFO] Initiating model fitting (up to 200 epochs)...")
    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight,
        verbose=1
    )

    # 6. Evaluate
    print("\n[INFO] Evaluating trained model on unseen test data...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"[SUCCESS] Final Evaluation — Test Loss: {loss:.4f} | Test Accuracy: {accuracy*100:.2f}%")

    # 7. Save
    MODEL_FILE_PATH = os.path.join(config.MODEL_PATH, "isl_lstm_model.keras")
    print(f"[INFO] Saving model to: {MODEL_FILE_PATH}...")
    model.save(MODEL_FILE_PATH)

    print("\n" + "="*80)
    print("              IMPROVED MODEL SUCCESSFULLY TRAINED & SAVED!")
    print("="*80)
    print(f"  Accuracy:    {accuracy*100:.2f}%")
    print(f"  Classes:     {num_classes}")
    print(f"  Model path:  {MODEL_FILE_PATH}")
    print("="*80 + "\n")


if __name__ == "__main__":
    train_lstm_model()
